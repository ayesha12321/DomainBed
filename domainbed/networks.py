import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet50
from domainbed.lib import wide_resnet
import copy
import gc
import timm


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)

class SpatialCosineRouter(nn.Module):
    def __init__(self, num_experts, embed_dim, top_k=2, temperature=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.temperature = temperature
        # The learnable codebook for experts (Visual Attributes)
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, embed_dim))
        nn.init.normal_(self.expert_embeddings, std=0.02)

    def forward(self, x):
        # x shape: [Batch * H * W, Channels] -> e.g., [4704, 512]
        x_norm = F.normalize(x, p=2, dim=-1)
        e_norm = F.normalize(self.expert_embeddings, p=2, dim=-1) # Shape: [4, 512]
        
        logits = F.linear(x_norm, e_norm) / self.temperature
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Re-normalize weights among the chosen top-k experts
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Calculate Load Balancing Auxiliary Losses (Importance and Load)
        importance = probs.sum(dim=0)
        load = torch.zeros_like(importance)
        load.scatter_add_(0, topk_indices.view(-1), torch.ones_like(topk_indices.view(-1), dtype=load.dtype))
        
        loss_imp = (importance.std() / (importance.mean() + 1e-10)) ** 2
        loss_load = (load.std() / (load.mean() + 1e-10)) ** 2
        aux_loss = loss_imp + loss_load
        
        return topk_probs, topk_indices, aux_loss, probs
class SpatialMoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.router = SpatialCosineRouter(num_experts, in_channels, top_k)
        
        # Drop-in replacement for any conv. Can handle ResNet50 (1x1) or ResNet18 (3x3).
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            for _ in range(num_experts)
        ])
        self.aux_loss = 0.0
        self.last_top1_routing = None 
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) # View pixels as "tokens"
        
        topk_probs, topk_indices, aux_loss, _ = self.router(x_flat)
        self.aux_loss = aux_loss
        self.last_top1_routing = topk_indices[:, 0].view(B, H, W).detach() # Save for visualization
        
        out = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x) 
            
            # Identify which pixels go to this expert
            expert_mask = (topk_indices == i) 
            
            expert_weights = torch.zeros(B * H * W, device=x.device)
            expert_weights[expert_mask.any(dim=-1)] = topk_probs[expert_mask]
            expert_weights = expert_weights.view(B, 1, H, W)
            
            out += expert_out * expert_weights
            
        return out

def inject_spatial_moe_resnet(model, num_experts=4, top_k=2):
    """Replaces the final convs in the last two blocks of ResNet layer4 with SpatialMoE"""
    for name, module in model.named_modules():
        if name == 'layer4':
            blocks = list(module.children())
            replace_indices = [-2, -1] if len(blocks) >= 2 else [-1]
            
            for idx in replace_indices:
                block = blocks[idx]
                if hasattr(block, 'conv3'): 
                    # ResNet-50 Bottleneck (Replacing 1x1 conv)
                    in_c = block.conv3.in_channels
                    out_c = block.conv3.out_channels
                    bias = block.conv3.bias is not None
                    
                    moe_layer = SpatialMoELayer(in_c, out_c, num_experts, top_k, bias=bias)
                    for exp in moe_layer.experts:
                        exp.weight.data = block.conv3.weight.data.clone()
                        if bias: exp.bias.data = block.conv3.bias.data.clone()
                    block.conv3 = moe_layer
                    
                elif hasattr(block, 'conv2'): 
                    # ResNet-18 BasicBlock (Replacing 3x3 conv)
                    in_c = block.conv2.in_channels
                    out_c = block.conv2.out_channels
                    bias = block.conv2.bias is not None
                    
                    moe_layer = SpatialMoELayer(in_c, out_c, num_experts, top_k, kernel_size=3, padding=1, bias=bias)
                    for exp in moe_layer.experts:
                        exp.weight.data = block.conv2.weight.data.clone()
                        if bias: exp.bias.data = block.conv2.bias.data.clone()
                    block.conv2 = moe_layer



class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            print("Loading Resnet18")
            if hparams['resnet18_pretrained']:
                print(">> resnet-18 loading pretrained")
                self.network = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            else:
                print(">> resnet-18 loading untrained")
            self.network = torchvision.models.resnet18(weights = None)
            self.n_outputs = 512
        else:
            if hparams['resnet50_pretrained']:
                print(">> resnet-50 loading pretrained")
                self.network = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
                self.n_outputs = 2048
                
            else:
                print(">> resnet-50 loaded weights")
                local_weights_path = "/content/resnet50-0676ba61.pth"  
                state_dict = torch.load(local_weights_path)

                self.network = resnet50(weights=None)  
                self.network.load_state_dict(state_dict)
                self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            print(">>> [DEBUG] TIMM ResNet50 created successfully")

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        # >>> GMOE INJECTION <<<
        if hparams.get('use_gmoe', False):
            print(">>> [DEBUG] Injecting Spatial MoE into ResNet layer4")
            inject_spatial_moe_resnet(
                self.network, 
                num_experts=hparams.get('gmoe_num_experts', 4), 
                top_k=hparams.get('gmoe_top_k', 2)
            )

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.activation = nn.Identity() 

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        out = self.activation(self.dropout(self.network(x)))
        
        # >>> COLLECT MOE LOAD BALANCING LOSSES <<<
        self.moe_aux_loss = 0.0
        for m in self.network.modules():
            if isinstance(m, SpatialMoELayer):
                self.moe_aux_loss += m.aux_loss
                
        return out

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class SmallViT(nn.Module):
    """ViT-Small feature extractor via timm"""
    def __init__(self, input_shape, hparams):
        super().__init__()
        # Create the timm ViT-Small model (use pretrained weights by default)
        self.network = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.n_outputs = self.network.embed_dim


        # Ensure input has 3 channels (RGB)
        if input_shape[0] != 3:
            raise RuntimeError("ViT-Small requires 3-channel input")


        # Optionally freeze the model for feature extraction
        if not hparams.get('vit_finetune', True):
            for p in self.network.parameters():
                p.requires_grad = False


        # Optional dropout for regularization
        self.dropout = nn.Dropout(hparams.get('vit_dropout', 0.1))


        print(">>> [DEBUG] vit small created successfully")


    def forward(self, x):
        # Extract features from the vision transformer
        x = self.network.forward_features(x)  # [B, 197, D]
        x = x[:, 0]                           # CLS token → [B, D]
        x = self.dropout(x)
        return x

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    # print(f">>> [DEBUG] Featurizer called with input_shape={input_shape}, hparams={hparams}")

    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                return SmallViT(input_shape, hparams)
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError
    # print(">>> [DEBUG] Featurizer returning model:", model.__class__.__name__)



def Classifier(in_features, out_features, is_nonlinear=False):
    # print(">>> [DEBUG] classifier returning model:")
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

class CosineRouter(nn.Module):
    def __init__(self, input_dim, num_experts, temp=1.0):
        super(CosineRouter, self).__init__()
        # One prototype vector per expert
        self.prototypes = nn.Parameter(torch.randn(num_experts, input_dim))
        self.temp = temp
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Cosine similarity: (a . b) / (|a| * |b|)
        x_norm = F.normalize(x, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        logits = torch.matmul(x_norm, proto_norm.t()) / self.temp
        return self.softmax(logits)

class MultiHeadNetwork(torch.nn.Module):
    """A network with a shared featurizer, multiple classification heads, and optional Router."""
    def __init__(self, input_shape, num_classes, num_heads, hparams):
        super().__init__()
        self.featurizer = Featurizer(input_shape, hparams)
        self.heads = torch.nn.ModuleList([
            Classifier(
                self.featurizer.n_outputs,
                num_classes,
                hparams['nonlinear_classifier'])
            for _ in range(num_heads)
        ])
        self.num_heads = num_heads
        
        # --- ROUTER LOGIC ---
        self.use_cosine_router = hparams.get('use_cosine_router', False)
        if self.use_cosine_router:
            self.router = CosineRouter(
                input_dim=self.featurizer.n_outputs,
                num_experts=num_heads,
                temp=hparams.get('router_temp', 1.0)
            )

    def forward(self, x, head_idx=None):
        features = self.featurizer(x)
        
        if head_idx is not None:
            # Training a specific head (Stage 1 & 2)
            if not (0 <= head_idx < self.num_heads):
                 raise ValueError(f"Invalid head_idx: {head_idx}.")
            return self.heads[head_idx](features)
        else:
            # Inference or Router Training (Stage 3)
            outputs = [head(features) for head in self.heads]
            
            if self.use_cosine_router:
                routing_weights = self.router(features)
                return features, outputs, routing_weights
            
            return features, outputs