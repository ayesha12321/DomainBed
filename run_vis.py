import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os

# Import your modified networks
from domainbed import networks

def visualize_expert_routing(model_path, image_path, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Checkpoint
    saved = torch.load(model_path, map_location=device)
    
    # 2. Clean the dictionary
    clean_dict = {}
    for k, v in saved['model_dict'].items():
        if k.startswith('network.0.'):
            clean_dict[k.replace('network.0.', '')] = v
        elif k.startswith('featurizer.'):
            clean_dict[k.replace('featurizer.', '')] = v

    # 3. Aggressively Auto-Detect the number of experts
    detected_num_experts = 4 # Fallback default
    found_key = False
    
    for key, tensor in clean_dict.items():
        if "router.expert_embeddings" in key:
            detected_num_experts = tensor.shape[0]
            print(f">>> [SUCCESS] Found key '{key}' with shape {tensor.shape}.")
            print(f">>> [SUCCESS] Auto-setting num_experts = {detected_num_experts}")
            found_key = True
            break
            
    if not found_key:
        print(">>> [ERROR] Could NOT find 'router.expert_embeddings' in checkpoint! Defaulting to 4.")

    # 4. Define hparams dynamically based on detection
    hparams = {
        'resnet18': True,
        'resnet18_pretrained': False,
        'resnet50_pretrained': False,
        'resnet50_augmix': False,
        'freeze_bn': False,
        'resnet_dropout': 0.0,
        'vit': False,
        'use_gmoe': True,
        'gmoe_num_experts': detected_num_experts, # <--- CRITICAL: Pass detected size here
        'gmoe_top_k': 2,
    }
    
    # 5. Build and Load Model
    featurizer = networks.Featurizer((3, 224, 224), hparams).to(device)
    featurizer.load_state_dict(clean_dict, strict=False)
    featurizer.eval()

    # Process Image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Forward Pass
    with torch.no_grad():
        _ = featurizer(img_tensor)

    moe_layers = [m for m in featurizer.modules() if isinstance(m, networks.SpatialMoELayer)]
    
    if not moe_layers:
        print("No SpatialMoELayers found! Did hparams['use_gmoe'] trigger?")
        return

    last_moe = moe_layers[-1]
    routing_map = last_moe.last_top1_routing[0].cpu().numpy() # Shape: (7, 7)
    
    routing_map_resized = cv2.resize(routing_map, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Expanded color palette to support up to 16 experts securely
    colors = [
        [255, 0, 0],   # 0: Red
        [0, 255, 0],   # 1: Green
        [0, 0, 255],   # 2: Blue
        [255, 255, 0], # 3: Yellow
        [255, 0, 255], # 4: Magenta
        [0, 255, 255], # 5: Cyan
        [255, 165, 0], # 6: Orange
        [128, 0, 128], # 7: Purple
        [0, 128, 128], # 8: Teal
        [128, 128, 0], # 9: Olive
        [255, 192, 203],# 10: Pink
        [165, 42, 42]  # 11: Brown
    ]
    
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(detected_num_experts):
        mask = (routing_map_resized == i)
        color_idx = i % len(colors) # Prevents index errors if > 12 experts
        overlay[mask] = colors[color_idx]

    original_np = np.array(img_resized)
    blended = cv2.addWeighted(original_np, 0.5, overlay, 0.5, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_np)
    axes[0].set_title(f"Original: {os.path.basename(image_path)}")
    axes[0].axis('off')
    
    axes[1].imshow(blended)
    axes[1].set_title("Expert Spatial Assignment (Top-1)")
    axes[1].axis('off')
    
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=np.array(colors[i % len(colors)])/255.0, label=f'Expert {i}') for i in range(detected_num_experts)]
    axes[1].legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"SUCCESS! Verification saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    data_dir = 'domainbed/data/PACS'
    
    # ---------------------------------------------------------
    # NEW: Define the master output directory
    master_output_dir = 'expert_maps_output'
    # ---------------------------------------------------------

    # Mapping the exact folder letters to human-readable domain names
    domains = {
        'A': 'art',
        'C': 'cartoon',
        'P': 'photo',
        'S': 'sketch'
    }
    
    # PACS standard class mapping
    pacs_classes = {
        '0': 'dog',
        '1': 'elephant',
        '2': 'giraffe',
        '3': 'guitar',
        '4': 'horse',
        '5': 'house',
        '6': 'person'
    }
    
    # Loop through models 0, 1, 2, and 3
    for test_domain in range(4):
        print(f"\n=======================================================")
        print(f"--- Starting batch processing for model {test_domain} ---")
        print(f"=======================================================\n")
        
        # Dynamically map to the correct checkpoint file
        model_checkpoint = f'local/resnet_gmoe_pacs{test_domain}/model.pkl'
        
        # Safety check: skip if this model hasn't been trained/downloaded
        if not os.path.exists(model_checkpoint):
            print(f"[ERROR] Checkpoint not found at {model_checkpoint}. Skipping model {test_domain}.")
            continue
            
        for target_class, class_name in pacs_classes.items():
            
            # Create nested output folder: master_dir/modelX/class_name/
            output_folder = os.path.join(master_output_dir, f"model{test_domain}", class_name)
            os.makedirs(output_folder, exist_ok=True)
            
            print(f"\n--- Hunting for PACS images in {data_dir} for Class: {class_name.upper()} ---")
            
            for d_code, d_name in domains.items():
                class_dir = os.path.join(data_dir, d_code, target_class)
                
                if os.path.exists(class_dir):
                    # Find all valid image files in this directory
                    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if images:
                        # Grab the first image found for this domain and class
                        img_path = os.path.join(class_dir, images[0])
                        
                        # Clean up the filename for saving
                        base_name = os.path.basename(img_path).split('.')[0]
                        save_name = f"gmoe_vis_{d_name}_{base_name}.png"
                        
                        # Construct full path inside the class-specific folder
                        full_save_path = os.path.join(output_folder, save_name)
                        
                        print(f"Processing {d_name.upper()} domain from {img_path}...")
                        visualize_expert_routing(model_checkpoint, img_path, full_save_path)
                    else:
                        print(f"[Warning] Found folder but no images in: {class_dir}")
                else:
                    print(f"[Warning] Could not find folder: {class_dir}")

    print(f"\nAll models processed successfully! Images saved in '{master_output_dir}' folder.")