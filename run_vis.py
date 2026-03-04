import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os

# Import your modified networks
from domainbed import networks

def visualize_expert_routing(model_path, image_path, save_path, num_experts=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    saved = torch.load(model_path, map_location=device)
    
    hparams = {
        'resnet18': True,
        'resnet18_pretrained': False,
        'resnet50_pretrained': False,
        'resnet50_augmix': False,
        'freeze_bn': False,
        'resnet_dropout': 0.0,
        'vit': False,
        'use_gmoe': True,
        'gmoe_num_experts': num_experts,
        'gmoe_top_k': 2,
    }
    
    featurizer = networks.Featurizer((3, 224, 224), hparams).to(device)
    
    clean_dict = {}
    for k, v in saved['model_dict'].items():
        if k.startswith('network.0.'):
            clean_dict[k.replace('network.0.', '')] = v
        elif k.startswith('featurizer.'):
            clean_dict[k.replace('featurizer.', '')] = v
            
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

    colors = [
        [255, 0, 0],   # Red
        [0, 255, 0],   # Green
        [0, 0, 255],   # Blue
        [255, 255, 0]  # Yellow
    ]
    
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(num_experts):
        mask = (routing_map_resized == i)
        overlay[mask] = colors[i]

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
    legend_patches = [mpatches.Patch(color=np.array(colors[i])/255.0, label=f'Expert {i}') for i in range(num_experts)]
    axes[1].legend(handles=legend_patches, loc='best')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"SUCCESS! Verification saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    model_checkpoint = 'local/resnet_gmoe_pacs0/model.pkl'
    data_dir = 'domainbed/data/PACS'
    
    # Define which test domain this model corresponds to
    test_domain = 0 
    
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
    
    print(f"--- Starting batch processing for model {test_domain} ---")
    
    # --- NEW: Loop through every class in the dictionary ---
    for target_class, class_name in pacs_classes.items():
        
        # Create output folder for the current class
        output_folder = f"model{test_domain}_{class_name}"
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
                    visualize_expert_routing(model_checkpoint, img_path, full_save_path, num_experts=4)
                else:
                    print(f"[Warning] Found folder but no images in: {class_dir}")
            else:
                print(f"[Warning] Could not find folder: {class_dir}")

    print("\nBatch processing complete!")