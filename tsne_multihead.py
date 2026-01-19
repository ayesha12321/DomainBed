import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from domainbed import algorithms

# ================= CONFIG =================
DATA_DIR = "/content/DomainBed/domainbed/data/VLCS"
FINAL_MODEL_PATH = "/content/drive/MyDrive/multihead_staged_vlcs_env3/multihead_staged_vlcs_env3/final_model.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD FINAL MODEL =================
print(">>> Loading Final Model...")

# Load checkpoint
checkpoint = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)

# Get algorithm class
algo_class = algorithms.get_algorithm_class(checkpoint["args"]["algorithm"])

# Make a copy of hparams and remove keys that can trigger reload
hparams = checkpoint["model_hparams"].copy()
hparams["load_trained_model_path"] = FINAL_MODEL_PATH  # required by constructor
# hparams.pop("finetune_head_idx", None)                # remove staged info

model = algo_class(
    checkpoint["model_input_shape"],
    checkpoint["model_num_classes"],
    checkpoint["model_num_domains"],
    hparams
)

model.load_state_dict(checkpoint["model_dict"])
model.eval().to(DEVICE)

print(">>> Final multi-head model loaded successfully!")
num_heads = len(model.network.heads)
print(f">>> Number of heads in the model: {num_heads} (Head 0 = generalist, Head 1..N = specialists)")

# ================= IMAGE TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ================= VLCS DOMAINS =================
domain_map = {"C": 0, "L": 1, "S": 2, "V": 3}
domain_names = ["Caltech101 (C)", "LabelMe (L)", "SUN09 (S)", "VOC2007 (V)"]

# ================= BUILD CLASS MAPPING =================
example_domain = list(domain_map.keys())[0]
class_names = sorted(
    d for d in os.listdir(os.path.join(DATA_DIR, example_domain))
    if os.path.isdir(os.path.join(DATA_DIR, example_domain, d))
)
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

print("\n>>> VLCS class mapping:")
for cls, idx in class_to_idx.items():
    print(f"  {cls} → {idx}")

# ================= FEATURE EXTRACTION =================
def extract_features(x, head_idx=0):
    """Extract features from a specific head."""
    with torch.no_grad():
        feat = model.network.featurizer(x)
        logits = model.network.heads[head_idx](feat)
    return feat.cpu().numpy().flatten(), logits.cpu().numpy().flatten()

# Collect features for all heads
features_dict = {f"head_{i}": [] for i in range(num_heads)}
class_labels = []
domain_labels = []

print("\n>>> Extracting features for all heads...")

for domain, domain_idx in domain_map.items():
    domain_path = os.path.join(DATA_DIR, domain)
    for class_name in tqdm(os.listdir(domain_path), desc=f"Domain {domain}"):
        class_path = os.path.join(domain_path, class_name)
        if not os.path.isdir(class_path):
            continue
        true_label = class_to_idx[class_name]

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue

            x = transform(img).unsqueeze(0).to(DEVICE)

            for head_idx in range(num_heads):
                feat, _ = extract_features(x, head_idx=head_idx)
                features_dict[f"head_{head_idx}"].append(feat)

            class_labels.append(true_label)
            domain_labels.append(domain_idx)

# Convert to numpy arrays
for key in features_dict:
    features_dict[key] = np.array(features_dict[key])
class_labels = np.array(class_labels)
domain_labels = np.array(domain_labels)

# ================= t-SNE PLOTTING =================
def plot_tsne(features, labels, title, legend_labels, cmap, save_name):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    embeddings = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=cmap, s=3, alpha=0.7)
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
    plt.legend(handles, legend_labels, title=title, loc="best")
    plt.title(f"t-SNE Visualization by {title}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

# Plot for generalist + specialists by class
for head_idx in range(num_heads):
    print(f"\n>>> Running t-SNE for Head {head_idx}...")
    plot_tsne(
        features_dict[f"head_{head_idx}"],
        class_labels,
        title=f"vlcs_env3_tsne_head{head_idx}_by_class",
        legend_labels=[idx_to_class[i] for i in range(len(idx_to_class))],
        cmap="tab20",
        save_name=f"vlcs_env3_tsne_head{head_idx}_by_class.png"
    )

# Plot for generalist + specialists by domain
for head_idx in range(num_heads):
    plot_tsne(
        features_dict[f"head_{head_idx}"],
        domain_labels,
        title=f"vlcs_env3_tsne_head{head_idx}_by_domain",
        legend_labels=domain_names,
        cmap="Set1",
        save_name=f"vlcs_env3_tsne_head{head_idx}_by_domain.png"
    )








