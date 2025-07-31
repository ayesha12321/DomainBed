import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== CONFIG ====
DATA_DIR = "/content/DomainBed/domainbed/data/PACS"
CHECKPOINT_PATH = "/content/model.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load Model ====
print(">>> Loading trained model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
from domainbed import algorithms
algo_class = algorithms.get_algorithm_class(checkpoint["args"]["algorithm"])
model = algo_class(checkpoint["model_input_shape"],
                   checkpoint["model_num_classes"],
                   checkpoint["model_num_domains"],
                   checkpoint["model_hparams"])
model.load_state_dict(checkpoint["model_dict"])
model.eval().to(DEVICE)
print(">>> Model loaded successfully.")

# ==== Image Transform ====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Load Images & Extract Features ====
features, class_labels, domain_labels = [], [], []
domain_map = {"A":0, "C":1, "P":2, "S":3}
domain_names = ["art_painting (A)", "cartoon (C)", "photo (P)", "sketch (S)"]

print(">>> Extracting features for all domains...")
for domain in domain_map.keys():
    domain_path = os.path.join(DATA_DIR, domain)
    print(f"\n>>> Processing domain: {domain}")
    for class_name in tqdm(os.listdir(domain_path), desc=f"Domain {domain}"):
        class_path = os.path.join(domain_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.featurizer(x).cpu().numpy().flatten()
            features.append(feat)
            class_labels.append(int(class_name))
            domain_labels.append(domain_map[domain])

features = np.array(features)
class_labels = np.array(class_labels)
domain_labels = np.array(domain_labels)

print(f"\n>>> Total extracted features: {features.shape[0]}")
print(f">>> Feature vector size: {features.shape[1]}")

# ==== Run t-SNE ====
print(">>> Running t-SNE on extracted features (this may take a while)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings = tsne.fit_transform(features)

# ==== Function to Plot t-SNE ====
def plot_tsne(embeddings, labels, title, legend_labels, cmap, save_name):
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embeddings[:,0], embeddings[:,1], c=labels, cmap=cmap, s=2, alpha=0.7)
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
    plt.legend(handles, legend_labels, title=title, loc="best")
    plt.title(f"t-SNE Visualization by {title}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ==== Plot by CLASS ====
unique_classes = sorted(np.unique(class_labels))
plot_tsne(embeddings, class_labels, "Class", [f"Class {c}" for c in unique_classes], "tab20", "tsne_by_class.png")

# ==== Plot by DOMAIN ====
plot_tsne(embeddings, domain_labels, "Domain", domain_names, "Set1", "tsne_by_domain.png")
