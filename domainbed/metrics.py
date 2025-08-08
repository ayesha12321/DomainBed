# full_end_to_end_analysis.py
import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial import ConvexHull
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = "/content/DomainBed/domainbed/data/PACS"

CHECKPOINTS = {
    "idd": "/content/all_domain_model.pkl",
    "erm_good": "/content/model.pkl",
    "erm_bad": "/content/bad_erm_model.pkl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

domain_map = {"A": 0, "C": 1, "P": 2, "S": 3}
domain_names = ["art_painting (A)", "cartoon (C)", "photo (P)", "sketch (S)"]

TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42

OUT_DIR = "./analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUT_DIR, "results_all_models.json")

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- UTILITIES ----------------
def mmd_rbf(X, Y, gamma=0.5):
    """Compute MMD between X and Y using RBF kernel (squared euclidean distances)."""
    if X is None or Y is None or len(X) == 0 or len(Y) == 0:
        return None
    XX = pairwise_distances(X, X, metric='sqeuclidean')
    YY = pairwise_distances(Y, Y, metric='sqeuclidean')
    XY = pairwise_distances(X, Y, metric='sqeuclidean')
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)
    return float(K_XX.mean() + K_YY.mean() - 2.0 * K_XY.mean())

def convex_hull_area_2d(points2d):
    """Return convex hull area for 2D points; 0.0 if insufficient points."""
    if points2d is None or len(points2d) < 3:
        return 0.0
    try:
        hull = ConvexHull(points2d)
        # In 2D, hull.area is the perimeter; hull.volume is area for 2D? 
        # SciPy docs: for 2-D, 'volume' is area. Use hull.volume for area to be safe.
        return float(hull.volume)
    except Exception:
        return 0.0

def safe_wasserstein_1d(X, Y):
    """Compute 1D Wasserstein by projecting onto principal axis (PCA first component)."""
    if X is None or Y is None or len(X) == 0 or len(Y) == 0:
        return None
    try:
        pca = PCA(n_components=1)
        XY = np.vstack([X, Y])
        pca.fit(XY)
        x_proj = pca.transform(X).ravel()
        y_proj = pca.transform(Y).ravel()
        return float(wasserstein_distance(x_proj, y_proj))
    except Exception:
        return None

# ---------------- MAIN ----------------
results_all = {}
from domainbed import algorithms  # assume domainbed available on PYTHONPATH

for model_name, ckpt_path in CHECKPOINTS.items():
    print("\n" + "="*70)
    print(f">>> MODEL: {model_name}")
    print(f">>> Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    algo_class = algorithms.get_algorithm_class(checkpoint["args"]["algorithm"])
    model = algo_class(checkpoint["model_input_shape"],
                       checkpoint["model_num_classes"],
                       checkpoint["model_num_domains"],
                       checkpoint["model_hparams"])
    model.load_state_dict(checkpoint["model_dict"])
    model.eval().to(DEVICE)
    print(">>> Model loaded and moved to device.")

    # containers
    features = []
    class_labels = []
    domain_labels = []
    preds = []
    correct_counts = [0] * 4
    total_counts = [0] * 4

    print(">>> Extracting features for entire dataset (all domains)...")
    for domain in domain_map.keys():
        d_idx = domain_map[domain]
        domain_path = os.path.join(DATA_DIR, domain)
        print(f"    Domain {domain} -> index {d_idx} -> path: {domain_path}")
        for class_name in tqdm(os.listdir(domain_path), desc=f"Domain {domain} classes"):
            class_path = os.path.join(domain_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat_tensor = model.featurizer(x)  # tensor shape [1, feat_dim]
                    feat = feat_tensor.cpu().numpy().reshape(-1)
                    logits = model.classifier(feat_tensor)
                    pred = int(logits.argmax(dim=1).item())
                features.append(feat)
                true_label = int(class_name)
                class_labels.append(true_label)
                domain_labels.append(d_idx)
                preds.append(pred)
                total_counts[d_idx] += 1
                if pred == true_label:
                    correct_counts[d_idx] += 1

    features = np.array(features)                # (N, D)
    class_labels = np.array(class_labels, int)   # (N,)
    domain_labels = np.array(domain_labels, int) # (N,)
    preds = np.array(preds, int)

    print(f">>> Extracted features: {features.shape[0]} samples, dim={features.shape[1]}")

    # ACCURACY (domain-wise)
    acc_per_domain = {}
    for i in range(4):
        acc = 0.0
        if total_counts[i] > 0:
            acc = 100.0 * correct_counts[i] / total_counts[i]
        acc_per_domain[domain_names[i]] = float(acc)
    if model_name == "idd":
        eval_accuracy = float(np.mean(list(acc_per_domain.values())))
    else:
        eval_accuracy = float(acc_per_domain[domain_names[0]])  # art painting (A) at index 0

    print(">>> Accuracy per domain (percent):")
    for k, v in acc_per_domain.items():
        print(f"    {k}: {v:.2f}%")
    print(f">>> Eval accuracy (rule): {eval_accuracy:.2f}%")

    # DOMAIN CENTROIDS (HD) & centroid drift L2
    domain_centroids_hd = {}
    for d in sorted(np.unique(domain_labels)):
        domain_centroids_hd[d] = features[domain_labels == d].mean(axis=0)
    centroid_drift = {}
    dlist = sorted(domain_centroids_hd.keys())
    for i_idx in range(len(dlist)):
        for j_idx in range(i_idx + 1, len(dlist)):
            i = dlist[i_idx]; j = dlist[j_idx]
            dist = float(np.linalg.norm(domain_centroids_hd[i] - domain_centroids_hd[j]))
            key = f"{domain_names[i]} <-> {domain_names[j]}"
            centroid_drift[key] = dist

    print(">>> Centroid drift (L2) computed.")

    # GENERAL DOMAIN ALIGNMENT (global MMD and Wasserstein approx)
    print(">>> Computing general domain-wise MMD and Wasserstein (projected)...")
    domain_mmd = {}
    domain_wass = {}
    for i_idx in range(len(dlist)):
        for j_idx in range(i_idx + 1, len(dlist)):
            i = dlist[i_idx]; j = dlist[j_idx]
            Xi = features[domain_labels == i]
            Xj = features[domain_labels == j]
            mmd_val = mmd_rbf(Xi, Xj, gamma=0.5)
            wass_val = safe_wasserstein_1d(Xi, Xj)
            key = f"{domain_names[i]} <-> {domain_names[j]}"
            domain_mmd[key] = None if mmd_val is None else float(mmd_val)
            domain_wass[key] = None if wass_val is None else float(wass_val)
    print(">>> Done domain-wise MMD/Wasserstein.")

    # CLASS-CONDITIONAL ALIGNMENT: class-wise MMD for each class & averaged per domain-pair
    print(">>> Computing class-conditional (per-class) MMD across domain pairs...")
    classes = sorted(np.unique(class_labels))
    class_cond_mmd_per_pair = {}   # accumulate per domain-pair list of MMDs, then average
    class_cond_mmd_details = {}    # detailed mapping class -> pair -> mmd

    for c in classes:
        class_cond_mmd_details[f"Class_{c}"] = {}
        # per-domain features for this class
        feats_by_domain = {}
        for d in dlist:
            feats_by_domain[d] = features[(class_labels == c) & (domain_labels == d)]
        # compute MMD for every domain pair
        for i_idx in range(len(dlist)):
            for j_idx in range(i_idx + 1, len(dlist)):
                i = dlist[i_idx]; j = dlist[j_idx]
                Xi = feats_by_domain[i]; Xj = feats_by_domain[j]
                key = f"{domain_names[i]} <-> {domain_names[j]}"
                if Xi.shape[0] > 0 and Xj.shape[0] > 0:
                    mmd_val = mmd_rbf(Xi, Xj, gamma=0.5)
                    class_cond_mmd_details[f"Class_{c}"][key] = None if mmd_val is None else float(mmd_val)
                    class_cond_mmd_per_pair.setdefault(key, []).append(float(mmd_val))
                else:
                    # mark missing if either domain has no samples for this class
                    class_cond_mmd_details[f"Class_{c}"][key] = None

    # average per-pair
    class_cond_mmd_avg = {}
    for pair, vals in class_cond_mmd_per_pair.items():
        if len(vals) > 0:
            class_cond_mmd_avg[pair] = float(np.mean(vals))
        else:
            class_cond_mmd_avg[pair] = None

    print(">>> Class-conditional MMD (per class) computed and averaged per pair.")

    # ----- t-SNE for visualization (one embedding reused for both plots) -----
    print(">>> Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE)
    embeddings_2d = tsne.fit_transform(features)
    print(">>> t-SNE done.")

    # 2D centroids for domains and class-domain centroids used for drawing lines
    domain_centroids_2d = {}
    for d in dlist:
        pts2 = embeddings_2d[domain_labels == d]
        domain_centroids_2d[d] = pts2.mean(axis=0).tolist()

    class_domain_centroids_2d = {}
    for c in classes:
        cname = f"Class_{c}"
        class_domain_centroids_2d[cname] = {}
        for d in dlist:
            pts = embeddings_2d[(class_labels == c) & (domain_labels == d)]
            if len(pts) > 0:
                class_domain_centroids_2d[cname][domain_names[d]] = pts.mean(axis=0).tolist()

    # convex hull areas (on 2D)
    convex_hull_area_by_domain = {}
    for d in dlist:
        pts2 = embeddings_2d[domain_labels == d]
        convex_hull_area_by_domain[domain_names[d]] = convex_hull_area_2d(pts2)

    # ---------------- PLOTTING ----------------
    # helper to map domain name -> index
    name_to_idx = {name: i for i, name in enumerate(domain_names)}

    # Plot 1: GENERAL DOMAIN ALIGNMENT (centroid lines: labelled with L2 and MMD/Wass)
    print(">>> Building general domain-alignment plot...")
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap("Set1")
    colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in range(10)]

    for d in dlist:
        idx = domain_labels == d
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=8, alpha=0.7, label=domain_names[d], color=colors[d % len(colors)])
        # shaded hull
        pts = embeddings_2d[idx]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                plt.fill(hull_pts[:, 0], hull_pts[:, 1], color=colors[d % len(colors)], alpha=0.12)
                # label area near centroid
                cen = domain_centroids_2d[d]
                plt.text(cen[0], cen[1], f"A:{convex_hull_area_by_domain[domain_names[d]]:.1f}", fontsize=8, ha='center')
            except Exception:
                pass

    for pair_key, l2val in centroid_drift.items():
        # find domain indices from pair_key like "art_painting (A) <-> photo (P)"
        left_name, right_name = pair_key.split(" <-> ")
        i = name_to_idx[left_name]; j = name_to_idx[right_name]
        c1 = domain_centroids_2d[i]; c2 = domain_centroids_2d[j]
        plt.plot([c1[0], c2[0]], [c1[1], c2[1]], linestyle='--', color='gray', linewidth=1, alpha=0.7)
        mid = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
        # label with L2 (HD) and MMD & WASS values
        mmd_val = domain_mmd.get(pair_key, None)
        wass_val = domain_wass.get(pair_key, None)
        txt = f"L2:{l2val:.2f}"
        if mmd_val is not None:
            txt += f"\nMMD:{mmd_val:.2f}"
        if wass_val is not None:
            txt += f"\nW:{wass_val:.2f}"
        plt.text(mid[0], mid[1], txt, fontsize=7, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.title(f"{model_name} — General Domain Alignment (centroids dashed; L2/MMD/W on labels)")
    plt.legend(markerscale=3)
    plt.axis('off')
    out_general = os.path.join(OUT_DIR, f"tsne_{model_name}_general.png")
    plt.tight_layout()
    plt.savefig(out_general, dpi=300)
    plt.close()
    print(f">>> Saved general plot: {out_general}")

    # Plot 2: CLASS-CONDITIONAL ALIGNMENT (dotted class-centroid lines with MMD per class)
    print(">>> Building class-conditional alignment plot...")
    plt.figure(figsize=(12, 10))
    for d in dlist:
        idx = domain_labels == d
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=8, alpha=0.65, label=domain_names[d], color=colors[d % len(colors)])
        # shaded hull again
        pts = embeddings_2d[idx]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                plt.fill(hull_pts[:, 0], hull_pts[:, 1], color=colors[d % len(colors)], alpha=0.12)
            except Exception:
                pass

    # draw dotted lines for each class where both domain centroids exist
    for c in classes:
        cname = f"Class_{c}"
        centroids_map = class_domain_centroids_2d.get(cname, {})
        # centroids_map: domain_name -> [x,y]
        domain_keys = list(centroids_map.keys())
        # for all pairs of domains that have centroids for this class
        for i_idx in range(len(domain_keys)):
            for j_idx in range(i_idx + 1, len(domain_keys)):
                left_name = domain_keys[i_idx]; right_name = domain_keys[j_idx]
                c1 = centroids_map[left_name]; c2 = centroids_map[right_name]
                plt.plot([c1[0], c2[0]], [c1[1], c2[1]], linestyle=':', color='darkred', linewidth=0.8, alpha=0.6)
                # label with class-wise MMD value if available
                pair_key = f"{left_name} <-> {right_name}"
                mmd_val = class_cond_mmd_details.get(cname, {}).get(pair_key, None)
                if mmd_val is not None:
                    mid = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
                    plt.text(mid[0], mid[1], f"{mmd_val:.2f}", fontsize=6, color='darkred', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    plt.title(f"{model_name} — Class-Conditional Alignment (dotted lines = class MMD)")
    plt.legend(markerscale=3)
    plt.axis('off')
    out_class = os.path.join(OUT_DIR, f"tsne_{model_name}_class_conditional.png")
    plt.tight_layout()
    plt.savefig(out_class, dpi=300)
    plt.close()
    print(f">>> Saved class-conditional plot: {out_class}")

    # ----------------- SAVE RESULTS -----------------
    results_all[model_name] = {
        "checkpoint": ckpt_path,
        "accuracy_per_domain_percent": acc_per_domain,
        "eval_accuracy_percent": eval_accuracy,
        "domain_centroids_hd": {domain_names[d]: domain_centroids_hd[d].tolist() for d in dlist},
        "centroid_drift_l2": centroid_drift,
        "domain_mmd": domain_mmd,
        "domain_wasserstein_1d": domain_wass,
        "class_conditional_mmd_per_class": class_cond_mmd_details,
        "class_conditional_mmd_avg_per_pair": class_cond_mmd_avg,
        "convex_hull_area_2d_by_domain": convex_hull_area_by_domain,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "tsne_general_png": out_general,
        "tsne_class_conditional_png": out_class
    }

    print(f">>> Finished model {model_name}.")

# write JSON summary
with open(RESULTS_JSON, "w") as f:
    json.dump(results_all, f, indent=2)
print("\n>>> All models processed. Results saved to:", RESULTS_JSON)
