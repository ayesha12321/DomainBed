import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import collections

from domainbed import datasets, algorithms
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
# ==== CONFIG ====
DATASET = "PACS"                  # PACS, VLCS, OfficeHome
DATA_DIR = "/content/DomainBed/domainbed/data"
MODEL_PATH = "/content/model.pkl"  # trained checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load checkpoint ====
print(f">>> Loading checkpoint from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
algorithm_dict = checkpoint["model_dict"]
hparams = checkpoint["model_hparams"]
args = checkpoint["args"]

# ==== Load dataset ====
dataset_class = vars(datasets)[DATASET]
dataset = dataset_class(DATA_DIR, args["test_envs"], hparams)

# ==== DomainBed-style splits for accuracy ====
in_splits, out_splits, uda_splits = [], [], []

for env_i, env in enumerate(dataset):
    uda = []

    # main split: out/in
    out, in_ = misc.split_dataset(
        env,
        int(len(env) * args["holdout_fraction"]),
        misc.seed_hash(args["trial_seed"], env_i)
    )

    # test env: split in_ further into uda/in
    if env_i in args["test_envs"]:
        uda, in_ = misc.split_dataset(
            in_,
            int(len(in_) * args.get("uda_holdout_fraction", 0.0)),
            misc.seed_hash(args["trial_seed"], env_i)
        )

    if hparams.get("class_balanced", False):
        in_weights = misc.make_weights_for_balanced_classes(in_)
        out_weights = misc.make_weights_for_balanced_classes(out)
        uda_weights = misc.make_weights_for_balanced_classes(uda) if len(uda) else None
    else:
        in_weights, out_weights, uda_weights = None, None, None

    in_splits.append((in_, in_weights))
    out_splits.append((out, out_weights))
    if len(uda):
        uda_splits.append((uda, uda_weights))

if args.get("task", "domain_generalization") == "domain_adaptation" and len(uda_splits) == 0:
    raise ValueError("Not enough unlabeled samples for domain adaptation.")

# ---- DataLoaders ----
train_loaders = [
    InfiniteDataLoader(dataset=env, weights=env_weights,
                       batch_size=hparams["batch_size"], num_workers=dataset.N_WORKERS)
    for i, (env, env_weights) in enumerate(in_splits) if i not in args["test_envs"]
]

uda_loaders = [
    InfiniteDataLoader(dataset=env, weights=env_weights,
                       batch_size=hparams["batch_size"], num_workers=dataset.N_WORKERS)
    for env, env_weights in uda_splits
]

eval_loaders = [
    FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
    for env, _ in (in_splits + out_splits + uda_splits)
]
eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
eval_loader_names = (
    [f"env{i}_in" for i in range(len(in_splits))] +
    [f"env{i}_out" for i in range(len(out_splits))] +
    [f"env{i}_uda" for i in range(len(uda_splits))]
)

# ==== Load algorithm ====
algo_class = algorithms.get_algorithm_class(args["algorithm"])
algorithm = algo_class(dataset.input_shape, dataset.num_classes,
                       len(dataset) - len(args["test_envs"]), hparams)
algorithm.load_state_dict(algorithm_dict, strict=False)
algorithm.to(DEVICE)
algorithm.eval()

# ==== Compute accuracy per split ====
print("\n>>> DomainBed split accuracies:")
results = {}
for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
    acc = misc.accuracy(algorithm, loader, weights, DEVICE)
    results[name + "_acc"] = acc
    print(f"{name:12s}: {acc:.2f}%")
avg_acc = np.mean(list(results.values()))
print(f"\n>>> Average Accuracy: {avg_acc:.2f}%")

