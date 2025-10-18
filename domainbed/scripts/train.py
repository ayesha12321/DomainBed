# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--train_domains', type=int, nargs='+')
    parser.add_argument('--load_model', type=str, default=None,
                    help="Path to a model.pkl file to load weights from before training.")
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.train_domains:
        # This block activates only when training a specialist.
        # It filters the in_splits to keep only the desired training domain.
        filtered_in_splits = []
        for i, (env, env_weights) in enumerate(in_splits):
            if i in args.train_domains:
                filtered_in_splits.append((env, env_weights))
        in_splits = filtered_in_splits

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)
    # print(">>> [DEBUG] Initializing algorithm:", args.algorithm)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    
    if args.load_model:
        try:
            loaded_dict = torch.load(args.load_model)
            algorithm.load_state_dict(loaded_dict['model_dict'])
            print(f">>> [INFO] Loaded pre-trained weights from: {args.load_model}")
        except Exception as e:
            print(f">>> [ERROR] Could not load model from {args.load_model}: {e}")

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

        # For plotting
    train_loss_logs = collections.defaultdict(list)  # store any numeric loss separately
    train_steps = []

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    # print(f">>> [DEBUG] Steps per epoch: {steps_per_epoch}, Total steps: {n_steps}")
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None

    for step in range(start_step, n_steps):
        print(f"\n>>> [DEBUG] ===== Step {step+1}/{n_steps} =====")
        print(f">>> [DEBUG] GPU Memory before step: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

        step_start_time = time.time()

        # ---- Load minibatches ----
        try:
            minibatches_device = [(x.to(device), y.to(device))
                                  for x, y in next(train_minibatches_iterator)]
        except Exception as e:
            print(f">>> [ERROR] Failed to load minibatch: {e}")
            break

        # ---- Handle UDA minibatch ----
        if args.task == "domain_adaptation":
            try:
                uda_device = [x.to(device) for x, _ in next(uda_minibatches_iterator)]
                print(">>> [DEBUG] UDA minibatch loaded.")
            except StopIteration:
                print(">>> [WARN] No UDA minibatch available.")
                uda_device = None
        else:
            uda_device = None

        # ---- Algorithm update ----
        try:
            step_vals = algorithm.update(minibatches_device, uda_device)
        except Exception as e:
            print(f">>> [ERROR] Algorithm update failed: {e}")
            break

        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

                # store loss for curve
        numeric_vals = {k: v for k, v in step_vals.items() if isinstance(v, (int, float))}
        if len(numeric_vals) > 0:
            train_steps.append(step)
            for k, v in numeric_vals.items():
                # store all numeric metrics/losses
                train_loss_logs[k].append(v)

        # ---- Checkpointing & Evaluation every 100 steps ----
        if (step > 0 and step % 1000 == 0) or (step == n_steps - 1):
            print(f">>> [DEBUG] Running evaluation at step {step}...")

            results = {'step': step, 'epoch': step / steps_per_epoch}
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # ---- Evaluate on all environments ----
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            # ---- Log memory usage ----
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
            print(f">>> [DEBUG] GPU Memory after evaluation: {results['mem_gb']:.2f} GB")

            # ---- Print results table ----
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            # ---- Save results ----
            results.update({'hparams': hparams, 'args': vars(args)})
            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # ---- Save checkpoint every 1000 steps ----
            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])
            if (step % 1000 == 0 and step > 0 ) or args.save_model_every_checkpoint or (step == n_steps - 1):
                save_checkpoint(f'model_step{step}.pkl')
                print(f">>> [DEBUG] Checkpoint model_step{step}.pkl saved.")

    # ---- Final checkpoint ----
    save_checkpoint('model.pkl')
    print(">>> [DEBUG] Final model checkpoint saved.")

# ---- Plot final training loss curve ----
    # ---- Plot all tracked losses ----
    if len(train_loss_logs) > 0:
        plt.figure(figsize=(8, 5))
        for loss_name, loss_values in train_loss_logs.items():
            plt.plot(train_steps, loss_values, label=loss_name)
        plt.xlabel("Step")
        plt.ylabel("Loss Value")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "training_loss_curves.png"))
        plt.close()
        print(f">>> [DEBUG] Training loss curves saved to {os.path.join(args.output_dir, 'training_loss_curves.png')}")

        # ---- Plot test env accuracy bar chart ----
    
        # ---- Plot test env accuracy bar chart from results.jsonl ----
    results_path = os.path.join(args.output_dir, "results.jsonl")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    test_env = args.test_envs[0]
    acc_key = f"env{test_env}_out_acc"

    if os.path.exists(results_path):
        steps = []
        accs = []

        with open(results_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if acc_key in record and "step" in record:
                        steps.append(record["step"])
                        accs.append(record[acc_key])
                except json.JSONDecodeError:
                    continue

        if len(steps) > 0:
            plt.figure(figsize=(10, 6))
            if len(steps) > 1:
                step_gap = steps[1] - steps[0]
                bar_width = step_gap * 0.6   # 60% of the distance between steps
            else:
                bar_width = 50

            bars = plt.bar(steps, [a * 100 for a in accs], width=bar_width, align='center')

            plt.title(f'Test Env {test_env} Out Accuracy over Steps')
            plt.xlabel('Training Step')
            plt.ylabel('Accuracy (%)')
            plt.grid(axis='y')

            # Add percentage labels above bars
            for bar, acc in zip(bars, accs):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                         f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=8)

            bar_path = os.path.join(plots_dir, f'env{test_env}_out_acc_bar.png')
            plt.tight_layout()
            plt.savefig(bar_path)
            plt.close()
            print(f">>> [DEBUG] Saved test env out accuracy bar chart: {bar_path}")
        else:
            print(f">>> [DEBUG] No valid accuracy entries found in {results_path}")
    else:
        print(f">>> [DEBUG] results.jsonl not found at {results_path}, skipping accuracy plot.")

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    print(">>> [DEBUG] Training completed successfully.")
