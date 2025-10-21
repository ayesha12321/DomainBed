import argparse
import json
import os
import time
import collections
import numpy as np
import torch
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader


# This is a function inside domainbed/scripts/predict.py
def format_log_entry(result, true_label, image_path, global_index, class_names, domain_map):
    """Formats a dictionary of results into a neat string for logging."""
    entry = (
        f"============================================================\n"
        f"SAMPLE #{global_index}\n"
        f"File: {os.path.basename(image_path)}\n"
        f"True Label: {class_names[true_label]}\n"
        f"------------------------------------------------------------\n"
        f"EVENT: {result['reason']}\n"
    )

    if 'gen_softmax' in result:
        gen_probs = result['gen_softmax']
        gen_top3_idx = np.argsort(gen_probs)[-3:][::-1]
        gen_top3_str = ", ".join([f"{class_names[i]} ({gen_probs[i]:.2%})" for i in gen_top3_idx])
        entry += f"Generalist Raw Prediction: {gen_top3_str}\n"

    # --- THIS IS THE CHANGE: Log the full 4-voter poll results ---
    if 'spec_poll_details' in result:
        vote_str_parts = []
        # Add the Generalist's vote to the list first
        gen_vote_name = class_names[result['gen_pred']]
        gen_conf = result['gen_confidence']
        vote_str_parts.append(f"Generalist: {gen_vote_name} ({gen_conf:.1%})")

        # Add the specialists' votes
        poll_details = result['spec_poll_details']
        for detail in sorted(poll_details, key=lambda x: x['domain']):
            domain_name = domain_map.get(detail['domain'], f"Domain {detail['domain']}")
            vote_name = class_names[detail['vote']]
            conf = detail['conf']
            vote_str_parts.append(f"{domain_name}: {vote_name} ({conf:.1%})")

        entry += f"Full Poll Results: [{', '.join(vote_str_parts)}]\n"
    # -------------------------------------------------------------

    if 'blended_softmax' in result:
        blend_probs = result['blended_softmax']
        blend_top3_idx = np.argsort(blend_probs)[-3:][::-1]
        blend_top3_str = ", ".join([f"{class_names[i]} ({blend_probs[i]:.2%})" for i in blend_top3_idx])
        entry += f"Blended Specialist Prediction: {blend_top3_str}\n"

    final_pred_class = class_names[result['final_pred']] if result['final_pred'] != -1 else "FLAGGED"
    entry += f"Final Decision: {final_pred_class}\n"
    entry += f"============================================================\n\n"

    return entry


def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    generalist_model_path = os.path.join(args.model_path_root, 'generalist', 'model.pkl')
    if not os.path.exists(generalist_model_path):
        raise FileNotFoundError(f"Generalist model not found at {generalist_model_path}")

    saved_state = torch.load(generalist_model_path)
    hparams = saved_state['model_hparams']

    print("--- Loaded HParams from saved generalist model ---")
    for k, v in sorted(hparams.items()):
        print(f"\t{k}: {v}")
    print("--------------------------------------------------")

    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    if args.dataset == 'PACS':
        domain_map = {0: 'Photo', 1: 'Art', 2: 'Cartoon', 3: 'Sketch'}
    elif args.dataset == 'VLCS':
        domain_map = {0: 'VOC', 1: 'LabelMe', 2: 'Caltech', 3: 'SUN'}
        domain_map = {0: 'VOC', 1: 'LabelMe', 2: 'Caltech', 3: 'SUN'}
    else:
        domain_map = {}

    test_env_index = args.test_envs[0]
    test_dataset = dataset[test_env_index]
    test_loader = FastDataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS
    )

    sample_paths = [s[0] for s in test_dataset.samples]
    class_names = test_dataset.classes

    hparams['model_path_root'] = args.model_path_root
    hparams['test_env'] = test_env_index
    hparams['confidence_thresh'] = args.confidence_threshold
    hparams['fallback_model'] = args.fallback_model
    hparams['consensus_mode'] = args.consensus_mode
    hparams['specialist_mode'] = args.specialist_mode
    hparams['weighting_mode'] = args.weighting_mode

    # --- Explicitly add missing keys ---
    hparams['data_dir'] = args.data_dir
    hparams['dataset'] = args.dataset
    hparams['data_augmentation'] = True
    hparams['random_seed'] = 0
    hparams['algorithm'] = args.algorithm
    algorithm = vars(algorithms)[args.algorithm](
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(args.test_envs),
        hparams
    )
    algorithm.to(device)

    print(f"Loading models from: {args.model_path_root}")
    print(f"Using Generalist Confidence Threshold: {args.confidence_threshold:.2%}")
    print(f"Using Specialist Mode: {args.specialist_mode.upper()}")
    if args.specialist_mode == 'voting':
        print(f"Using Consensus Level: {args.consensus_mode.upper()}")
    print(f"Using Fallback Model: {args.fallback_model}")
    print(f"Logging noteworthy events to: inference_log.txt")
    print("-" * 30)

    log_path = os.path.join(args.model_path_root, "inference_log.txt")
    os.makedirs(args.model_path_root, exist_ok=True)

    print(f"Logging noteworthy events to: {log_path}")
    print("-" * 30)

    total_counts = collections.defaultdict(int)
    correct_counts = collections.defaultdict(int)

    with open(log_path, "w") as log_file:
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            batch_results = algorithm.predict(x)

            for j, result in enumerate(batch_results):
                global_index = i * args.batch_size + j
                final_pred, true_label, reason = result['final_pred'], y[j].item(), result['reason']

                total_counts[reason] += 1
                if final_pred == true_label:
                    correct_counts[reason] += 1

                if reason != 'GENERALIST_HIGH_CONFIDENCE':
                    log_entry = format_log_entry(
                        result, true_label,
                        sample_paths[global_index],
                        global_index,
                        class_names, domain_map
                    )
                    log_file.write(log_entry)

    num_total, num_correct = sum(total_counts.values()), sum(correct_counts.values())
    overall_accuracy = num_correct / num_total if num_total > 0 else 0.0

    print("--- Overall Performance ---")
    print(f"Total Test Samples: {num_total}")
    print(f"Total Correct Predictions: {num_correct}")
    print(f"Overall System Accuracy: {overall_accuracy:.2%}\n")

    print("--- Accuracy Breakdown by Decision Path ---")
    for reason, total in sorted(total_counts.items()):
        correct = correct_counts[reason]
        accuracy = correct / total if total > 0 else 0.0
        contribution = correct / num_total if num_total > 0 else 0.0
        print(f" Reason: {reason}")
        print(f" - Cases Handled: {total} ({total/num_total:.1%})")
        print(f" - Accuracy of this Path: {accuracy:.2%}")
        print(f" - Contribution to Overall Accuracy: {contribution:.2%}")
    print("------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DomainBed Inference with Logging")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--model_path_root', type=str, required=True)
    parser.add_argument('--test_envs', type=int, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '--confidence_threshold', type=float, default=0.85,
        help='Confidence threshold for the generalist model.'
    )
    parser.add_argument(
        '--fallback_model', type=str, default='specialist_2',
        choices=['erm', 'specialist_0', 'specialist_1', 'specialist_2'],
        help='Model to use as fallback.'
    )
    parser.add_argument(
        '--consensus_mode', type=str, default='majority',
        choices=['majority', 'strict'],
        help="Consensus level required from specialists."
    )
    parser.add_argument(
        '--specialist_mode', type=str, default='voting',
        choices=['voting', 'weighting'],
        help="Method to combine specialists: simple 'voting' or 'weighting' full probabilities."
    )

    parser.add_argument('--weighting_mode', type=str, default='confidence',
                    choices=['confidence', 'entropy', 'domain','domain_dynamic'],
                    help="Type of weighting used in HybridEnsemble when specialist_mode=weighting.")


    args = parser.parse_args()
    main(args)
