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

def format_log_entry(result, true_label, image_path, global_index, class_names, domain_map):
    """Formats a dictionary of results into a neat string for logging."""
    entry = (
        f"============================================================\n"
        f"SAMPLE #{global_index}\n"
        f"File: {os.path.basename(image_path)}\n"
        f"True Label: {class_names[true_label]}\n"
        f"Generalist Top Prediction: {class_names[result['gen_pred']]}\n"
        f"------------------------------------------------------------\n"
        f"EVENT: {result['reason']}\n"
        f"Generalist Confidence: {result['gen_confidence']:.2%}\n"
    )

    if 'gen_softmax' in result:
        gen_probs = result['gen_softmax']
        gen_top3_idx = np.argsort(gen_probs)[-3:][::-1]
        gen_top3_str = ", ".join([f"{class_names[i]} ({gen_probs[i]:.2%})" for i in gen_top3_idx])
        entry += f"Generalist Top Predictions (Full): {gen_top3_str}\n"

    if 'router_weights' in result:
        weights = result['router_weights']
        weight_str = ", ".join([f"Head {i}: {w:.2f}" for i, w in enumerate(weights)])
        entry += f"Router Weights: [{weight_str}]\n"

    if 'spec_poll_details' in result:
        poll_details = result['spec_poll_details']
        vote_str_parts = []
        for detail in sorted(poll_details, key=lambda x: x['domain']):
            domain_name = domain_map.get(detail['domain'], f"Domain {detail['domain']}")
            vote_name = class_names[detail['vote']]
            conf = detail['conf']
            vote_str_parts.append(f"{domain_name}: {vote_name} ({conf:.1%})")
        
        vote_str = ", ".join(vote_str_parts)
        entry += f"Specialist Individual Predictions: [{vote_str}]\n"
    
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

    # --- Load hparams differently based on algorithm ---
    if args.algorithm == 'HybridEnsembleMultiHead':
        if not args.model_path:
             raise ValueError("--model_path is required for HybridEnsembleMultiHead")
        model_path = args.model_path
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"MultiHead model not found at {model_path}")
        
        saved_state = torch.load(model_path, map_location='cpu')
        hparams = saved_state['model_hparams']
        print("--- Loaded HParams from saved MultiHead model ---")
        
        # [CRITICAL FIX] Apply Command Line Overrides
        if args.hparams:
            overrides = json.loads(args.hparams)
            hparams.update(overrides)
            print(f"--- Applying Command Line HParam Overrides: {overrides} ---")

        hparams['load_trained_model_path'] = model_path 
        hparams['confidence_thresh'] = args.confidence_threshold
        # ---------------------------

    elif args.algorithm == 'HybridEnsemble':
        if not args.model_path_root:
             raise ValueError("--model_path_root is required for HybridEnsemble")
        generalist_model_path = os.path.join(args.model_path_root, 'generalist', 'model.pkl')
        if not os.path.exists(generalist_model_path):
             raise FileNotFoundError(f"Generalist model not found at {generalist_model_path}")
        saved_state = torch.load(generalist_model_path, map_location='cpu')
        hparams = saved_state['model_hparams']
        print("--- Loaded HParams from saved generalist model ---")
        hparams['model_path_root'] = args.model_path_root
        hparams['confidence_thresh'] = args.confidence_threshold
        hparams['fallback_model'] = args.fallback_model
        hparams['consensus_mode'] = args.consensus_mode
        hparams['specialist_mode'] = args.specialist_mode
        hparams['weighting_mode'] = args.weighting_mode 
        
        if args.hparams:
            hparams.update(json.loads(args.hparams))

    else:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        if args.hparams: hparams.update(json.loads(args.hparams))
        print("--- Using Default HParams ---")

    for k, v in sorted(hparams.items()): print(f"\t{k}: {v}")
    print("--------------------------------------------------")

    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    if args.dataset == 'PACS':
        domain_map = {0: 'Photo', 1: 'Art', 2: 'Cartoon', 3: 'Sketch'}
    elif args.dataset == 'VLCS':
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

    hparams['test_env'] = test_env_index 
    
    algorithm = vars(algorithms)[args.algorithm](
        dataset.input_shape, 
        dataset.num_classes, 
        len(dataset) - len(args.test_envs), 
        hparams
    )
    algorithm.to(device)

    if args.algorithm not in ['HybridEnsembleMultiHead', 'HybridEnsemble']:
            if args.model_path:
                print(f"Loading trained weights for {args.algorithm} from: {args.model_path}")
                saved_state = torch.load(args.model_path, map_location=device)
                algorithm.load_state_dict(saved_state['model_dict'], strict=False)
            else:
                print("WARNING: No --model_path provided! Evaluating an UNTRAINED model.")
    
    print(f"\n--- Starting Inference ---")
    print(f"Algorithm: {args.algorithm}")
    if args.algorithm == 'HybridEnsembleMultiHead':
        print(f"Loaded model from: {hparams['load_trained_model_path']}")
         
    print(f"Using Generalist Confidence Threshold: {args.confidence_threshold:.2%}")
    
    log_dir = args.model_path_root 
    if args.algorithm == 'HybridEnsembleMultiHead':
        log_dir = os.path.dirname(args.model_path)

    if log_dir is None:
        print("Warning: --model_path_root was not provided. Logging to current directory ('.').")
        log_dir = "."

    log_path = os.path.join(log_dir, "inference_log.txt")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Logging noteworthy events to: {log_path}")
    print("-" * 30)

    total_counts = collections.defaultdict(int)
    correct_counts = collections.defaultdict(int)

    with open(log_path, "w") as log_file:
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            raw_results = algorithm.predict(x)

            # --- FIX: Standardize outputs for non-ensemble models ---
            if isinstance(raw_results, torch.Tensor):
                # Standard algorithm (like GMoE_ERM or ERM) returning raw logits
                probs = torch.nn.functional.softmax(raw_results, dim=1)
                confidences, preds = torch.max(probs, dim=1)
                
                batch_results = []
                for b_idx in range(len(preds)):
                    batch_results.append({
                        'final_pred': preds[b_idx].item(),
                        'gen_pred': preds[b_idx].item(),
                        'gen_confidence': confidences[b_idx].item(),
                        'reason': 'STANDARD_MODEL_PREDICTION'
                    })
            else:
                # HybridEnsemble returning pre-formatted dictionaries
                batch_results = raw_results
            # ---------------------------------------------------------

            for j, result in enumerate(batch_results):
                global_index = i * args.batch_size + j
                final_pred, true_label, reason = result['final_pred'], y[j].item(), result['reason']

                total_counts[reason] += 1
                if final_pred == true_label:
                    correct_counts[reason] += 1

                # Log if it's not a standard high confidence ensemble, OR if it's the standard model
                if reason != 'GENERALIST_HIGH_CONFIDENCE' or args.algorithm in ['HybridEnsembleMultiHead', 'GMoE_ERM']:
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
    parser.add_argument('--model_path_root', type=str, required=False, help='Root directory for HybridEnsemble models.')

    parser.add_argument('--model_path', type=str, default=None, help='Path to the single model file (used by MultiHead).')
    # --------------------------
    parser.add_argument('--test_envs', type=int, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '--confidence_threshold', type=float, default=0.85,
        help='Confidence threshold for the generalist model/head.'
    )
    parser.add_argument(
        '--fallback_model', type=str, default='specialist_2',
        choices=['erm', 'specialist_0', 'specialist_1', 'specialist_2'],
        help='Model to use as fallback (HybridEnsemble voting only).'
    )
    parser.add_argument(
        '--consensus_mode', type=str, default='majority',
        choices=['majority', 'strict'],
        help="Consensus level required (HybridEnsemble voting only)."
    )
    parser.add_argument(
        '--specialist_mode', type=str, default='voting',
        choices=['voting', 'weighting'],
        help="Method to combine specialists (HybridEnsemble only)."
    )

    parser.add_argument(
        '--weighting_mode', type=str, default='confidence',
        choices=['confidence', 'entropy', 'domain','domain_dynamic', 'cosine_router'],
        help="Type of weighting used in HybridEnsemble."
    )
    parser.add_argument(
        '--hparams', type=str, default=None, 
        help='JSON-serialized hparams dict (used for standard algorithms).'
    )
    # ---

    args = parser.parse_args()
    main(args)