import os
import subprocess
import argparse
import json 
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MultiHead Staged: Generalist then Specialists')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_envs', type=int, nargs='+', required=True)
    # Hparams arguments are removed
    parser.add_argument('--stage2_steps', type=int, default=501, help='Number of steps for Stage 2 fine-tuning')

    args = parser.parse_args()

    # --- Hardcoded Hyperparameters ---
    stage1_hparams = {
        "batch_size": 8,
        "resnet18": True,
        "resnet18_pretrained": True,
        "lr": 5e-4,
        "resnet50_augmix": False
    }
    stage2_hparams = {
        "batch_size": 8,
        "resnet18": True,
        "resnet18_pretrained": True,
        "lr": 5e-4,
        "resnet50_augmix": False
    }
    # Convert stage 1 hparams to JSON string for the command
    stage1_hparams_str = json.dumps(stage1_hparams)
    # ---------------------------------

    data_dir_abs = os.path.abspath(args.data_dir)
    print(f"Using absolute data path: {data_dir_abs}")

    if args.dataset == 'PACS': num_domains = 4
    elif args.dataset == 'VLCS': num_domains = 4
    else: raise ValueError(f"Dataset '{args.dataset}' is not configured.")

    all_domains = list(range(num_domains))
    final_test_env = args.test_envs[0]
    source_domains = [d for d in all_domains if d != final_test_env] # Domain indices for specialists

    print(f"Dataset: {args.dataset}")
    print(f"Final Test Environment: {final_test_env}")
    print(f"Source Domains for Specialists: {source_domains}")
    print(f"Stage 1 HParams: {stage1_hparams_str}")
    print(f"Stage 2 HParams Base: {json.dumps(stage2_hparams)}")
    print(f"Stage 2 Steps: {args.stage2_steps}")
    print("-" * 30)

    # --- STAGE 1: Train Generalist Head + Backbone ---
    print(">>> STAGE 1: Training Generalist Head + Backbone...")
    stage1_output_dir = os.path.join(args.output_dir, 'stage1_generalist')
    stage1_model_path = os.path.join(stage1_output_dir, 'model.pkl')

    stage1_command = [
        'python', '-m', 'domainbed.scripts.train',
        '--data_dir', data_dir_abs,
        '--dataset', args.dataset,
        '--algorithm', 'ERMGeneralistHeadOnly',
        '--test_envs', str(final_test_env),
        '--output_dir', stage1_output_dir,
        '--hparams', stage1_hparams_str # Pass the hardcoded string
    ]
    subprocess.run(stage1_command, input='y\n', text=True)
    print("-" * 30)

# --- STAGE 2: Fine-Tune Specialist Heads ---
    print(f">>> STAGE 2: Fine-Tuning Specialist Heads from {stage1_model_path}...")
    current_model_path = stage1_model_path # Start with stage 1 model

    # Enumerate source domains to map their order to head indices 1, 2, 3...
    for i, domain_idx in enumerate(source_domains):
        specialist_head_idx = i + 1 # Head 1 for the *first* source domain, Head 2 for the *second*, etc.
    # -----------------------

        print(f"Fine-Tuning Head {specialist_head_idx} for Source Domain {domain_idx}...")

        stage2_output_dir = os.path.join(args.output_dir, f'stage2_finetune_head_{specialist_head_idx}_domain_{domain_idx}')

        # Create hparams string for this specific fine-tuning run
        finetune_hparams = stage2_hparams.copy() # Start with base Stage 2 hparams
        finetune_hparams['load_trained_model_path'] = current_model_path # Load previous model
        finetune_hparams['finetune_head_idx'] = specialist_head_idx
        finetune_hparams_str = json.dumps(finetune_hparams)

        # To train on ONLY domain_idx, treat all other domains (incl. final test) as test domains for this run
        current_run_test_envs = [str(d) for d in all_domains if d != domain_idx]

        stage2_command = [
            'python', '-m', 'domainbed.scripts.train',
            '--data_dir', data_dir_abs,
            '--dataset', args.dataset,
            '--algorithm', 'FineTuneSpecialistHead',
            '--test_envs', *current_run_test_envs, # This isolates the training data to domain_idx
            '--output_dir', stage2_output_dir,
            '--hparams', finetune_hparams_str,
            '--steps', str(args.stage2_steps),
        ]
        subprocess.run(stage2_command, input='y\n', text=True, check=True) # Added check=True

        # The output of this run becomes the input for the next fine-tuning step
        current_model_path = os.path.join(stage2_output_dir, 'model.pkl')
        print("-" * 30)

    print(f"Staged training complete. Final model saved at: {current_model_path}")
    final_dest_path = os.path.join(args.output_dir, 'final_model.pkl')
    shutil.copy(current_model_path, final_dest_path)
    print(f"Copied final model to: {final_dest_path}")