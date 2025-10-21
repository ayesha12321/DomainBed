import os
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Generalist and Fine-Tune Specialists')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_envs', type=int, nargs='+', required=True)
    args = parser.parse_args()

    # Convert to absolute path to prevent subprocess issues
    data_dir_abs = os.path.abspath(args.data_dir)
    print(f"Using absolute data path: {data_dir_abs}")

    # Configure dataset parameters
    if args.dataset == 'PACS':
        num_domains = 4
    elif args.dataset == 'VLCS':
        num_domains = 4
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not configured.")

    all_domains = list(range(num_domains))
    final_test_env = args.test_envs[0]
    source_domains = [d for d in all_domains if d != final_test_env]

    print(f"Dataset: {args.dataset}")
    print(f"Final Test Environment: {final_test_env}")
    print(f"Source Domains: {source_domains}")
    print("-" * 30)

    # --- Define Hyperparameters ---
    gen_hparams_str = '{"batch_size": 8, "resnet18": true, "resnet50_augmix": false, "resnet18_pretrained": true, "lr": 5e-4}'

    sp_hparams_str = '{"batch_size": 8, "resnet18": true, "resnet50_augmix": false, "resnet18_pretrained": true, "lr": 5e-4}'
    sp_steps = 501  # Fine-tune for a smaller number of steps (e.g., one epoch).

    # --- STAGE 1: Train the Generalist (ERM) Model ---
    print(">>> STAGE 1: Training Generalist Model...")
    generalist_output_dir = os.path.join(args.output_dir, 'generalist')
    generalist_model_path = os.path.join(generalist_output_dir, 'model.pkl')
    
    generalist_command = [
        'python', '-m', 'domainbed.scripts.train',
        '--data_dir', data_dir_abs,
        '--dataset', args.dataset,
        '--algorithm', args.algorithm,
        '--test_envs', str(final_test_env),
        '--output_dir', generalist_output_dir,
        '--holdout_fraction', '0.2',
        '--hparams', gen_hparams_str
    ]
    subprocess.run(generalist_command, input='y\n', text=True)
    print("-" * 30)

    # --- STAGE 2: Fine-Tune Specialist Models from the Generalist ---
    print(f">>> STAGE 2: Fine-Tuning Specialists from {generalist_model_path}...")
    for domain_idx in source_domains:
        print(f"Fine-Tuning Specialist Model for domain: {domain_idx}")
        specialist_output_dir = os.path.join(args.output_dir, f'specialist_domain_{domain_idx}')
        
        # To train on one domain, treat all others as test domains
        specialist_test_envs = [str(d) for d in all_domains if d != domain_idx]

        specialist_command = [
            'python', '-m', 'domainbed.scripts.train',
            '--data_dir', data_dir_abs,
            '--dataset', args.dataset,
            '--algorithm', args.algorithm,
            '--test_envs', *specialist_test_envs,
            '--output_dir', specialist_output_dir,
            '--hparams', sp_hparams_str,
            '--steps', str(sp_steps),
            '--holdout_fraction', '0.2',
            '--load_model', generalist_model_path  # <-- Key: Load the generalist's weights
        ]
        subprocess.run(specialist_command, input='y\n', text=True)
        print("-" * 30)

    print("All expert models have been trained and fine-tuned successfully.")