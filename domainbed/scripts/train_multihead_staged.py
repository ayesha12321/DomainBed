import os
import subprocess
import argparse
import json 
import shutil
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MultiHead Staged: Generalist -> Specialists -> Router')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_envs', type=int, nargs='+', required=True)
    parser.add_argument('--stage2_steps', type=int, default=501, help='Steps for Stage 2 (Specialists)')
    parser.add_argument('--hparams', type=str, default=None, help='JSON overrides (e.g., {"use_cosine_router": true})')

    args = parser.parse_args()

    # --- Base Hyperparameters (ResNet-18 Default) ---
    base_hparams = {
        "batch_size": 16,             # Increased to 16 for stability
        "resnet18": True,             # Force ResNet18
        "resnet18_pretrained": True,
        "lr": 5e-4,
        "resnet50_augmix": False,
        "weight_decay": 0.0,
        "use_cosine_router": False    # Default off, enable via CLI
    }

    # Apply Command Line Overrides
    if args.hparams:
        overrides = json.loads(args.hparams)
        print(f"Applying overrides: {overrides}")
        base_hparams.update(overrides)

    stage1_hparams_str = json.dumps(base_hparams)
    # ---------------------------------

    data_dir_abs = os.path.abspath(args.data_dir)
    if args.dataset == 'PACS': num_domains = 4
    elif args.dataset == 'VLCS': num_domains = 4
    else: raise ValueError(f"Dataset '{args.dataset}' is not configured.")

    all_domains = list(range(num_domains))
    final_test_env = args.test_envs[0]
    source_domains = [d for d in all_domains if d != final_test_env]

    print(f"Dataset: {args.dataset}")
    print(f"Source Domains: {source_domains}")
    print(f"HParams: {base_hparams}")
    print("-" * 30)

    # ==========================================
    # STAGE 1: Train Generalist Head + Backbone
    # ==========================================
    print(">>> STAGE 1: Training Generalist Head + Backbone...")
    stage1_output_dir = os.path.join(args.output_dir, 'stage1_generalist')
    
    stage1_command = [
        sys.executable, '-m', 'domainbed.scripts.train',
        '--data_dir', data_dir_abs,
        '--dataset', args.dataset,
        '--algorithm', 'ERMGeneralistHeadOnly',
        '--test_envs', str(final_test_env),
        '--output_dir', stage1_output_dir,
        '--hparams', stage1_hparams_str
    ]
    subprocess.run(stage1_command, input='y\n', text=True, check=True)
    
    stage1_model_path = os.path.join(stage1_output_dir, 'model.pkl')
    print("-" * 30)

    # ==========================================
    # STAGE 2: Fine-Tune Specialist Heads
    # ==========================================
    print(f">>> STAGE 2: Fine-Tuning Specialist Heads from {stage1_model_path}...")
    current_model_path = stage1_model_path 

    for i, domain_idx in enumerate(source_domains):
        specialist_head_idx = i + 1 
        print(f"Fine-Tuning Head {specialist_head_idx} for Source Domain {domain_idx}...")

        stage2_output_dir = os.path.join(args.output_dir, f'stage2_head_{specialist_head_idx}_dom_{domain_idx}')

        finetune_hparams = base_hparams.copy()
        finetune_hparams['load_trained_model_path'] = current_model_path
        finetune_hparams['finetune_head_idx'] = specialist_head_idx
        # Critical: Tell the algorithm how many heads to expect (Gen + Specialists)
        finetune_hparams['num_heads'] = len(source_domains) + 1 
        
        current_run_test_envs = [str(d) for d in all_domains if d != domain_idx]

        stage2_command = [
            sys.executable, '-m', 'domainbed.scripts.train',
            '--data_dir', data_dir_abs,
            '--dataset', args.dataset,
            '--algorithm', 'FineTuneSpecialistHead',
            '--test_envs', *current_run_test_envs,
            '--output_dir', stage2_output_dir,
            '--hparams', json.dumps(finetune_hparams),
            '--steps', str(args.stage2_steps),
        ]
        subprocess.run(stage2_command, input='y\n', text=True, check=True)
        current_model_path = os.path.join(stage2_output_dir, 'model.pkl')
        print("-" * 30)

    # ==========================================
    # STAGE 3: Train Cosine Router (OPTIONAL)
    # ==========================================
    if base_hparams.get('use_cosine_router', False):
        print(f">>> STAGE 3: Training Cosine Router from {current_model_path}...")
        
        stage3_output_dir = os.path.join(args.output_dir, 'stage3_router')
        
        router_hparams = base_hparams.copy()
        router_hparams['load_trained_model_path'] = current_model_path
        router_hparams['use_cosine_router'] = True 
        router_hparams['num_heads'] = len(source_domains) + 1

        stage3_command = [
            sys.executable, '-m', 'domainbed.scripts.train',
            '--data_dir', data_dir_abs,
            '--dataset', args.dataset,
            '--algorithm', 'TrainRouterOnly',
            '--test_envs', str(final_test_env), 
            '--output_dir', stage3_output_dir,
            '--hparams', json.dumps(router_hparams),
            '--steps', str(args.stage2_steps) 
        ]
        subprocess.run(stage3_command, input='y\n', text=True, check=True)
        
        current_model_path = os.path.join(stage3_output_dir, 'model.pkl')
        print("-" * 30)
    else:
        print(">>> Skipping Stage 3 (Router) because 'use_cosine_router' is False.")

    # ==========================================
    # Finalize
    # ==========================================
    print(f"Staged training complete.")
    final_dest_path = os.path.join(args.output_dir, 'final_model.pkl')
    shutil.copy(current_model_path, final_dest_path)
    print(f"Copied final model to: {final_dest_path}")