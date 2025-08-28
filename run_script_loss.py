#!/usr/bin/env python3
"""
Script to test different loss function ablation configurations.
This script mimics run_script.py but focuses on testing the involvement of different loss terms.
"""

import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    args = parser.parse_args()
    
    # Fixed parameters for loss function testing
    w_bits = [2, 4, 2, 4]
    a_bits = [2, 2, 4, 4]
    
    # Fixed cluster affine parameters as specified
    alpha_list = [0.6]
    num_clusters_list = [64]
    pca_dim_list = [50]
    
    # Define 5 different seeds for loss function testing (reduced from 10 for efficiency)
    seeds = [1001, 1002, 1003, 1004, 1005]
    
    # Loss function ablation configurations to test
    ablation_configs = [
        # Config 1: Full loss (all terms enabled)
        {
            'name': 'full_loss',
            'args': [],
            'description': 'Full Loss (rec_loss + round_loss + pd_loss)'
        },
        # Config 2: Ablate reconstruction loss only
        {
            'name': 'no_rec_loss',
            'args': ['--no_rec_loss'],
            'description': 'Ablate Reconstruction Loss (round_loss + pd_loss)'
        },
        # Config 3: Ablate rounding loss only
        {
            'name': 'no_round_loss',
            'args': ['--no_round_loss'],
            'description': 'Ablate Rounding Loss (rec_loss + pd_loss)'
        },
        # Config 4: Ablate prediction difference loss only
        {
            'name': 'no_pd_loss',
            'args': ['--no_pd_loss'],
            'description': 'Ablate Prediction Difference Loss (rec_loss + round_loss)'
        },
        # Config 5: Use only reconstruction loss
        {
            'name': 'rec_loss_only',
            'args': ['--no_round_loss', '--no_pd_loss'],
            'description': 'Reconstruction Loss Only (rec_loss)'
        },
        # Config 6: Use only rounding loss
        {
            'name': 'round_loss_only',
            'args': ['--no_rec_loss', '--no_pd_loss'],
            'description': 'Rounding Loss Only (round_loss)'
        },
        # Config 7: Use only prediction difference loss
        {
            'name': 'pd_loss_only',
            'args': ['--no_rec_loss', '--no_round_loss'],
            'description': 'Prediction Difference Loss Only (pd_loss)'
        }
    ]
    
    print(f"Testing loss function ablation configurations for {args.exp_name}")
    print(f"Fixed parameters: alpha={alpha_list}, clusters={num_clusters_list}, pca_dim={pca_dim_list}")
    print(f"Testing {len(ablation_configs)} different loss configurations")
    print(f"Using {len(seeds)} seeds for robustness")
    print("="*80)
    
    if args.exp_name == "resnet18":
        for seed in seeds:
            for i in range(4):
                for config in ablation_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing: {config['description']}")
                    print(f"Architecture: {args.exp_name}, Seed: {seed}, Config: {i+1}/4")
                    print(f"Bits: W{w_bits[i]}/A{a_bits[i]}")
                    print(f"{'='*60}")
                    
                    # Build command with all parameters
                    cmd_parts = [
                        "python", "main_imagenet.py",
                        "--data_path", "/home/alz07xz/imagenet",
                        "--arch", args.exp_name,
                        "--seed", str(seed),
                        "--n_bits_w", str(w_bits[i]),
                        "--n_bits_a", str(a_bits[i]),
                        "--weight", "0.01",
                        "--T", "4.0",
                        "--lamb_c", "0.02",
                        "--alpha_list", "0.6",
                        "--num_clusters_list", "64",
                        "--pca_dim_list", "50"
                    ]
                    
                    # Add ablation arguments
                    cmd_parts.extend(config['args'])
                    
                    # Execute command
                    cmd = " ".join(cmd_parts)
                    print(f"Executing: {cmd}")
                    
                    try:
                        os.system(cmd)
                        print(f"✓ Completed: {config['name']} for {args.exp_name} seed {seed} config {i+1}")
                    except Exception as e:
                        print(f"✗ Error in {config['name']}: {e}")
                    
                    time.sleep(0.5)

    elif args.exp_name == "resnet50":
        for seed in seeds:
            for i in range(4):
                for config in ablation_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing: {config['description']}")
                    print(f"Architecture: {args.exp_name}, Seed: {seed}, Config: {i+1}/4")
                    print(f"Bits: W{w_bits[i]}/A{a_bits[i]}")
                    print(f"{'='*60}")
                    
                    cmd_parts = [
                        "python", "main_imagenet.py",
                        "--data_path", "/datasets/imagenet",
                        "--arch", args.exp_name,
                        "--seed", str(seed),
                        "--n_bits_w", str(w_bits[i]),
                        "--n_bits_a", str(a_bits[i]),
                        "--weight", "0.01",
                        "--T", "4.0",
                        "--lamb_c", "0.02",
                        "--alpha_list", "0.6",
                        "--num_clusters_list", "64",
                        "--pca_dim_list", "50"
                    ]
                    
                    cmd_parts.extend(config['args'])
                    cmd = " ".join(cmd_parts)
                    print(f"Executing: {cmd}")
                    
                    try:
                        os.system(cmd)
                        print(f"✓ Completed: {config['name']} for {args.exp_name} seed {seed} config {i+1}")
                    except Exception as e:
                        print(f"✗ Error in {config['name']}: {e}")
                    
                    time.sleep(0.5)

    elif args.exp_name == "regnetx_600m":
        for seed in seeds:
            for i in range(4):
                for config in ablation_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing: {config['description']}")
                    print(f"Architecture: {args.exp_name}, Seed: {seed}, Config: {i+1}/4")
                    print(f"Bits: W{w_bits[i]}/A{a_bits[i]}")
                    print(f"{'='*60}")
                    
                    cmd_parts = [
                        "python", "main_imagenet.py",
                        "--data_path", "/datasets/imagenet",
                        "--arch", args.exp_name,
                        "--seed", str(seed),
                        "--n_bits_w", str(w_bits[i]),
                        "--n_bits_a", str(a_bits[i]),
                        "--weight", "0.01",
                        "--T", "4.0",
                        "--lamb_c", "0.01",
                        "--alpha_list", "0.6",
                        "--num_clusters_list", "64",
                        "--pca_dim_list", "50"
                    ]
                    
                    cmd_parts.extend(config['args'])
                    cmd = " ".join(cmd_parts)
                    print(f"Executing: {cmd}")
                    
                    try:
                        os.system(cmd)
                        print(f"✓ Completed: {config['name']} for {args.exp_name} seed {seed} config {i+1}")
                    except Exception as e:
                        print(f"✗ Error in {config['name']}: {e}")
                    
                    time.sleep(0.5)
    
    elif args.exp_name == "regnetx_3200m":
        for seed in seeds:
            for i in range(4):
                for config in ablation_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing: {config['description']}")
                    print(f"Architecture: {args.exp_name}, Seed: {seed}, Config: {i+1}/4")
                    print(f"Bits: W{w_bits[i]}/A{a_bits[i]}")
                    print(f"{'='*60}")
                    
                    cmd_parts = [
                        "python", "main_imagenet.py",
                        "--data_path", "/datasets/imagenet",
                        "--arch", args.exp_name,
                        "--seed", str(seed),
                        "--n_bits_w", str(w_bits[i]),
                        "--n_bits_a", str(a_bits[i]),
                        "--weight", "0.01",
                        "--T", "4.0",
                        "--lamb_c", "0.01",
                        "--alpha_list", "0.6",
                        "--num_clusters_list", "64",
                        "--pca_dim_list", "50"
                    ]
                    
                    cmd_parts.extend(config['args'])
                    cmd = " ".join(cmd_parts)
                    print(f"Executing: {cmd}")
                    
                    try:
                        os.system(cmd)
                        print(f"✓ Completed: {config['name']} for {args.exp_name} seed {seed} config {i+1}")
                    except Exception as e:
                        print(f"✗ Error in {config['name']}: {e}")
                    
                    time.sleep(0.5)
    
    elif args.exp_name == "mobilenetv2":
        for seed in seeds:
            for i in range(4):
                for config in ablation_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing: {config['description']}")
                    print(f"Architecture: {args.exp_name}, Seed: {seed}, Config: {i+1}/4")
                    print(f"Bits: W{w_bits[i]}/A{a_bits[i]}")
                    print(f"{'='*60}")
                    
                    cmd_parts = [
                        "python", "main_imagenet.py",
                        "--data_path", "/datasets/imagenet",
                        "--arch", args.exp_name,
                        "--seed", str(seed),
                        "--n_bits_w", str(w_bits[i]),
                        "--n_bits_a", str(a_bits[i]),
                        "--weight", "0.1",
                        "--T", "1.0",
                        "--lamb_c", "0.005",
                        "--alpha_list", "0.6",
                        "--num_clusters_list", "64",
                        "--pca_dim_list", "50"
                    ]
                    
                    cmd_parts.extend(config['args'])
                    cmd = " ".join(cmd_parts)
                    print(f"Executing: {cmd}")
                    
                    try:
                        os.system(cmd)
                        print(f"✓ Completed: {config['name']} for {args.exp_name} seed {seed} config {i+1}")
                    except Exception as e:
                        print(f"✗ Error in {config['name']}: {e}")
                    
                    time.sleep(0.5)
    
    elif args.exp_name == "mnasnet":
        for seed in seeds:
            for i in range(4):
                for config in ablation_configs:
                    print(f"\n{'='*60}")
                    print(f"Testing: {config['description']}")
                    print(f"Architecture: {args.exp_name}, Seed: {seed}, Config: {i+1}/4")
                    print(f"Bits: W{w_bits[i]}/A{a_bits[i]}")
                    print(f"{'='*60}")
                    
                    cmd_parts = [
                        "python", "main_imagenet.py",
                        "--data_path", "/datasets/imagenet",
                        "--arch", args.exp_name,
                        "--seed", str(seed),
                        "--n_bits_w", str(w_bits[i]),
                        "--n_bits_a", str(a_bits[i]),
                        "--weight", "0.2",
                        "--T", "1.0",
                        "--lamb_c", "0.001",
                        "--alpha_list", "0.6",
                        "--num_clusters_list", "64",
                        "--pca_dim_list", "50"
                    ]
                    
                    cmd_parts.extend(config['args'])
                    cmd = " ".join(cmd_parts)
                    print(f"Executing: {cmd}")
                    
                    try:
                        os.system(cmd)
                        print(f"✓ Completed: {config['name']} for {args.exp_name} seed {seed} config {i+1}")
                    except Exception as e:
                        print(f"✗ Error in {config['name']}: {e}")
                    
                    time.sleep(0.5)

    print(f"\n{'='*80}")
    print("LOSS FUNCTION ABLATION TESTING COMPLETED")
    print(f"{'='*80}")
    print(f"Architecture tested: {args.exp_name}")
    print(f"Total configurations tested: {len(ablation_configs)}")
    print(f"Total experiments: {len(seeds) * 4 * len(ablation_configs)}")
    print(f"Fixed parameters:")
    print(f"  - Alpha: {alpha_list}")
    print(f"  - Number of clusters: {num_clusters_list}")
    print(f"  - PCA dimensions: {pca_dim_list}")
    print(f"\nLoss configurations tested:")
    for i, config in enumerate(ablation_configs, 1):
        print(f"  {i}. {config['name']}: {config['description']}")
    print(f"\nResults will be saved in separate directories for each configuration.")
    print("Check the output logs for detailed results and any errors.")
