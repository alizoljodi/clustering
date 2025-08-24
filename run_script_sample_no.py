import os
import argparse
import time
import gc
import torch

def cleanup_memory():
    """Clean up PyTorch memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    args = parser.parse_args()
    w_bits = [2, 4, 2, 4]
    a_bits = [2, 2, 4, 4]
    
    # Define sample numbers to test - covering entire ImageNet training set domain
    # ImageNet has ~1.2M training images, so we'll test from 1K to 1.2M
    sample_numbers = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 750000, 1200000]
    
    # Define 10 different seeds for robust testing
    seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    
    if args.exp_name == "resnet18":
        for seed in seeds:
            for i in range(4):
                for sample_no in sample_numbers:
                    print(f"\n{'='*80}")
                    print(f"Running ResNet18 - Seed: {seed}, W-bits: {w_bits[i]}, A-bits: {a_bits[i]}, Samples: {sample_no}")
                    print(f"{'='*80}")
                    os.system(f"python main_imagenet.py --data_path /home/alz07xz/imagenet --arch resnet18 --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02 --alpha_list 0.6 --num_clusters_list 64 --pca_dim_list 100 --num_samples {sample_no}")
                    cleanup_memory()
                    time.sleep(0.5)

    if args.exp_name == "resnet50":
        for seed in seeds:
            for i in range(4):
                for sample_no in sample_numbers:
                    print(f"\n{'='*80}")
                    print(f"Running ResNet50 - Seed: {seed}, W-bits: {w_bits[i]}, A-bits: {a_bits[i]}, Samples: {sample_no}")
                    print(f"{'='*80}")
                    os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch resnet50 --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02 --alpha_list 0.2 0.4 0.6 --num_clusters_list 8 16 64 --pca_dim_list 25 50 100 --num_samples {sample_no}")
                    cleanup_memory()
                    time.sleep(0.5)

    if args.exp_name == "regnetx_600m":
        for seed in seeds:
            for i in range(4):
                for sample_no in sample_numbers:
                    print(f"\n{'='*80}")
                    print(f"Running RegNetX-600M - Seed: {seed}, W-bits: {w_bits[i]}, A-bits: {a_bits[i]}, Samples: {sample_no}")
                    print(f"{'='*80}")
                    os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_600m --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01 --alpha_list 0.2 0.4 0.6 --num_clusters_list 8 16 64 --pca_dim_list 25 50 100 --num_samples {sample_no}")
                    cleanup_memory()
                    time.sleep(0.5)
    
    if args.exp_name == "regnetx_3200m":
        for seed in seeds:
            for i in range(4):
                for sample_no in sample_numbers:
                    print(f"\n{'='*80}")
                    print(f"Running RegNetX-3200M - Seed: {seed}, W-bits: {w_bits[i]}, A-bits: {a_bits[i]}, Samples: {sample_no}")
                    print(f"{'='*80}")
                    os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_3200m --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01 --alpha_list 0.2 0.4 0.6 --num_clusters_list 8 16 64 --pca_dim_list 25 50 100 --num_samples {sample_no}")
                    cleanup_memory()
                    time.sleep(0.5)
    
    if args.exp_name == "mobilenetv2":
        for seed in seeds:
            for i in range(4):
                for sample_no in sample_numbers:
                    print(f"\n{'='*80}")
                    print(f"Running MobileNetV2 - Seed: {seed}, W-bits: {w_bits[i]}, A-bits: {a_bits[i]}, Samples: {sample_no}")
                    print(f"{'='*80}")
                    os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mobilenetv2 --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.1 --T 1.0 --lamb_c 0.005 --alpha_list 0.2 0.4 0.6 --num_clusters_list 8 16 64 --pca_dim_list 25 50 100 --num_samples {sample_no}")
                    cleanup_memory()
                    time.sleep(0.5)
    
    if args.exp_name == "mnasnet":
        for seed in seeds:
            for i in range(4):
                for sample_no in sample_numbers:
                    print(f"\n{'='*80}")
                    print(f"Running MnasNet - Seed: {seed}, W-bits: {w_bits[i]}, A-bits: {a_bits[i]}, Samples: {sample_no}")
                    print(f"{'='*80}")
                    os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mnasnet --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.2 --T 1.0 --lamb_c 0.001 --alpha_list 0.2 0.4 0.6 --num_clusters_list 8 16 64 --pca_dim_list 25 50 100 --num_samples {sample_no}")
                    cleanup_memory()
                    time.sleep(0.5)

    