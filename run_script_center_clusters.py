import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    args = parser.parse_args()
    
    # Bit configurations to test for center loss experiments
    w_bits = [2, 4]  # Test both 2-bit and 4-bit weights
    a_bits = [2, 4]  # Test both 2-bit and 4-bit activations
    
    # Define cluster numbers to test
    cluster_numbers = [8, 32, 64, 128, 256]
    
    # Define lambda_center values to test (0.1 to 1.0)
    lambda_center_values = [0.1,  0.5,  1.0]
    
    # Fixed seed for reproducibility in center loss experiments
    seed = 1005
    
    # Fixed alpha and PCA dimension for center loss experiments
    alpha = 0.4
    pca_dim = 50
    
    if args.exp_name == "resnet18":
        for w_bit in w_bits:
            for a_bit in a_bits:
                for cluster_num in cluster_numbers:
                    for lambda_center in lambda_center_values:
                        os.system(f"python main_imagenet.py --data_path /home/alz07xz/imagenet --arch resnet18 --seed {seed} "
                                 f"--n_bits_w {w_bit} --n_bits_a {a_bit} --weight 0.01 --T 4.0 --lamb_c 0.02 "
                                 f"--alpha {alpha} --num_clusters {cluster_num} --pca_dim {pca_dim} "
                                 f"--lambda_center {lambda_center}")
                        time.sleep(0.5)

    if args.exp_name == "resnet50":
        for w_bit in w_bits:
            for a_bit in a_bits:
                for cluster_num in cluster_numbers:
                    for lambda_center in lambda_center_values:
                        os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch resnet50 --seed {seed} "
                                 f"--n_bits_w {w_bit} --n_bits_a {a_bit} --weight 0.01 --T 4.0 --lamb_c 0.02 "
                                 f"--alpha {alpha} --num_clusters {cluster_num} --pca_dim {pca_dim} "
                                 f"--lambda_center {lambda_center}")
                        time.sleep(0.5)

    if args.exp_name == "regnetx_600m":
        for w_bit in w_bits:
            for a_bit in a_bits:
                for cluster_num in cluster_numbers:
                    for lambda_center in lambda_center_values:
                        os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_600m --seed {seed} "
                                 f"--n_bits_w {w_bit} --n_bits_a {a_bit} --weight 0.01 --T 4.0 --lamb_c 0.01 "
                                 f"--alpha {alpha} --num_clusters {cluster_num} --pca_dim {pca_dim} "
                                 f"--lambda_center {lambda_center}")
                        time.sleep(0.5)
    
    if args.exp_name == "regnetx_3200m":
        for w_bit in w_bits:
            for a_bit in a_bits:
                for cluster_num in cluster_numbers:
                    for lambda_center in lambda_center_values:
                        os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_3200m --seed {seed} "
                                 f"--n_bits_w {w_bit} --n_bits_a {a_bit} --weight 0.01 --T 4.0 --lamb_c 0.01 "
                                 f"--alpha {alpha} --num_clusters {cluster_num} --pca_dim {pca_dim} "
                                 f"--lambda_center {lambda_center}")
                        time.sleep(0.5)
    
    if args.exp_name == "mobilenetv2":
        for w_bit in w_bits:
            for a_bit in a_bits:
                for cluster_num in cluster_numbers:
                    for lambda_center in lambda_center_values:
                        os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mobilenetv2 --seed {seed} "
                                 f"--n_bits_w {w_bit} --n_bits_a {a_bit} --weight 0.1 --T 1.0 --lamb_c 0.005 "
                                 f"--alpha {alpha} --num_clusters {cluster_num} --pca_dim {pca_dim} "
                                 f"--lambda_center {lambda_center}")
                        time.sleep(0.5)
    
    if args.exp_name == "mnasnet":
        for w_bit in w_bits:
            for a_bit in a_bits:
                for cluster_num in cluster_numbers:
                    for lambda_center in lambda_center_values:
                        os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mnasnet --seed {seed} "
                                 f"--n_bits_w {w_bit} --n_bits_a {a_bit} --weight 0.2 --T 1.0 --lamb_c 0.001 "
                                 f"--alpha {alpha} --num_clusters {cluster_num} --pca_dim {pca_dim} "
                                 f"--lambda_center {lambda_center}")
                        time.sleep(0.5)

    print(f"\nCenter Loss Experiments Completed for {args.exp_name}")
    print(f"Tested {len(w_bits)} weight bit configurations: {w_bits}")
    print(f"Tested {len(a_bits)} activation bit configurations: {a_bits}")
    print(f"Tested {len(cluster_numbers)} cluster numbers: {cluster_numbers}")
    print(f"Tested {len(lambda_center_values)} lambda_center values: {lambda_center_values}")
    print(f"Total experiments: {len(w_bits) * len(a_bits) * len(cluster_numbers) * len(lambda_center_values)}")
    print(f"Fixed parameters: seed={seed}, alpha={alpha}, pca_dim={pca_dim}")
