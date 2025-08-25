import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    args = parser.parse_args()
    w_bits = [2, 4, 2, 4]
    a_bits = [2, 2, 4, 4]
    
    # Define 10 different seeds for robust testing
    seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    
    if args.exp_name == "resnet18":
        for seed in seeds:
            for i in range(4):
                os.system(f"python main_imagenet.py --data_path /home/alz07xz/imagenet --arch resnet18 --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02 --alpha_list 0.2 0.4 0.6 --use_global_tensors")
                time.sleep(0.5)

    if args.exp_name == "resnet50":
        for seed in seeds:
            for i in range(4):
                os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch resnet50 --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.02 --alpha_list 0.2 0.4 0.6 --use_global_tensors")
                time.sleep(0.5)

    if args.exp_name == "regnetx_600m":
        for seed in seeds:
            for i in range(4):
                os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_600m --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01 --alpha_list 0.2 0.4 0.6 --use_global_tensors")
                time.sleep(0.5)
    
    if args.exp_name == "regnetx_3200m":
        for seed in seeds:
            for i in range(4):
                os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch regnetx_3200m --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01 --T 4.0 --lamb_c 0.01 --alpha_list 0.2 0.4 0.6 --use_global_tensors")
                time.sleep(0.5)
    
    if args.exp_name == "mobilenetv2":
        for seed in seeds:
            for i in range(4):
                os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mobilenetv2 --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.1 --T 1.0 --lamb_c 0.005 --alpha_list 0.2 0.4 0.6 --use_global_tensors")
                time.sleep(0.5)
    
    if args.exp_name == "mnasnet":
        for seed in seeds:
            for i in range(4):
                os.system(f"python main_imagenet.py --data_path /datasets/imagenet --arch mnasnet --seed {seed} --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.2 --T 1.0 --lamb_c 0.001 --alpha_list 0.2 0.4 0.6 --use_global_tensors")
                time.sleep(0.5)
