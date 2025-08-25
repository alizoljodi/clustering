#!/usr/bin/env python3
"""
Comprehensive run script for testing both clustering and global tensors approaches.
This script allows users to choose between different approaches and testing scenarios.
"""

import os
import argparse
import time
import subprocess

def run_experiment(cmd, description):
    """Run a single experiment and handle errors gracefully."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"✅ Experiment completed successfully in {end_time - start_time:.2f} seconds")
        print(f"Output preview (last 10 lines):")
        
        # Show last 10 lines of output
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines[-10:]:
            print(f"  {line}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_clustering_experiments(arch, data_path, seeds, w_bits, a_bits, alpha_list, cluster_list, pca_list):
    """Run clustering-based experiments."""
    print(f"\n🔍 Running CLUSTERING experiments for {arch}")
    
    success_count = 0
    total_count = 0
    
    for seed in seeds:
        for i in range(len(w_bits)):
            for alpha in alpha_list:
                for num_clusters in cluster_list:
                    for pca_dim in pca_list:
                        total_count += 1
                        
                        # Get model-specific parameters
                        if arch == "mobilenetv2":
                            weight, T, lamb_c = 0.1, 1.0, 0.005
                        elif arch == "mnasnet":
                            weight, T, lamb_c = 0.2, 1.0, 0.001
                        else:
                            weight, T, lamb_c = 0.01, 4.0, 0.02
                        
                        cmd = [
                            "python", "main_imagenet.py",
                            "--data_path", data_path,
                            "--arch", arch,
                            "--seed", str(seed),
                            "--n_bits_w", str(w_bits[i]),
                            "--n_bits_a", str(a_bits[i]),
                            "--weight", str(weight),
                            "--T", str(T),
                            "--lamb_c", str(lamb_c),
                            "--alpha", str(alpha),
                            "--num_clusters", str(num_clusters),
                            "--pca_dim", str(pca_dim)
                        ]
                        
                        description = f"{arch} - Seed {seed}, W{w_bits[i]}A{a_bits[i]}, α={alpha}, Clusters={num_clusters}, PCA={pca_dim}"
                        
                        if run_experiment(cmd, description):
                            success_count += 1
                        
                        time.sleep(0.5)  # Small delay between experiments
    
    print(f"\n📊 Clustering experiments completed: {success_count}/{total_count} successful")
    return success_count, total_count

def run_global_tensors_experiments(arch, data_path, seeds, w_bits, a_bits, alpha_list):
    """Run global tensors experiments."""
    print(f"\n🌐 Running GLOBAL TENSORS experiments for {arch}")
    
    success_count = 0
    total_count = 0
    
    for seed in seeds:
        for i in range(len(w_bits)):
            for alpha in alpha_list:
                total_count += 1
                
                # Get model-specific parameters
                if arch == "mobilenetv2":
                    weight, T, lamb_c = 0.1, 1.0, 0.005
                elif arch == "mnasnet":
                    weight, T, lamb_c = 0.2, 1.0, 0.001
                else:
                    weight, T, lamb_c = 0.01, 4.0, 0.02
                
                cmd = [
                    "python", "main_imagenet.py",
                    "--data_path", data_path,
                    "--arch", arch,
                    "--seed", str(seed),
                    "--n_bits_w", str(w_bits[i]),
                    "--n_bits_a", str(a_bits[i]),
                    "--weight", str(weight),
                    "--T", str(T),
                    "--lamb_c", str(lamb_c),
                    "--alpha", str(alpha),
                    "--use_global_tensors"
                ]
                
                description = f"{arch} - Seed {seed}, W{w_bits[i]}A{a_bits[i]}, α={alpha}, Global Tensors"
                
                if run_experiment(cmd, description):
                    success_count += 1
                
                time.sleep(0.5)  # Small delay between experiments
    
    print(f"\n📊 Global tensors experiments completed: {success_count}/{total_count} successful")
    return success_count, total_count

def run_quick_comparison(arch, data_path, seed=1001):
    """Run a quick comparison between clustering and global tensors approaches."""
    print(f"\n⚡ Running QUICK COMPARISON for {arch} (Seed {seed})")
    
    w_bits, a_bits = [4], [4]  # Just test 4-bit quantization
    alpha_list = [0.4]  # Just test one alpha value
    cluster_list = [32]  # Just test one cluster configuration
    pca_list = [50]  # Just test one PCA dimension
    
    # Test clustering approach
    print(f"\n🔍 Testing clustering approach...")
    clustering_success, clustering_total = run_clustering_experiments(
        arch, data_path, [seed], w_bits, a_bits, alpha_list, cluster_list, pca_list
    )
    
    # Test global tensors approach
    print(f"\n🌐 Testing global tensors approach...")
    global_success, global_total = run_global_tensors_experiments(
        arch, data_path, [seed], w_bits, a_bits, alpha_list
    )
    
    print(f"\n📊 Quick comparison results:")
    print(f"  Clustering: {clustering_success}/{clustering_total} successful")
    print(f"  Global Tensors: {global_success}/{global_total} successful")
    
    return clustering_success, clustering_total, global_success, global_total

def main():
    parser = argparse.ArgumentParser(description='Comprehensive experiment runner for clustering vs global tensors')
    parser.add_argument("exp_name", type=str, 
                       choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument("--approach", type=str, choices=['clustering', 'global', 'both', 'quick'], 
                       default='both', help='Which approach to test')
    parser.add_argument("--data_path", type=str, 
                       default='/datasets/imagenet', help='Path to ImageNet dataset')
    parser.add_argument("--seeds", type=int, nargs='+', 
                       default=[1001, 1002, 1003], help='Seeds to test')
    parser.add_argument("--alpha_list", type=float, nargs='+', 
                       default=[0.2, 0.4, 0.6], help='Alpha values to test')
    parser.add_argument("--cluster_list", type=int, nargs='+', 
                       default=[8, 16, 64], help='Number of clusters to test')
    parser.add_argument("--pca_list", type=int, nargs='+', 
                       default=[25, 50, 100], help='PCA dimensions to test')
    
    args = parser.parse_args()
    
    # Define quantization bit configurations
    w_bits = [2, 4, 2, 4]
    a_bits = [2, 2, 4, 4]
    
    # Adjust data path for resnet18 (different path in original script)
    if args.exp_name == "resnet18":
        data_path = "/home/alz07xz/imagenet"
    else:
        data_path = args.data_path
    
    print(f"🚀 Starting comprehensive experiments for {args.exp_name}")
    print(f"📁 Data path: {data_path}")
    print(f"🔢 Seeds: {args.seeds}")
    print(f"📊 Approach: {args.approach}")
    
    start_time = time.time()
    
    if args.approach == 'clustering':
        success, total = run_clustering_experiments(
            args.exp_name, data_path, args.seeds, w_bits, a_bits, 
            args.alpha_list, args.cluster_list, args.pca_list
        )
        
    elif args.approach == 'global':
        success, total = run_global_tensors_experiments(
            args.exp_name, data_path, args.seeds, w_bits, a_bits, args.alpha_list
        )
        
    elif args.approach == 'both':
        print(f"\n🔄 Running BOTH approaches for comprehensive comparison")
        
        # Run clustering experiments
        clustering_success, clustering_total = run_clustering_experiments(
            args.exp_name, data_path, args.seeds, w_bits, a_bits, 
            args.alpha_list, args.cluster_list, args.pca_list
        )
        
        # Run global tensors experiments
        global_success, global_total = run_global_tensors_experiments(
            args.exp_name, data_path, args.seeds, w_bits, a_bits, args.alpha_list
        )
        
        success = clustering_success + global_success
        total = clustering_total + global_total
        
        print(f"\n📊 Overall results:")
        print(f"  Clustering: {clustering_success}/{clustering_total} successful")
        print(f"  Global Tensors: {global_success}/{global_total} successful")
        print(f"  Total: {success}/{total} successful")
        
    elif args.approach == 'quick':
        clustering_success, clustering_total, global_success, global_total = run_quick_comparison(
            args.exp_name, data_path, args.seeds[0]
        )
        success = clustering_success + global_success
        total = clustering_total + global_total
    
    end_time = time.time()
    
    print(f"\n🎉 All experiments completed!")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"📈 Success rate: {success}/{total} ({100*success/total:.1f}%)")
    
    if args.approach == 'both':
        print(f"\n💡 Recommendation:")
        if clustering_success > global_success:
            print("  Clustering approach performed better in this run")
        elif global_success > clustering_success:
            print("  Global tensors approach performed better in this run")
        else:
            print("  Both approaches performed similarly")

if __name__ == "__main__":
    main()
