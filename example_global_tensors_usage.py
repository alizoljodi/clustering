#!/usr/bin/env python3
"""
Example script demonstrating how to use the global alpha/beta tensors approach
instead of clustering for logit refinement.

This approach computes single alpha and beta tensors for the entire dataset
without performing clustering, which can be faster and simpler for some use cases.
"""

import subprocess
import sys

def run_global_tensors_example():
    """
    Run the main script with the --use_global_tensors flag to use global alpha/beta tensors
    instead of clustering.
    """
    
    # Example command with global tensors approach
    cmd = [
        "python", "main_imagenet.py",
        "--arch", "resnet18",           # Use ResNet-18 model
        "--n_bits_w", "4",              # 4-bit weight quantization
        "--n_bits_a", "4",              # 4-bit activation quantization
        "--num_samples", "512",         # Use 512 samples for calibration
        "--batch_size", "32",           # Batch size
        "--alpha", "0.5",               # Alpha blending parameter
        "--use_global_tensors",         # Use global tensors instead of clustering
        "--seed", "42"                  # Random seed for reproducibility
    ]
    
    print("Running with global alpha/beta tensors approach...")
    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("1. Extract logits from quantized and full-precision models")
    print("2. Compute global alpha and beta tensors for the entire dataset")
    print("3. Apply global affine correction with alpha blending")
    print("4. Evaluate accuracy without clustering")
    print("\n" + "="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command completed successfully!")
        print("\nOutput:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: main_imagenet.py not found. Make sure you're in the correct directory.")
        return False
    
    return True

def run_clustering_example():
    """
    Run the main script with clustering approach (original behavior).
    """
    
    # Example command with clustering approach
    cmd = [
        "python", "main_imagenet.py",
        "--arch", "resnet18",           # Use ResNet-18 model
        "--n_bits_w", "4",              # 4-bit weight quantization
        "--n_bits_a", "4",              # 4-bit activation quantization
        "--num_samples", "512",         # Use 512 samples for calibration
        "--batch_size", "32",           # Batch size
        "--alpha", "0.5",               # Alpha blending parameter
        "--num_clusters", "32",         # Number of clusters
        "--pca_dim", "50",              # PCA dimension for clustering
        "--seed", "42"                  # Random seed for reproducibility
    ]
    
    print("Running with clustering approach...")
    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("1. Extract logits from quantized and full-precision models")
    print("2. Perform clustering on the logits")
    print("3. Apply per-cluster affine correction with alpha blending")
    print("4. Evaluate accuracy with clustering")
    print("\n" + "="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command completed successfully!")
        print("\nOutput:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: main_imagenet.py not found. Make sure you're in the correct directory.")
        return False
    
    return True

if __name__ == "__main__":
    print("Global Alpha/Beta Tensors vs Clustering Example")
    print("=" * 50)
    
    print("\nChoose an approach:")
    print("1. Global Alpha/Beta Tensors (no clustering)")
    print("2. Clustering approach")
    print("3. Show both approaches")
    
    try:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            run_global_tensors_example()
        elif choice == "2":
            run_clustering_example()
        elif choice == "3":
            print("\n" + "="*60)
            print("RUNNING BOTH APPROACHES FOR COMPARISON")
            print("="*60)
            
            print("\nFirst, running with global tensors...")
            run_global_tensors_example()
            
            print("\n" + "="*60)
            print("Now running with clustering...")
            print("="*60)
            run_clustering_example()
            
        else:
            print("Invalid choice. Please run the script again and choose 1, 2, or 3.")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
