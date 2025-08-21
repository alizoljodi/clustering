#!/usr/bin/env python3
"""
Example script showing how to load and analyze the saved logits data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def load_logits_data(results_dir, logits_type="quantized"):
    """
    Load logits data from CSV files.
    
    Args:
        results_dir: Directory containing the logits CSV files
        logits_type: Type of logits to load ("quantized", "fullprecision", "corrected")
    
    Returns:
        logits_data: numpy array of logits
        metadata: dictionary of metadata
    """
    
    # Find the appropriate CSV file
    pattern = f"*_{logits_type}.csv"
    csv_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not csv_files:
        print(f"No {logits_type} logits files found in {results_dir}")
        return None, None
    
    # Load the first matching file
    csv_file = csv_files[0]
    print(f"Loading {logits_type} logits from: {csv_file}")
    
    # Load logits data
    logits_data = pd.read_csv(csv_file, header=None).values
    print(f"Loaded logits shape: {logits_data.shape}")
    
    # Try to load metadata
    metadata_file = csv_file.replace(f"_{logits_type}.csv", "_metadata.csv")
    metadata = {}
    if os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file)
        metadata = metadata_df.iloc[0].to_dict()
        print(f"Loaded metadata: {metadata}")
    
    return logits_data, metadata

def analyze_logits_comparison(results_dir):
    """
    Analyze and compare different types of logits.
    """
    
    print(f"Analyzing logits in: {results_dir}")
    
    # Load all three types of logits
    logits_types = ["quantized", "fullprecision", "corrected"]
    logits_data = {}
    metadata = {}
    
    for logits_type in logits_types:
        data, meta = load_logits_data(results_dir, logits_type)
        if data is not None:
            logits_data[logits_type] = data
            metadata[logits_type] = meta
    
    if not logits_data:
        print("No logits data found!")
        return
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    for logits_type, data in logits_data.items():
        print(f"\n{logits_type.capitalize()} Logits:")
        print(f"  Shape: {data.shape}")
        print(f"  Mean: {np.mean(data):.4f}")
        print(f"  Std: {np.std(data):.4f}")
        print(f"  Min: {np.min(data):.4f}")
        print(f"  Max: {np.max(data):.4f}")
    
    # Compare distributions
    print("\n=== Distribution Comparison ===")
    
    # Flatten all logits for histogram comparison
    plt.figure(figsize=(15, 5))
    
    for i, (logits_type, data) in enumerate(logits_data.items()):
        plt.subplot(1, 3, i+1)
        plt.hist(data.flatten(), bins=50, alpha=0.7, label=logits_type)
        plt.title(f'{logits_type.capitalize()} Logits Distribution')
        plt.xlabel('Logit Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'logits_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Distribution comparison plot saved to: {results_dir}")
    
    # Error analysis (if we have both quantized and full-precision)
    if "quantized" in logits_data and "fullprecision" in logits_data:
        print("\n=== Error Analysis ===")
        
        q_logits = logits_data["quantized"]
        fp_logits = logits_data["fullprecision"]
        
        # Mean squared error per sample
        mse_per_sample = np.mean((q_logits - fp_logits) ** 2, axis=1)
        print(f"  Mean MSE per sample: {np.mean(mse_per_sample):.6f}")
        print(f"  Std MSE per sample: {np.std(mse_per_sample):.6f}")
        
        # Mean squared error per class
        mse_per_class = np.mean((q_logits - fp_logits) ** 2, axis=0)
        print(f"  Mean MSE per class: {np.mean(mse_per_class):.6f}")
        print(f"  Std MSE per class: {np.std(mse_per_class):.6f}")
        
        # Plot MSE distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(mse_per_sample, bins=50, alpha=0.7)
        plt.title('MSE Distribution per Sample')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(mse_per_class, bins=50, alpha=0.7)
        plt.title('MSE Distribution per Class')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'mse_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"MSE analysis plot saved to: {results_dir}")
    
    # Top-k accuracy analysis
    print("\n=== Top-k Analysis ===")
    
    for logits_type, data in logits_data.items():
        # Get top-5 predictions for each sample
        top5_indices = np.argsort(data, axis=1)[:, -5:]
        
        # Calculate entropy of top-5 probabilities
        top5_values = np.take_along_axis(data, top5_indices, axis=1)
        top5_probs = np.exp(top5_values) / np.sum(np.exp(data), axis=1, keepdims=True)
        
        # Shannon entropy
        entropy = -np.sum(top5_probs * np.log2(top5_probs + 1e-10), axis=1)
        
        print(f"  {logits_type.capitalize()} Logits:")
        print(f"    Mean top-5 entropy: {np.mean(entropy):.4f}")
        print(f"    Std top-5 entropy: {np.std(entropy):.4f}")
    
    plt.show()

def find_and_analyze_all_logits():
    """
    Find all logits directories and analyze them.
    """
    
    # Find all results directories
    results_dirs = glob.glob("results_*")
    initial_dirs = glob.glob("initial_logits_*")
    
    all_dirs = results_dirs + initial_dirs
    
    if not all_dirs:
        print("No logits directories found!")
        return
    
    print(f"Found {len(all_dirs)} logits directories:")
    for dir_path in all_dirs:
        print(f"  - {dir_path}")
    
    # Analyze each directory
    for dir_path in all_dirs:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dir_path}")
        print(f"{'='*60}")
        
        try:
            analyze_logits_comparison(dir_path)
        except Exception as e:
            print(f"Error analyzing {dir_path}: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    Main function to demonstrate logits analysis.
    """
    
    print("Logits Analysis Tool")
    print("===================")
    
    # Check if we have any logits data
    if not (glob.glob("results_*") or glob.glob("initial_logits_*")):
        print("\nNo logits data found!")
        print("Please run main_imagenet.py first to generate logits data.")
        print("\nExample usage:")
        print("  python main_imagenet.py --data_path /path/to/imagenet --arch resnet18 --seed 1001 --n_bits_w 4 --n_bits_a 4")
        return
    
    # Analyze all available logits data
    find_and_analyze_all_logits()
    
    print("\nAnalysis complete!")
    print("\nTo analyze specific results, use:")
    print("  analyze_logits_comparison('results_alpha0.4_clusters16_pca50_resnet18_w4bit_a4bit')")

if __name__ == "__main__":
    main()

