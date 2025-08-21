#!/usr/bin/env python3
"""
Test script to verify the logits saving functionality.
"""

import torch
import pandas as pd
import os
import numpy as np
from main_imagenet import save_logits_to_csv, save_logits_in_chunks, create_logits_summary_csv

def test_logits_saving():
    """Test the logits saving functionality with dummy data."""
    
    # Create dummy logits data
    num_samples = 100
    num_classes = 1000
    
    # Create dummy logits (small dataset)
    all_q_logits = [torch.randn(32, num_classes) for _ in range(4)]  # 4 batches of 32 samples
    all_fp_logits = [torch.randn(32, num_classes) for _ in range(4)]
    all_corrected_logits = [torch.randn(32, num_classes) for _ in range(4)]
    
    # Test directory
    test_dir = "test_logits_output"
    os.makedirs(test_dir, exist_ok=True)
    
    print("Testing logits saving functionality...")
    
    # Test 1: Save small dataset (should not use chunking)
    print("\nTest 1: Small dataset (no chunking)")
    success = save_logits_to_csv(all_q_logits, all_fp_logits, all_corrected_logits, 
                                test_dir, "resnet18", 4, 4, 1001, chunk_size=1000)
    print(f"Small dataset save: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 2: Save large dataset (should use chunking)
    print("\nTest 2: Large dataset (with chunking)")
    large_q_logits = [torch.randn(100, num_classes) for _ in range(20)]  # 20 batches of 100 samples = 2000 total
    large_fp_logits = [torch.randn(100, num_classes) for _ in range(20)]
    large_corrected_logits = [torch.randn(100, num_classes) for _ in range(20)]
    
    success = save_logits_to_csv(large_q_logits, large_fp_logits, large_corrected_logits, 
                                test_dir, "resnet50", 2, 2, 1002, chunk_size=1000)
    print(f"Large dataset save: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 3: Create summary CSV
    print("\nTest 3: Create summary CSV")
    results_summary = [
        {'alpha': 0.2, 'num_clusters': 8, 'pca_dim': 25, 'top1_accuracy': 75.5, 'top5_accuracy': 92.3},
        {'alpha': 0.4, 'num_clusters': 16, 'pca_dim': 50, 'top1_accuracy': 76.2, 'top5_accuracy': 92.8},
        {'alpha': 0.6, 'num_clusters': 64, 'pca_dim': 100, 'top1_accuracy': 75.8, 'top5_accuracy': 92.5}
    ]
    
    summary_csv = create_logits_summary_csv("resnet18", 4, 4, 1001, results_summary)
    print(f"Summary CSV creation: {'SUCCESS' if summary_csv else 'FAILED'}")
    
    # Test 4: Verify files were created
    print("\nTest 4: Verify files were created")
    expected_files = [
        os.path.join(test_dir, "logits_resnet18_w4bit_a4bit_seed1001_quantized.csv"),
        os.path.join(test_dir, "logits_resnet18_w4bit_a4bit_seed1001_fullprecision.csv"),
        os.path.join(test_dir, "logits_resnet18_w4bit_a4bit_seed1001_corrected.csv"),
        os.path.join(test_dir, "logits_resnet18_w4bit_a4bit_seed1001_metadata.csv")
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)} exists")
        else:
            print(f"✗ {os.path.basename(file_path)} missing")
    
    # Test 5: Verify chunked files for large dataset
    print("\nTest 5: Verify chunked files for large dataset")
    chunked_files = [
        os.path.join(test_dir, "logits_resnet50_w2bit_a2bit_seed1002_quantized_chunk001_of_002.csv"),
        os.path.join(test_dir, "logits_resnet50_w2bit_a2bit_seed1002_quantized_chunk002_of_002.csv"),
        os.path.join(test_dir, "logits_resnet50_w2bit_a2bit_seed1002_chunked_metadata.csv")
    ]
    
    for file_path in chunked_files:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)} exists")
        else:
            print(f"✗ {os.path.basename(file_path)} missing")
    
    print(f"\nTest completed. Check the '{test_dir}' directory for output files.")
    
    # Clean up test files
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    test_logits_saving()

