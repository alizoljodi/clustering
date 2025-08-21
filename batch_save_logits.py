#!/usr/bin/env python3
"""
Batch script to save logits for multiple models efficiently.
"""

import os
import subprocess
import time
import argparse
from pathlib import Path

def run_model_and_save_logits(arch, w_bits, a_bits, seed, data_path, additional_args=""):
    """
    Run a single model configuration and save its logits.
    """
    
    print(f"\n{'='*80}")
    print(f"Processing: {arch} - W{w_bits}bit A{a_bits}bit - Seed {seed}")
    print(f"{'='*80}")
    
    # Build command
    cmd = [
        "python", "main_imagenet.py",
        "--data_path", data_path,
        "--arch", arch,
        "--seed", str(seed),
        "--n_bits_w", str(w_bits),
        "--n_bits_a", str(a_bits),
        "--weight", "0.01",
        "--T", "4.0",
        "--lamb_c", "0.02",
        "--alpha_list", "0.2", "0.4", "0.6",
        "--num_clusters_list", "8", "16", "64",
        "--pca_dim_list", "25", "50", "100"
    ]
    
    # Add additional arguments if provided
    if additional_args:
        cmd.extend(additional_args.split())
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        print(f"✓ Successfully completed in {end_time - start_time:.2f} seconds")
        print(f"Output: {result.stdout[-500:]}")  # Last 500 characters
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"✗ Failed after {end_time - start_time:.2f} seconds")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def batch_process_models(architectures, data_path, seeds=None, additional_args=""):
    """
    Process multiple models in batch.
    """
    
    if seeds is None:
        seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    
    # Define bit configurations
    w_bits_list = [2, 4, 2, 4]
    a_bits_list = [2, 2, 4, 4]
    
    # Track results
    results = {
        'success': [],
        'failed': []
    }
    
    total_models = len(architectures) * len(seeds) * len(w_bits_list)
    current_model = 0
    
    print(f"Starting batch processing of {total_models} model configurations...")
    
    for arch in architectures:
        for seed in seeds:
            for i, (w_bits, a_bits) in enumerate(zip(w_bits_list, a_bits_list)):
                current_model += 1
                
                print(f"\nProgress: {current_model}/{total_models} ({(current_model/total_models)*100:.1f}%)")
                
                success = run_model_and_save_logits(
                    arch, w_bits, a_bits, seed, data_path, additional_args
                )
                
                if success:
                    results['success'].append({
                        'arch': arch,
                        'w_bits': w_bits,
                        'a_bits': a_bits,
                        'seed': seed
                    })
                else:
                    results['failed'].append({
                        'arch': arch,
                        'w_bits': w_bits,
                        'a_bits': a_bits,
                        'seed': seed
                    })
                
                # Small delay between runs
                time.sleep(1)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total models processed: {total_models}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['success']:
        print(f"\nSuccessful configurations:")
        for config in results['success']:
            print(f"  ✓ {config['arch']} - W{config['w_bits']}bit A{config['a_bits']}bit - Seed {config['seed']}")
    
    if results['failed']:
        print(f"\nFailed configurations:")
        for config in results['failed']:
            print(f"  ✗ {config['arch']} - W{config['w_bits']}bit A{config['a_bits']}bit - Seed {config['seed']}")
    
    return results

def main():
    """
    Main function for batch processing.
    """
    
    parser = argparse.ArgumentParser(description="Batch process models to save logits")
    parser.add_argument("--architectures", nargs="+", 
                       default=["resnet18", "resnet50", "mobilenetv2", "regnetx_600m", "regnetx_3200m", "mnasnet"],
                       help="List of architectures to process")
    parser.add_argument("--data_path", required=True,
                       help="Path to ImageNet dataset")
    parser.add_argument("--seeds", nargs="+", type=int,
                       default=[1001, 1002, 1003, 1004, 1005],
                       help="List of seeds to use")
    parser.add_argument("--additional_args", default="",
                       help="Additional arguments to pass to main_imagenet.py")
    parser.add_argument("--single_model", action="store_true",
                       help="Process only one model configuration for testing")
    
    args = parser.parse_args()
    
    print("Batch Logits Saving Tool")
    print("========================")
    print(f"Architectures: {args.architectures}")
    print(f"Data path: {args.data_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Additional args: {args.additional_args}")
    
    if args.single_model:
        print("\nSingle model mode - processing only first configuration...")
        arch = args.architectures[0]
        seed = args.seeds[0]
        w_bits, a_bits = 4, 4
        
        success = run_model_and_save_logits(arch, w_bits, a_bits, seed, args.data_path, args.additional_args)
        
        if success:
            print(f"\n✓ Single model test completed successfully!")
        else:
            print(f"\n✗ Single model test failed!")
        
        return
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist!")
        return
    
    # Start batch processing
    print(f"\nStarting batch processing...")
    results = batch_process_models(
        args.architectures, 
        args.data_path, 
        args.seeds, 
        args.additional_args
    )
    
    # Save results summary
    import json
    summary_file = f"batch_processing_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults summary saved to: {summary_file}")
    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()

