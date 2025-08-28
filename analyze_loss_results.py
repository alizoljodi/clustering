#!/usr/bin/env python3
"""
Helper script to analyze and summarize results from loss function ablation testing.
This script helps organize and compare results from different loss configurations.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def find_results_directories():
    """Find all results directories from loss function ablation testing."""
    # Look for directories matching the pattern from main_imagenet.py
    pattern = "results_alpha0.60_clusters64_pca50_*"
    results_dirs = glob.glob(pattern)
    
    # Also look for initial logits directories
    initial_pattern = "initial_logits_*"
    initial_dirs = glob.glob(initial_pattern)
    
    return results_dirs, initial_dirs

def extract_config_from_dirname(dirname):
    """Extract configuration information from directory name."""
    try:
        # Parse directory name like: results_alpha0.60_clusters64_pca50_resnet18_w4bit_a4bit
        parts = dirname.split('_')
        
        # Find architecture
        arch_idx = parts.index('clusters64') + 2
        arch = parts[arch_idx]
        
        # Find bit configuration
        w_bits = None
        a_bits = None
        for part in parts:
            if part.startswith('w') and part.endswith('bit'):
                w_bits = int(part[1:-3])
            elif part.startswith('a') and part.endswith('bit'):
                a_bits = int(part[1:-3])
        
        return {
            'architecture': arch,
            'weight_bits': w_bits,
            'activation_bits': a_bits,
            'alpha': 0.60,
            'clusters': 64,
            'pca_dim': 50
        }
    except:
        return None

def analyze_loss_configuration_results():
    """Analyze results from different loss function configurations."""
    results_dirs, initial_dirs = find_results_directories()
    
    print("LOSS FUNCTION ABLATION RESULTS ANALYSIS")
    print("="*60)
    
    if not results_dirs:
        print("No results directories found. Make sure you've run the ablation tests first.")
        return
    
    print(f"Found {len(results_dirs)} results directories")
    print(f"Found {len(initial_dirs)} initial logits directories")
    
    # Group results by architecture and bit configuration
    results_by_config = defaultdict(list)
    
    for dir_path in results_dirs:
        config = extract_config_from_dirname(dir_path)
        if config:
            key = f"{config['architecture']}_W{config['weight_bits']}A{config['activation_bits']}"
            results_by_config[key].append({
                'dir_path': dir_path,
                'config': config
            })
    
    print(f"\nResults grouped by configuration:")
    for key, configs in results_by_config.items():
        print(f"  {key}: {len(configs)} experiments")
    
    # Analyze each configuration
    for key, configs in results_by_config.items():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {key}")
        print(f"{'='*50}")
        
        # Look for experiment parameters CSV files
        for config in configs:
            dir_path = config['dir_path']
            params_file = os.path.join(dir_path, "experiment_parameters.csv")
            
            if os.path.exists(params_file):
                try:
                    params_df = pd.read_csv(params_file)
                    print(f"  ✓ Found parameters: {dir_path}")
                    
                    # Look for logits files
                    logits_files = glob.glob(os.path.join(dir_path, "logits_*.csv"))
                    if logits_files:
                        print(f"    - Logits files: {len(logits_files)} found")
                    else:
                        print(f"    - No logits files found")
                        
                except Exception as e:
                    print(f"  ✗ Error reading {params_file}: {e}")
            else:
                print(f"  - No parameters file found in {dir_path}")

def create_results_summary():
    """Create a summary of all ablation test results."""
    print("\n" + "="*60)
    print("CREATING RESULTS SUMMARY")
    print("="*60)
    
    # Create a summary directory
    summary_dir = "loss_ablation_summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    # Find all results
    results_dirs, initial_dirs = find_results_directories()
    
    summary_data = []
    
    # Process results directories
    for dir_path in results_dirs:
        config = extract_config_from_dirname(dir_path)
        if config:
            # Check for experiment parameters
            params_file = os.path.join(dir_path, "experiment_parameters.csv")
            if os.path.exists(params_file):
                try:
                    params_df = pd.read_csv(params_file)
                    # Add to summary
                    summary_data.append({
                        'directory': dir_path,
                        'architecture': config['architecture'],
                        'weight_bits': config['weight_bits'],
                        'activation_bits': config['activation_bits'],
                        'alpha': config['alpha'],
                        'clusters': config['clusters'],
                        'pca_dim': config['pca_dim'],
                        'has_parameters': True,
                        'has_logits': len(glob.glob(os.path.join(dir_path, "logits_*.csv"))) > 0,
                        'has_plots': len(glob.glob(os.path.join(dir_path, "*.png"))) > 0
                    })
                except:
                    pass
    
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(summary_dir, "ablation_results_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Summary saved to: {summary_file}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total experiments: {len(summary_df)}")
        print(f"  Architectures tested: {summary_df['architecture'].nunique()}")
        print(f"  Bit configurations: {len(summary_df[['weight_bits', 'activation_bits']].drop_duplicates())}")
        
        # Group by architecture
        arch_summary = summary_df.groupby('architecture').agg({
            'weight_bits': 'count',
            'has_logits': 'sum',
            'has_plots': 'sum'
        }).rename(columns={'weight_bits': 'total_experiments'})
        
        print(f"\nResults by Architecture:")
        print(arch_summary)
        
    else:
        print("No results data found to summarize.")

def suggest_next_steps():
    """Suggest next steps for analysis."""
    print("\n" + "="*60)
    print("NEXT STEPS FOR ANALYSIS")
    print("="*60)
    
    print("1. Check individual experiment results:")
    print("   - Look for 'experiment_parameters.csv' files in each results directory")
    print("   - Verify logits data was saved correctly")
    print("   - Check for generated plots and visualizations")
    
    print("\n2. Compare loss configurations:")
    print("   - Full loss vs. individual components")
    print("   - Performance impact of each loss term")
    print("   - Convergence behavior differences")
    
    print("\n3. Analyze specific architectures:")
    print("   - Which architectures benefit most from each loss component")
    print("   - Bit-width sensitivity to loss function changes")
    print("   - Optimal loss combinations for different model types")
    
    print("\n4. Generate comparative plots:")
    print("   - Accuracy comparison across loss configurations")
    print("   - Loss convergence curves")
    print("   - Parameter sensitivity analysis")

def main():
    """Main analysis function."""
    print("LOSS FUNCTION ABLATION ANALYSIS TOOL")
    print("="*60)
    
    # Analyze existing results
    analyze_loss_configuration_results()
    
    # Create summary
    create_results_summary()
    
    # Suggest next steps
    suggest_next_steps()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Use the generated summary files to analyze your ablation results.")
    print("Check individual experiment directories for detailed outputs.")

if __name__ == "__main__":
    main()
