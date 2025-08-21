#!/usr/bin/env python3
"""
Example script demonstrating how to use different quantization methods
from the restructured main_imagenet.py file.

This script shows how to:
1. Use different quantization methods (Adaround, LAPQ, ACIQ-Mix, QDrop)
2. Compare their performance
3. Apply cluster affine correction
"""

import torch
import torch.nn as nn
from main_imagenet import (
    QuantizationManager,
    ClusterAffineCorrection,
    LogitsAnalyzer,
    extract_model_logits,
    evaluate_cluster_affine_with_alpha,
    validate_model
)
from quant import QuantModel
import hubconf
import copy


def create_dummy_model():
    """Create a dummy model for demonstration purposes"""
    # Use a small ResNet18 for demonstration
    model = hubconf.resnet18(pretrained=False)
    model.eval()
    return model


def create_dummy_data():
    """Create dummy data for demonstration"""
    # Create dummy calibration data
    cali_data = torch.randn(100, 3, 224, 224)
    cali_target = torch.randint(0, 1000, (100,))
    
    # Create dummy test data
    test_data = torch.randn(50, 3, 224, 224)
    test_target = torch.randint(0, 1000, (50,))
    
    return cali_data, cali_target, test_data, test_target


def compare_quantization_methods():
    """Compare different quantization methods"""
    print("=" * 60)
    print("Quantization Methods Comparison")
    print("=" * 60)
    
    # Create models
    fp_model = create_dummy_model()
    q_model = copy.deepcopy(fp_model)
    
    # Build quantization parameters
    wq_params = {'n_bits': 4, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': 4, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True, 'prob': 0.5}
    
    # Create quantized models
    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    q_model = QuantModel(model=q_model, weight_quant_params=wq_params, act_quant_params=aq_params)
    
    # Get quantization manager
    quant_manager = QuantizationManager()
    print(f"Available quantization methods: {quant_manager.list_available_methods()}")
    
    # Test each quantization method
    results = {}
    
    for method_name in quant_manager.list_available_methods():
        print(f"\n--- Testing {method_name.upper()} ---")
        
        # Create fresh copy for each method
        test_model = copy.deepcopy(q_model)
        
        # Get quantization method instance
        quant_method = quant_manager.get_quantization_method(
            method_name,
            iters_w=1000,  # Reduced for demonstration
            weight=0.01,
            b_start=20,
            b_end=2,
            warmup=0.2,
            lr=4e-5,
            input_prob=0.5,
            lamb_r=0.1,
            T=4.0,
            bn_lr=1e-3,
            lamb_c=0.02,
            num_samples=100,
            keep_cpu=False
        )
        
        try:
            # Apply quantization
            quantized_model = quant_method.quantize_model(
                test_model, fp_model, 
                train_loader=None,  # We'll use dummy data
                num_samples=100
            )
            
            # Set quantization state
            quantized_model.set_quant_state(True, True)
            
            # Get quantization parameters
            params = quant_method.get_quantization_params()
            print(f"  Quantization parameters: {params}")
            
            results[method_name] = {
                'model': quantized_model,
                'params': params,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results[method_name] = {
                'model': None,
                'params': {},
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def demonstrate_cluster_affine_correction():
    """Demonstrate cluster affine correction"""
    print("\n" + "=" * 60)
    print("Cluster Affine Correction Demonstration")
    print("=" * 60)
    
    # Create dummy logits data
    num_samples = 1000
    num_classes = 1000
    
    # Simulate quantized and full-precision logits
    q_logits = torch.randn(num_samples, num_classes) * 0.1  # Lower variance for quantized
    fp_logits = torch.randn(num_samples, num_classes) * 0.5  # Higher variance for full-precision
    
    print(f"Created dummy logits: {q_logits.shape}")
    
    # Test different cluster configurations
    configs = [
        {'alpha': 0.2, 'num_clusters': 16, 'pca_dim': 20},
        {'alpha': 0.4, 'num_clusters': 32, 'pca_dim': 30},
        {'alpha': 0.6, 'num_clusters': 64, 'pca_dim': 50},
    ]
    
    for config in configs:
        print(f"\n--- Testing config: {config} ---")
        
        # Create cluster correction
        cluster_correction = ClusterAffineCorrection(
            alpha=config['alpha'],
            num_clusters=config['num_clusters'],
            pca_dim=config['pca_dim']
        )
        
        # Build correction model
        cluster_correction.build_correction_model(q_logits, fp_logits)
        
        # Apply correction
        corrected_logits = cluster_correction.apply_correction(q_logits, alpha=config['alpha'])
        
        print(f"  Original quantized logits std: {q_logits.std():.4f}")
        print(f"  Full-precision logits std: {fp_logits.std():.4f}")
        print(f"  Corrected logits std: {corrected_logits.std():.4f}")
        
        # Calculate improvement
        original_diff = torch.abs(q_logits - fp_logits).mean()
        corrected_diff = torch.abs(corrected_logits - fp_logits).mean()
        improvement = (original_diff - corrected_diff) / original_diff * 100
        
        print(f"  Improvement: {improvement:.2f}%")


def demonstrate_logits_analysis():
    """Demonstrate logits analysis functionality"""
    print("\n" + "=" * 60)
    print("Logits Analysis Demonstration")
    print("=" * 60)
    
    # Create dummy logits data
    num_samples = 500
    num_classes = 1000
    
    q_logits = [torch.randn(50, num_classes) * 0.1 for _ in range(10)]  # 10 batches of 50 samples
    fp_logits = [torch.randn(50, num_classes) * 0.5 for _ in range(10)]
    corrected_logits = [torch.randn(50, num_classes) * 0.3 for _ in range(10)]
    
    # Create logits analyzer
    results_dir = "demo_logits_analysis"
    analyzer = LogitsAnalyzer(results_dir)
    
    # Save logits to CSV
    success = analyzer.save_logits_to_csv(
        q_logits, fp_logits, corrected_logits,
        arch="resnet18", n_bit_w=4, n_bit_a=4, seed=42
    )
    
    if success:
        print(f"Logits analysis data saved to: {results_dir}")
    else:
        print("Failed to save logits analysis data")


def main():
    """Main demonstration function"""
    print("Quantization Methods Demonstration")
    print("This script demonstrates the restructured quantization framework")
    
    # Compare quantization methods
    results = compare_quantization_methods()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Quantization Methods Summary")
    print("=" * 60)
    for method_name, result in results.items():
        status = result['status']
        print(f"{method_name:12}: {status}")
        if status == 'failed':
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Demonstrate cluster affine correction
    demonstrate_cluster_affine_correction()
    
    # Demonstrate logits analysis
    demonstrate_logits_analysis()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nTo use with real data, run:")
    print("python main_imagenet.py --quant_method adaround --arch resnet18")
    print("python main_imagenet.py --quant_method lapq --arch resnet18")
    print("python main_imagenet.py --quant_method aciq_mix --arch resnet18")
    print("python main_imagenet.py --quant_method qdrop --arch resnet18")


if __name__ == "__main__":
    main()


