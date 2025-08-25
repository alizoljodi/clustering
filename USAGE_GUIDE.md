# Run Scripts Usage Guide

This guide explains how to use the different run scripts for testing clustering vs global tensors approaches.

## Available Scripts

### 1. `run_script.py` (Original)
- **Purpose**: Runs clustering experiments only
- **Usage**: Original clustering approach with all parameter combinations

### 2. `run_global_tensors_script.py` (New)
- **Purpose**: Runs global tensors experiments only
- **Usage**: Global alpha/beta tensors approach without clustering

### 3. `run_comprehensive_script.py` (New)
- **Purpose**: Comprehensive testing with choice of approach
- **Usage**: Choose between clustering, global tensors, or both

## Quick Start Examples

### Basic Global Tensors Testing
```bash
# Test ResNet-18 with global tensors approach
python run_global_tensors_script.py resnet18

# Test MobileNetV2 with global tensors approach
python run_global_tensors_script.py mobilenetv2
```

### Comprehensive Testing
```bash
# Test both approaches for ResNet-18
python run_comprehensive_script.py resnet18 --approach both

# Test only clustering for MobileNetV2
python run_comprehensive_script.py mobilenetv2 --approach clustering

# Test only global tensors for ResNet-50
python run_comprehensive_script.py resnet50 --approach global

# Quick comparison (minimal testing)
python run_comprehensive_script.py resnet18 --approach quick
```

### Customized Testing
```bash
# Test with custom seeds and alpha values
python run_comprehensive_script.py resnet18 \
    --approach both \
    --seeds 1001 1002 1003 \
    --alpha_list 0.1 0.3 0.5 0.7 0.9 \
    --cluster_list 16 32 \
    --pca_list 25 50

# Test with custom data path
python run_comprehensive_script.py resnet50 \
    --approach global \
    --data_path /path/to/your/imagenet
```

## Script Comparison

| Feature | `run_script.py` | `run_global_tensors_script.py` | `run_comprehensive_script.py` |
|---------|----------------|--------------------------------|--------------------------------|
| **Clustering** | ✅ Yes | ❌ No | ✅ Yes (configurable) |
| **Global Tensors** | ❌ No | ✅ Yes | ✅ Yes (configurable) |
| **Approach Choice** | ❌ No | ❌ No | ✅ Yes |
| **Custom Parameters** | ❌ No | ❌ No | ✅ Yes |
| **Progress Tracking** | ❌ No | ❌ No | ✅ Yes |
| **Error Handling** | ❌ No | ❌ No | ✅ Yes |
| **Quick Comparison** | ❌ No | ❌ No | ✅ Yes |

## Parameter Configurations

### Quantization Bits
All scripts use the same quantization configurations:
- **Weight bits**: [2, 4, 2, 4]
- **Activation bits**: [2, 2, 4, 4]

### Model-Specific Parameters
Each model has optimized hyperparameters:

| Model | Weight | T | Lambda_c |
|-------|--------|---|----------|
| **ResNet-18/50** | 0.01 | 4.0 | 0.02 |
| **RegNetX-600M/3200M** | 0.01 | 4.0 | 0.01 |
| **MobileNetV2** | 0.1 | 1.0 | 0.005 |
| **MNASNet** | 0.2 | 1.0 | 0.001 |

### Default Testing Parameters
- **Alpha values**: [0.2, 0.4, 0.6]
- **Clusters**: [8, 16, 64]
- **PCA dimensions**: [25, 50, 100]
- **Seeds**: [1001, 1002, 1003]

## Output and Results

### What Each Script Produces
1. **Logits extraction** from both quantized and full-precision models
2. **Model building** (clustering or global tensors)
3. **Accuracy evaluation** on test set
4. **Results summary** with best configurations
5. **CSV files** with detailed results

### Result Files
- `logits_summary_*.csv`: Summary of all experiments
- `initial_logits_*`: Raw extracted logits
- Various visualization and analysis files

## Performance Considerations

### Global Tensors Approach
- **Faster**: No clustering computation
- **Less memory**: Single parameter set
- **Simpler**: Fewer hyperparameters to tune

### Clustering Approach
- **More flexible**: Per-cluster optimization
- **Potentially better accuracy**: Fine-grained control
- **More parameters**: Requires tuning clusters and PCA

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `--num_samples` in main script
2. **CUDA errors**: Check GPU memory and batch size
3. **Data path errors**: Verify ImageNet dataset location

### Debug Mode
Use the comprehensive script with `--approach quick` for fast debugging:
```bash
python run_comprehensive_script.py resnet18 --approach quick
```

### Monitoring Progress
The comprehensive script provides:
- Real-time progress updates
- Success/failure tracking
- Timing information
- Output previews

## Advanced Usage

### Batch Processing
```bash
# Run multiple models in sequence
for model in resnet18 resnet50 mobilenetv2; do
    python run_comprehensive_script.py $model --approach both
done
```

### Custom Testing Scenarios
```bash
# Test extreme alpha values
python run_comprehensive_script.py resnet18 \
    --approach global \
    --alpha_list 0.0 0.1 0.9 1.0

# Test minimal clustering
python run_comprehensive_script.py resnet18 \
    --approach clustering \
    --cluster_list 2 4 \
    --pca_list 10
```

### Integration with Existing Workflows
The scripts can be easily integrated into existing automation:
```bash
# Example: Run after model training
python train_model.py
python run_comprehensive_script.py resnet18 --approach both
python analyze_results.py
```

## Best Practices

1. **Start with quick comparison** to understand performance differences
2. **Use appropriate approach** based on your needs:
   - Speed: Global tensors
   - Accuracy: Clustering
   - Research: Both approaches
3. **Monitor resources** during long runs
4. **Save results** for later analysis
5. **Use multiple seeds** for robust evaluation
