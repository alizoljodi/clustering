# Loss Function Ablation Testing Guide

This guide explains how to use the loss function ablation testing scripts to systematically evaluate the contribution of different loss components in the quantization process.

## Overview

The loss function ablation testing allows you to systematically test different combinations of the three main loss components:

1. **Reconstruction Loss (rec_loss)**: Ensures quantized model output matches full-precision output
2. **Rounding Loss (round_loss)**: Regularizes weight quantization policy  
3. **Prediction Difference Loss (pd_loss)**: Aligns final predictions between models

## Scripts Overview

### 1. `run_script_loss.py` - Main Testing Script
This script mimics `run_script.py` but focuses on testing different loss function configurations.

### 2. `analyze_loss_results.py` - Results Analysis Script
This script helps analyze and summarize the results from the ablation testing.

## Usage

### Running Loss Function Ablation Tests

```bash
# Test ResNet18 with all loss configurations
python run_script_loss.py resnet18

# Test MobileNetV2 with all loss configurations  
python run_script_loss.py mobilenetv2

# Test RegNet with all loss configurations
python run_script_loss.py regnetx_600m
```

### What Gets Tested

The script automatically tests **7 different loss configurations** for each architecture:

| Config | Name | Loss Components | Description |
|--------|------|-----------------|-------------|
| 1 | `full_loss` | rec_loss + round_loss + pd_loss | Complete loss function |
| 2 | `no_rec_loss` | round_loss + pd_loss | Ablate reconstruction loss |
| 3 | `no_round_loss` | rec_loss + pd_loss | Ablate rounding loss |
| 4 | `no_pd_loss` | rec_loss + round_loss | Ablate prediction difference loss |
| 5 | `rec_loss_only` | rec_loss | Only reconstruction loss |
| 6 | `round_loss_only` | round_loss | Only rounding loss |
| 7 | `pd_loss_only` | pd_loss | Only prediction difference loss |

### Fixed Parameters

The script uses fixed parameters for consistent testing:
- **Alpha**: 0.6 (blending parameter)
- **Number of clusters**: 64
- **PCA dimensions**: 50
- **Seeds**: 5 different random seeds (1001-1005)
- **Bit configurations**: 4 different weight/activation bit combinations

### Test Matrix

For each architecture, the script runs:
- **5 seeds** × **4 bit configurations** × **7 loss configurations** = **140 total experiments**

## Expected Output Structure

### Directory Naming Convention

Results are saved in directories following this pattern:
```
results_alpha0.60_clusters64_pca50_{architecture}_w{weight_bits}bit_a{activation_bits}bit/
```

Example:
```
results_alpha0.60_clusters64_pca50_resnet18_w4bit_a4bit/
```

### Files Generated

Each experiment directory contains:
- `experiment_parameters.csv` - All experiment parameters
- `cluster_comparison.png` - Cluster comparison plots
- `combined_logits.png` - Combined logits visualization
- `*_histogram.png` - Distribution histograms
- `cluster_visualization_*.png` - Cluster visualizations
- `logits_*.csv` - Raw logits data
- `README.md` - Experiment description

## Analyzing Results

### 1. Run the Analysis Script

```bash
python analyze_loss_results.py
```

This script will:
- Find all results directories
- Group results by architecture and configuration
- Create a summary CSV file
- Provide statistics and analysis

### 2. Check Individual Results

Look for these key files in each results directory:
- **`experiment_parameters.csv`** - Verify experiment settings
- **`logits_*.csv`** - Check if data was saved correctly
- **`*.png`** - Review generated visualizations

### 3. Compare Loss Configurations

The analysis will help you understand:
- Which loss components are most important for each architecture
- How different bit-widths respond to loss function changes
- Optimal loss combinations for different model types

## Example Test Run

```bash
# Start testing ResNet18
python run_script_loss.py resnet18

# Output will show:
# ============================================================
# Testing loss function ablation configurations for resnet18
# Fixed parameters: alpha=[0.6], clusters=[64], pca_dim=[50]
# Testing 7 different loss configurations
# Using 5 seeds for robustness
# ============================================================
# 
# ============================================================
# Testing: Full Loss (rec_loss + round_loss + pd_loss)
# Architecture: resnet18, Seed: 1001, Config: 1/4
# Bits: W2/A2
# ============================================================
# Executing: python main_imagenet.py --data_path /home/alz07xz/imagenet --arch resnet18 --seed 1001 --n_bits_w 2 --n_bits_a 2 --weight 0.01 --T 4.0 --lamb_c 0.02 --alpha_list 0.6 --num_clusters_list 64 --pca_dim_list 50
# ✓ Completed: full_loss for resnet18 seed 1001 config 1
```

## Monitoring Progress

The script provides detailed progress information:
- **Current configuration being tested**
- **Command being executed**
- **Completion status** (✓ or ✗)
- **Error messages** if something fails

## Expected Duration

**Total experiments per architecture**: 140
**Estimated time per experiment**: 10-30 minutes (depending on hardware)
**Total estimated time**: 23-70 hours per architecture

**Recommendation**: Start with a smaller architecture like ResNet18 to test the setup.

## Troubleshooting

### Common Issues

1. **No results directories found**
   - Make sure you've run the ablation tests first
   - Check that `main_imagenet.py` completed successfully

2. **Missing logits files**
   - Verify the experiment completed without errors
   - Check disk space and permissions

3. **Script stops unexpectedly**
   - Check for memory issues
   - Verify all dependencies are installed
   - Check the error logs

### Debugging Tips

1. **Start small**: Test with just one architecture first
2. **Monitor logs**: Check the output for error messages
3. **Verify paths**: Ensure data paths are correct for your system
4. **Check resources**: Monitor GPU memory and disk space

## Customization

### Modifying Test Parameters

To change the fixed parameters, edit `run_script_loss.py`:

```python
# Change these values as needed
alpha_list = [0.6]           # Change to [0.4, 0.6, 0.8] for multiple values
num_clusters_list = [64]     # Change to [32, 64, 128] for multiple values
pca_dim_list = [50]         # Change to [25, 50, 100] for multiple values
```

### Adding New Loss Configurations

To test additional loss combinations, add to the `ablation_configs` list:

```python
# Config 8: Custom combination
{
    'name': 'custom_config',
    'args': ['--no_rec_loss', '--weight', '0.05'],  # Custom arguments
    'description': 'Custom Loss Configuration'
}
```

## Best Practices

1. **Start with validation**: Run a small test first to verify everything works
2. **Monitor resources**: Keep track of GPU memory and disk usage
3. **Document results**: Note any unexpected behavior or errors
4. **Backup data**: Save important results before running long experiments
5. **Use screen/tmux**: For long-running experiments on remote servers

## Expected Insights

The ablation testing should reveal:

1. **Component Importance**: Which loss terms contribute most to performance
2. **Architecture Sensitivity**: How different models respond to loss changes
3. **Bit-width Effects**: How quantization precision affects loss function choice
4. **Optimal Combinations**: Best loss configurations for each model type
5. **Convergence Behavior**: How different loss combinations affect training stability

## Next Steps

After completing the ablation tests:

1. **Analyze results** using `analyze_loss_results.py`
2. **Compare configurations** to identify optimal combinations
3. **Generate comparative plots** for publication/presentation
4. **Document findings** for future reference
5. **Optimize parameters** based on discovered insights

## Support

If you encounter issues:

1. Check the error messages in the script output
2. Verify all dependencies are correctly installed
3. Ensure sufficient disk space and GPU memory
4. Check that data paths are correct for your system
5. Review the main script logs for detailed error information

---

**Note**: This testing framework provides a systematic way to understand the contribution of each loss component. Use the results to optimize your quantization process and improve model performance.
