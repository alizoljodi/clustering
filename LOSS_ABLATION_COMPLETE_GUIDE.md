# Complete Loss Function Ablation System Guide

This guide provides a comprehensive overview of the loss function ablation system that allows you to systematically test different combinations of loss components in neural network quantization.

## 🎯 Overview

The system enables you to control and test the involvement of three main loss components:

1. **Reconstruction Loss (rec_loss)**: Ensures quantized model output matches full-precision output
2. **Rounding Loss (round_loss)**: Regularizes weight quantization policy  
3. **Prediction Difference Loss (pd_loss)**: Aligns final predictions between models

## 🏗️ System Architecture

### Core Components

- **`main_imagenet.py`**: Main script with ablation command-line arguments
- **`run_script_loss.py`**: Testing script that runs different loss configurations
- **`job_loss.sh`**: SLURM job script for cluster execution
- **`analyze_loss_results.py`**: Results analysis and visualization
- **`quant/block_recon.py`**: Block reconstruction with ablation support
- **`quant/layer_recon.py`**: Layer reconstruction with ablation support

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Test a single architecture with all loss configurations
python run_script_loss.py resnet18

# Test with specific architecture
python run_script_loss.py resnet50
```

### 2. Command Line Control

```bash
# Enable/disable specific loss components
python main_imagenet.py --arch resnet18 --use_rec_loss --use_round_loss --no_pd_loss

# Full control over all components
python main_imagenet.py --arch resnet18 \
    --use_rec_loss \
    --use_round_loss \
    --use_pd_loss
```

## 📊 Loss Configurations Tested

The system automatically tests 7 different loss configurations:

| Configuration | rec_loss | round_loss | pd_loss | Description |
|---------------|-----------|------------|---------|-------------|
| `full_loss` | ✓ | ✓ | ✓ | All components enabled |
| `no_rec_loss` | ✗ | ✓ | ✓ | Without reconstruction loss |
| `no_round_loss` | ✓ | ✗ | ✓ | Without rounding loss |
| `no_pd_loss` | ✓ | ✓ | ✗ | Without prediction difference loss |
| `rec_loss_only` | ✓ | ✗ | ✗ | Only reconstruction loss |
| `round_loss_only` | ✗ | ✓ | ✗ | Only rounding loss |
| `pd_loss_only` | ✗ | ✗ | ✓ | Only prediction difference loss |

## 🔧 Implementation Details

### Loss Function Ablation

The `LossFunction` class in both `block_recon.py` and `layer_recon.py` has been enhanced with:

```python
class LossFunction:
    def __init__(self, 
                 # ... existing parameters ...
                 use_rec_loss: bool = True,
                 use_round_loss: bool = True,
                 use_pd_loss: bool = True):
        
        # Ablation flags
        self.use_rec_loss = use_rec_loss
        self.use_round_loss = use_round_loss
        self.use_pd_loss = use_pd_loss
    
    def __call__(self, pred, tgt, output, output_fp):
        # Calculate individual loss components
        if self.use_rec_loss:
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            rec_loss = 0.0
            
        if self.use_round_loss:
            round_loss = self.weight * (1 - ((2 * self.temp_decay(self.count) - 1) ** 2))
        else:
            round_loss = 0.0
            
        if self.use_pd_loss:
            pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), 
                                 F.softmax(output_fp / self.T, dim=1)) / self.lam
        else:
            pd_loss = 0.0
        
        # Combine based on ablation flags
        total_loss = rec_loss + round_loss + pd_loss
        return total_loss
```

### Command Line Arguments

Added to `main_imagenet.py`:

```python
# Loss function ablation arguments
parser.add_argument('--use_rec_loss', action='store_true', help='Enable reconstruction loss')
parser.add_argument('--use_round_loss', action='store_true', help='Enable rounding loss')
parser.add_argument('--use_pd_loss', action='store_true', help='Enable prediction difference loss')
parser.add_argument('--no_rec_loss', action='store_true', help='Disable reconstruction loss')
parser.add_argument('--no_round_loss', action='store_true', help='Disable rounding loss')
parser.add_argument('--no_pd_loss', action='store_true', help='Disable prediction difference loss')
```

## 📈 Testing Framework

### Test Parameters

- **Fixed Parameters**: `alpha=0.6`, `clusters=64`, `pca_dim=50`
- **Bit Configurations**: W2/A2, W4/A2, W2/A4, W4/A4
- **Seeds**: 5 different random seeds (1001-1005)
- **Total Experiments**: 140 per architecture (7 loss configs × 4 bit configs × 5 seeds)

### Execution Flow

1. **Setup**: Initialize environment and parameters
2. **Testing Loop**: For each seed, bit config, and loss config:
   - Build command with appropriate ablation flags
   - Execute `main_imagenet.py`
   - Parse accuracy from output
   - Store results
3. **Analysis**: Generate comprehensive results summary
4. **Output**: Save results to JSON and display summary

## 📊 Results Analysis

### Accuracy Parsing

The system automatically parses accuracy from various output formats:

- **Standard**: `Top-1 Accuracy: 76.54%`
- **Decimal**: `Accuracy: 0.8234`
- **Percentage**: `78.9%`
- **Contextual**: `Accuracy=0.8765`

### Results Summary

```
RESNET18 RESULTS:
----------------

  W2/A2:
    full_loss           :  45.23% (min: 44.89%, max: 45.67%) [5/5 seeds]
    no_rec_loss         :  42.15% (min: 41.78%, max: 42.52%) [5/5 seeds]
    no_round_loss       :  43.89% (min: 43.45%, max: 44.23%) [5/5 seeds]
    ...

  W4/A2:
    full_loss           :  67.89% (min: 67.45%, max: 68.23%) [5/5 seeds]
    ...
```

## 🚀 Cluster Execution

### SLURM Job Script

```bash
# Submit job for specific architecture
sbatch job_loss.sh resnet18

# Submit multiple jobs
sbatch job_loss.sh resnet18
sbatch job_loss.sh resnet50
sbatch job_loss.sh mobilenetv2
```

### Job Features

- **Automatic Logging**: Timestamped log directories
- **Progress Tracking**: Real-time progress updates
- **Results Analysis**: Automatic post-processing
- **Summary Reports**: Comprehensive job summaries
- **Error Handling**: Robust error handling and cleanup

## 📁 Output Structure

```
loss_ablation_logs_{JOB_ID}_{TIMESTAMP}/
├── job_progress.log          # Main progress log
├── {architecture}_test.log   # Architecture-specific test log
├── analysis.log              # Results analysis log
└── job_summary_{TIMESTAMP}.txt # Final summary report

results_alpha0.60_clusters64_pca50_*
├── experiment_parameters.csv
├── cluster_comparison.png
├── combined_logits.png
└── ...

loss_ablation_results.json    # Complete results in JSON format
```

## 🔍 Analysis and Visualization

### Automatic Analysis

The `analyze_loss_results.py` script provides:

- **Results Grouping**: By architecture, bit configuration, and loss configuration
- **Statistical Analysis**: Mean, min, max accuracy across seeds
- **Visualization**: Charts and plots for easy comparison
- **Export**: Results in various formats (CSV, JSON, PNG)

### Key Metrics

- **Accuracy Comparison**: Across different loss configurations
- **Seed Robustness**: Variation across different random seeds
- **Bit Configuration Impact**: How quantization affects different loss combinations
- **Statistical Significance**: Confidence intervals and significance testing

## 🛠️ Customization

### Adding New Loss Components

1. **Update LossFunction Class**:
   ```python
   def __init__(self, ..., use_new_loss: bool = True):
       self.use_new_loss = use_new_loss
   
   def __call__(self, ...):
       if self.use_new_loss:
           new_loss = calculate_new_loss(...)
       else:
           new_loss = 0.0
   ```

2. **Add Command Line Arguments**:
   ```python
   parser.add_argument('--use_new_loss', action='store_true')
   parser.add_argument('--no_new_loss', action='store_true')
   ```

3. **Update Testing Script**:
   ```python
   loss_configs.append({
       'name': 'new_loss_only',
       'use_rec_loss': False,
       'use_round_loss': False,
       'use_pd_loss': False,
       'use_new_loss': True
   })
   ```

### Modifying Test Parameters

- **Change Bit Configurations**: Modify `w_bits` and `a_bits` arrays
- **Adjust Seeds**: Modify the `seeds` list
- **Add Architectures**: Update the architecture choices in argument parser

## 📋 Best Practices

### 1. Testing Strategy

- **Start Small**: Begin with ResNet18 to validate the setup
- **Monitor Resources**: Track GPU memory and disk usage
- **Use Multiple Seeds**: Ensure results are robust across randomness
- **Incremental Testing**: Test one loss configuration at a time initially

### 2. Result Interpretation

- **Compare Baselines**: Always compare against full loss configuration
- **Statistical Significance**: Consider confidence intervals across seeds
- **Context Matters**: Understand what each loss component contributes
- **Look for Patterns**: Identify consistent trends across bit configurations

### 3. Performance Optimization

- **Parallel Execution**: Use multiple GPUs if available
- **Resource Management**: Monitor and adjust SLURM parameters
- **Caching**: Save intermediate results to avoid recomputation
- **Cleanup**: Regularly clean up temporary files and logs

## 🚨 Troubleshooting

### Common Issues

1. **Accuracy Not Parsed**:
   - Check output format in `main_imagenet.py`
   - Verify accuracy parsing patterns in `parse_accuracy_from_output()`
   - Test with `test_accuracy_parsing.py`

2. **Job Failures**:
   - Check SLURM logs: `loss_ablation_{JOB_ID}.err`
   - Verify virtual environment and dependencies
   - Check GPU availability and memory

3. **Results Inconsistency**:
   - Verify random seed handling
   - Check for deterministic operations
   - Ensure proper cleanup between experiments

### Debugging Tips

- **Local Testing**: Test scripts locally before submitting to cluster
- **Verbose Logging**: Enable detailed logging in main scripts
- **Step-by-Step**: Run individual experiments to isolate issues
- **Resource Monitoring**: Use `nvidia-smi` and `htop` for monitoring

## 📚 Example Workflows

### Workflow 1: Quick Validation

```bash
# Test single architecture with minimal configuration
python run_script_loss.py resnet18

# Check results
python analyze_loss_results.py
```

### Workflow 2: Comprehensive Analysis

```bash
# Submit cluster jobs for multiple architectures
sbatch job_loss.sh resnet18
sbatch job_loss.sh resnet50
sbatch job_loss.sh mobilenetv2

# Monitor progress
squeue -u $USER
tail -f loss_ablation_*.out

# Analyze all results
python analyze_loss_results.py
```

### Workflow 3: Custom Loss Configuration

```bash
# Test specific loss combination
python main_imagenet.py --arch resnet18 \
    --use_rec_loss \
    --no_round_loss \
    --use_pd_loss \
    --wbits 4 --abits 4
```

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Visualization**: Interactive dashboards and plots
2. **Statistical Testing**: Hypothesis testing and confidence intervals
3. **Automated Reporting**: PDF reports and presentations
4. **Integration**: Integration with other quantization frameworks
5. **Real-time Monitoring**: Live progress tracking and alerts

### Extension Points

- **New Loss Components**: Easy addition of custom loss functions
- **Multi-objective Optimization**: Pareto frontier analysis
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Cross-validation**: K-fold cross-validation support

## 📞 Support and Resources

### Documentation

- **This Guide**: Complete system overview
- **Code Comments**: Inline documentation in all scripts
- **README Files**: Specific usage instructions for each component

### Testing and Validation

- **Unit Tests**: `test_accuracy_parsing.py` for core functionality
- **Integration Tests**: End-to-end testing with sample data
- **Validation Scripts**: Automated validation of results

### Community and Updates

- **Version Control**: Track changes and improvements
- **Issue Tracking**: Report bugs and request features
- **Contributions**: Guidelines for contributing improvements

---

## 🎉 Conclusion

The loss function ablation system provides a comprehensive framework for understanding the contribution of different loss components in neural network quantization. By systematically testing different combinations, you can:

- **Understand Model Behavior**: See how each loss component affects quantization
- **Optimize Performance**: Find the best loss combination for your use case
- **Ensure Robustness**: Validate results across multiple seeds and configurations
- **Scale Experiments**: Run comprehensive tests on cluster infrastructure

The system is designed to be flexible, robust, and easy to use, enabling both quick experiments and comprehensive analysis. Whether you're doing research, development, or production optimization, this framework provides the tools you need to understand and improve your quantization process.
