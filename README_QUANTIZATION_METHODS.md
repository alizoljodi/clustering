# Quantization Methods Framework

This document describes the restructured quantization framework that supports multiple quantization methods including LAPQ, ACIQ-Mix, Adaround, and QDrop.

## Overview

The framework has been restructured to provide a modular, extensible architecture for different quantization methods. Each quantization method is implemented as a separate class that inherits from a common base class, making it easy to add new methods or modify existing ones.

## Architecture

### Core Classes

#### 1. `BaseQuantizationMethod` (Abstract Base Class)
- **Purpose**: Defines the interface that all quantization methods must implement
- **Key Methods**:
  - `quantize_model()`: Applies quantization to a model
  - `get_quantization_params()`: Returns quantization parameters
  - `get_method_name()`: Returns the name of the quantization method

#### 2. `QuantizationManager`
- **Purpose**: Factory class that manages different quantization methods
- **Key Methods**:
  - `get_quantization_method()`: Returns an instance of a specific quantization method
  - `list_available_methods()`: Lists all available quantization methods

#### 3. `ClusterAffineCorrection`
- **Purpose**: Handles cluster-based affine correction for quantized models
- **Key Methods**:
  - `build_correction_model()`: Builds the correction model from logits
  - `apply_correction()`: Applies correction to new logits

#### 4. `LogitsAnalyzer`
- **Purpose**: Handles saving and analyzing logits data
- **Key Methods**:
  - `save_logits_to_csv()`: Saves logits data to CSV files
  - `_save_logits_in_chunks()`: Handles large datasets with chunking

## Available Quantization Methods

### 1. Adaround
- **Class**: `AdaroundQuantization`
- **Description**: Adaptive rounding quantization method
- **Key Parameters**:
  - `iters_w`: Number of iterations for weight calibration
  - `weight`: Weight of rounding cost vs reconstruction loss
  - `b_start`, `b_end`: Temperature parameters
  - `warmup`: Warmup period for regularization
  - `lr`: Learning rate for LSQ
  - `lamb_r`, `T`: Regularization parameters
  - `bn_lr`, `lamb_c`: Batch normalization parameters

### 2. LAPQ (Layer-wise Asymmetric Quantization)
- **Class**: `LAPQQuantization`
- **Description**: Layer-wise asymmetric quantization method
- **Key Parameters**:
  - `num_bits`: Bitwidth for quantization
  - `channel_wise`: Whether to apply channel-wise quantization
  - `asymmetric`: Whether to use asymmetric quantization
  - `per_channel`: Whether to apply per-channel quantization

### 3. ACIQ-Mix
- **Class**: `ACIQMixQuantization`
- **Description**: Adaptive clipping with quantization mixing
- **Key Parameters**:
  - `num_bits`: Bitwidth for quantization
  - `aciq_method`: ACIQ method type
  - `mix_ratio`: Mixing ratio for different quantization strategies

### 4. QDrop
- **Class**: `QDropQuantization`
- **Description**: Quantization with dropout
- **Key Parameters**:
  - `num_bits`: Bitwidth for quantization
  - `dropout_rate`: Dropout rate during quantization
  - `stochastic`: Whether to use stochastic quantization

## Usage Examples

### Basic Usage

```python
from main_imagenet import QuantizationManager

# Get quantization manager
quant_manager = QuantizationManager()

# Get a specific quantization method
adaround = quant_manager.get_quantization_method('adaround', iters_w=20000)

# Apply quantization
quantized_model = adaround.quantize_model(model, fp_model, train_loader)
```

### Command Line Usage

```bash
# Use Adaround quantization
python main_imagenet.py --quant_method adaround --arch resnet18 --n_bits_w 4 --n_bits_a 4

# Use LAPQ quantization
python main_imagenet.py --quant_method lapq --arch resnet18 --n_bits_w 4 --n_bits_a 4

# Use ACIQ-Mix quantization
python main_imagenet.py --quant_method aciq_mix --arch resnet18 --n_bits_w 4 --n_bits_a 4

# Use QDrop quantization
python main_imagenet.py --quant_method qdrop --arch resnet18 --n_bits_w 4 --n_bits_a 4
```

### Advanced Usage with Multiple Parameters

```bash
# Test multiple alpha values, cluster numbers, and PCA dimensions
python main_imagenet.py \
    --quant_method adaround \
    --arch resnet18 \
    --alpha_list 0.2 0.4 0.6 \
    --num_clusters_list 32 64 128 \
    --pca_dim_list 30 50 100
```

## Adding New Quantization Methods

To add a new quantization method:

1. **Create a new class** that inherits from `BaseQuantizationMethod`:

```python
class NewQuantizationMethod(BaseQuantizationMethod):
    def __init__(self, **kwargs):
        super().__init__("NewMethod", **kwargs)
        self.default_params = {
            'param1': 'value1',
            'param2': 'value2'
        }
        self.default_params.update(kwargs)
    
    def quantize_model(self, model, fp_model, train_loader, **kwargs):
        # Implement your quantization logic here
        print(f"Applying {self.name} quantization...")
        
        # Your quantization implementation
        # ...
        
        return model
    
    def get_quantization_params(self):
        return self.default_params
```

2. **Register the method** in `QuantizationManager`:

```python
class QuantizationManager:
    def __init__(self):
        self.quantization_methods = {
            'adaround': AdaroundQuantization,
            'lapq': LAPQQuantization,
            'aciq_mix': ACIQMixQuantization,
            'qdrop': QDropQuantization,
            'new_method': NewQuantizationMethod  # Add this line
        }
```

## Configuration Parameters

### General Parameters
- `--seed`: Random seed for reproducibility
- `--arch`: Model architecture (resnet18, resnet50, mobilenetv2, etc.)
- `--batch_size`: Batch size for data loading
- `--workers`: Number of workers for data loading
- `--data_path`: Path to ImageNet data

### Quantization Method Selection
- `--quant_method`: Choose quantization method (adaround, lapq, aciq_mix, qdrop)

### Quantization Parameters
- `--n_bits_w`: Weight quantization bitwidth
- `--n_bits_a`: Activation quantization bitwidth
- `--channel_wise`: Whether to use channel-wise quantization
- `--disable_8bit_head_stem`: Disable 8-bit quantization for first/last layers

### Cluster Affine Correction Parameters
- `--alpha`: Alpha blending parameter for correction
- `--num_clusters`: Number of clusters for correction
- `--pca_dim`: PCA dimension for clustering
- `--alpha_list`, `--num_clusters_list`, `--pca_dim_list`: Multiple values to test

## Output Structure

The framework generates organized output directories:

```
results_alpha{alpha}_clusters{num_clusters}_pca{pca_dim}_{arch}_w{n_bits_w}bit_a{n_bits_a}bit/
├── logits_{arch}_w{n_bits_w}bit_a{n_bits_a}bit_seed{seed}_quantized.csv
├── logits_{arch}_w{n_bits_w}bit_a{n_bits_a}bit_seed{seed}_fullprecision.csv
├── logits_{arch}_w{n_bits_w}bit_a{n_bits_a}bit_seed{seed}_corrected.csv
└── logits_{arch}_w{n_bits_w}bit_a{n_bits_a}bit_seed{seed}_metadata.csv
```

## Example Scripts

### `example_quantization_usage.py`
A demonstration script showing how to use the framework programmatically.

### Running the Example
```bash
python example_quantization_usage.py
```

## Performance Comparison

The framework automatically compares different quantization methods and parameter combinations, providing:

- Top-1 and Top-5 accuracy for each configuration
- Summary of all results
- Best performing configuration identification
- Logits analysis and visualization

## Dependencies

- PyTorch
- scikit-learn (for clustering and PCA)
- matplotlib (for visualization)
- pandas (for data handling)
- numpy

## Notes

- **LAPQ, ACIQ-Mix, and QDrop implementations are placeholders** and need to be implemented with the actual algorithms
- The framework maintains backward compatibility with existing Adaround implementation
- All quantization methods can be used with the cluster affine correction system
- The modular design makes it easy to extend and modify individual components

## Future Enhancements

1. **Implement actual LAPQ algorithm**
2. **Implement actual ACIQ-Mix algorithm**
3. **Implement actual QDrop algorithm**
4. **Add more quantization methods**
5. **Enhanced visualization and analysis tools**
6. **Performance benchmarking suite**
7. **Automated hyperparameter optimization**

## Troubleshooting

### Common Issues

1. **Method not found**: Ensure the quantization method name is spelled correctly
2. **Memory issues**: Reduce batch size or number of samples
3. **CUDA errors**: Check GPU memory and ensure models are on the correct device

### Debug Mode

For debugging, you can modify the quantization method classes to add more verbose output and error handling.

## Contributing

When contributing new quantization methods:

1. Follow the existing class structure
2. Add comprehensive error handling
3. Include parameter validation
4. Add unit tests if possible
5. Update this documentation

## License

This framework is part of the clustering project and follows the same license terms.


