# Global Alpha/Beta Tensors for Logit Refinement

This document explains how to use the new global alpha/beta tensors approach for logit refinement, which provides an alternative to the clustering-based approach.

## Overview

The global tensors approach computes single alpha (gamma) and beta tensors for the entire dataset without performing clustering. This can be:
- **Faster**: No clustering computation required
- **Simpler**: Single set of parameters instead of per-cluster parameters
- **More interpretable**: Direct relationship between quantized and full-precision logits

## Usage

### Command Line Interface

To use the global tensors approach, add the `--use_global_tensors` flag:

```bash
python main_imagenet.py \
    --arch resnet18 \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 512 \
    --alpha 0.5 \
    --use_global_tensors \
    --seed 42
```

### Without the Flag (Default Clustering)

To use the original clustering approach, simply omit the flag:

```bash
python main_imagenet.py \
    --arch resnet18 \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 512 \
    --alpha 0.5 \
    --num_clusters 32 \
    --pca_dim 50 \
    --seed 42
```

## How It Works

### Global Tensors Approach

1. **Logit Extraction**: Extract logits from both quantized and full-precision models
2. **Global Computation**: Compute single gamma and beta tensors for the entire dataset:
   - `gamma = Cov(q, fp) / Var(q)` (per-class)
   - `beta = Mean(fp) - gamma * Mean(q)` (per-class)
3. **Application**: Apply correction: `corrected = q + alpha * (gamma * q + beta - q)`

### Clustering Approach (Original)

1. **Logit Extraction**: Extract logits from both models
2. **Clustering**: Perform K-means clustering on logits (with optional PCA)
3. **Per-cluster Parameters**: Compute gamma and beta for each cluster
4. **Application**: Apply cluster-specific correction with alpha blending

## Mathematical Details

### Global Affine Correction

For each class `c`:
- `gamma_c = E[(q_c - μ_q_c)(fp_c - μ_fp_c)] / E[(q_c - μ_q_c)²]`
- `beta_c = μ_fp_c - gamma_c * μ_q_c`

Where:
- `q_c` and `fp_c` are logits for class `c`
- `μ_q_c` and `μ_fp_c` are means for class `c`
- `E[·]` denotes expectation over samples

### Alpha Blending

The final corrected logits are computed as:
```
corrected = q + alpha * (affine_corrected - q)
```

Where `affine_corrected = gamma * q + beta`

## Example Script

Use the provided `example_global_tensors_usage.py` script to easily test both approaches:

```bash
python example_global_tensors_usage.py
```

This script allows you to:
1. Run with global tensors only
2. Run with clustering only  
3. Run both approaches for comparison

## When to Use Each Approach

### Use Global Tensors When:
- You want faster execution
- You prefer simpler, more interpretable models
- Your dataset has relatively uniform logit distributions
- You want to avoid clustering hyperparameters

### Use Clustering When:
- You need fine-grained control over different logit regions
- Your dataset has distinct clusters in logit space
- You want to capture complex, non-linear relationships
- You have time for clustering computation

## Performance Comparison

| Aspect | Global Tensors | Clustering |
|--------|----------------|------------|
| **Speed** | Fast | Slower (clustering overhead) |
| **Memory** | Low | Higher (cluster parameters) |
| **Flexibility** | Low | High |
| **Interpretability** | High | Medium |
| **Hyperparameters** | Few | Many |

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `--num_samples` if using large models
2. **Poor Results**: Try different `--alpha` values
3. **Slow Execution**: Use `--use_global_tensors` for faster processing

### Debugging

- Check that logits are being extracted correctly
- Verify alpha values are in reasonable range (0.0 to 1.0)
- Ensure sufficient calibration samples

## Advanced Usage

### Multiple Alpha Testing

Test multiple alpha values with global tensors:

```bash
python main_imagenet.py \
    --arch resnet18 \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --alpha_list 0.1 0.3 0.5 0.7 0.9 \
    --use_global_tensors
```

### Custom Implementation

You can also use the functions directly in your code:

```python
from main_imagenet import build_global_affine, apply_global_affine

# Build global model
gamma, beta = build_global_affine(all_q, all_fp)

# Apply correction
corrected, affine_corrected = apply_global_affine(q_logits, gamma, beta, alpha=0.5)
```

## References

- Original clustering approach: Based on K-means clustering with PCA
- Global tensors approach: Direct least-squares optimization
- Alpha blending: Linear interpolation between original and corrected logits
