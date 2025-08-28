# Loss Function Ablation Guide

This guide explains how to use the ablation functionality to control different components of the loss function during quantization.

## Overview

The quantization process uses a composite loss function with three main components:

1. **Reconstruction Loss (rec_loss)**: Ensures quantized model output matches full-precision output
2. **Rounding Loss (round_loss)**: Regularizes weight quantization policy
3. **Prediction Difference Loss (pd_loss)**: Aligns final predictions between models

## Command Line Arguments

### Enable/Disable Loss Components

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_rec_loss` | Enable reconstruction loss | True |
| `--use_round_loss` | Enable rounding loss | True |
| `--use_pd_loss` | Enable prediction difference loss | True |
| `--no_rec_loss` | Disable reconstruction loss | False |
| `--no_round_loss` | Disable rounding loss | False |
| `--no_pd_loss` | Disable prediction difference loss | False |

### Usage Examples

#### 1. Full Loss (All Components)
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4
```
This uses the complete loss function: `rec_loss + round_loss + pd_loss`

#### 2. Ablate Reconstruction Loss Only
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_rec_loss
```
This uses: `round_loss + pd_loss`

#### 3. Ablate Rounding Loss Only
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_round_loss
```
This uses: `rec_loss + pd_loss`

#### 4. Ablate Prediction Difference Loss Only
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_pd_loss
```
This uses: `rec_loss + round_loss`

#### 5. Use Only Reconstruction Loss
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_round_loss --no_pd_loss
```
This uses: `rec_loss` only

#### 6. Use Only Rounding Loss
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_rec_loss --no_pd_loss
```
This uses: `round_loss` only

#### 7. Use Only Prediction Difference Loss
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_rec_loss --no_round_loss
```
This uses: `pd_loss` only

## Implementation Details

### Files Modified

1. **`main_imagenet.py`**: Added command line arguments and passes ablation parameters
2. **`quant/block_recon.py`**: Updated `LossFunction` class with ablation flags
3. **`quant/layer_recon.py`**: Updated `LossFunction` class with ablation flags

### How It Works

1. **Command Line Parsing**: Arguments are parsed in `main_imagenet.py`
2. **Parameter Processing**: Ablation flags are processed and passed to reconstruction functions
3. **Loss Computation**: Each loss component is computed only if enabled
4. **Logging**: The ablation configuration is printed during execution

### Code Structure

```python
# In main_imagenet.py
use_rec_loss = args.use_rec_loss and not args.no_rec_loss
use_round_loss = args.use_round_loss and not args.no_round_loss
use_pd_loss = args.use_pd_loss and not args.no_pd_loss

kwargs = dict(
    # ... other parameters ...
    use_rec_loss=use_rec_loss,
    use_round_loss=use_round_loss,
    use_pd_loss=use_pd_loss
)
```

```python
# In LossFunction.__call__()
if self.use_rec_loss:
    rec_loss = lp_loss(pred, tgt, p=self.p)
else:
    rec_loss = torch.tensor(0.0, device=pred.device)

if self.use_pd_loss:
    pd_loss = self.pd_loss(...)
else:
    pd_loss = torch.tensor(0.0, device=pred.device)

if self.use_round_loss:
    round_loss = ...
else:
    round_loss = torch.tensor(0.0, device=pred.device)

total_loss = rec_loss + round_loss + pd_loss
```

## Use Cases

### Research Applications
- **Ablation Studies**: Understand the contribution of each loss component
- **Component Analysis**: Evaluate individual loss terms in isolation
- **Hyperparameter Tuning**: Find optimal loss combinations

### Debugging
- **Performance Issues**: Identify which loss component causes problems
- **Convergence Problems**: Test different loss configurations
- **Memory Optimization**: Reduce computational overhead by disabling components

### Custom Loss Functions
- **Novel Approaches**: Combine existing components with custom terms
- **Domain-Specific**: Adapt loss functions for specific applications
- **Experimental**: Test new loss function designs

## Monitoring and Logging

During execution, the system provides detailed logging:

```
Loss function ablation configuration:
  - Reconstruction Loss: ✓
  - Rounding Loss: ✗
  - Prediction Difference Loss: ✓

Total loss:	0.456 (rec:0.234, pd:0.222, round:0.000)	b=0.00	count=500
Ablation: rec_loss=True, round_loss=False, pd_loss=True
```

## Best Practices

1. **Start with Full Loss**: Always begin with all components enabled
2. **Gradual Ablation**: Disable one component at a time to understand its impact
3. **Performance Monitoring**: Track accuracy and convergence with different configurations
4. **Documentation**: Record which configurations work best for your use case
5. **Validation**: Test ablation results on validation data

## Troubleshooting

### Common Issues

1. **No Loss Components Enabled**: Ensure at least one loss component is active
2. **Convergence Problems**: Some loss combinations may not converge well
3. **Memory Issues**: Disabling components can affect memory usage patterns

### Debugging Tips

1. **Check Logs**: Verify ablation configuration is correct
2. **Monitor Loss Values**: Ensure individual components are computed correctly
3. **Validate Results**: Compare performance with full loss configuration

## Examples for Different Architectures

### ResNet
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --no_rec_loss
```

### MobileNet
```bash
python main_imagenet.py --arch mobilenetv2 --n_bits_w 4 --n_bits_a 4 --no_round_loss
```

### RegNet
```bash
python main_imagenet.py --arch regnetx_600m --n_bits_w 4 --n_bits_a 4 --no_pd_loss
```

## Conclusion

The ablation functionality provides a powerful tool for understanding and optimizing the quantization process. By selectively enabling or disabling loss components, you can:

- Analyze the contribution of each loss term
- Optimize performance for specific use cases
- Debug quantization issues
- Develop novel loss function combinations

Use this functionality systematically to improve your quantization results and gain deeper insights into the quantization process.
