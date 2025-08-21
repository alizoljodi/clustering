# Logits Saving Functionality

This document explains how to use the new logits saving functionality in `main_imagenet.py` to save all logits data as CSV files for all models.

## Overview

The enhanced `main_imagenet.py` script now automatically saves three types of logits data for each model configuration:

1. **Quantized logits** (`all_q_logits`) - Output from the quantized model
2. **Full-precision logits** (`all_fp_logits`) - Output from the full-precision model  
3. **Corrected logits** (`all_corrected_logits`) - Output after applying cluster-based correction

## What Gets Saved

### 1. Initial Logits (Before Clustering)
- **Location**: `initial_logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}/`
- **Files**: 
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_quantized.csv`
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_fullprecision.csv`
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_metadata.csv`

### 2. Corrected Logits (After Clustering)
- **Location**: `results_alpha{alpha}_clusters{clusters}_pca{pca}_{arch}_w{bits}bit_a{bits}bit/`
- **Files**:
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_quantized.csv`
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_fullprecision.csv`
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_corrected.csv`
  - `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_metadata.csv`

### 3. Summary Files
- **Location**: `logits_summary_{arch}_w{bits}bit_a{bits}bit_seed{seed}/`
- **Files**: `logits_summary_{arch}_w{bits}bit_a{bits}bit_seed{seed}.csv`

## Automatic Chunking

For large datasets (>1000 samples), the script automatically splits the data into chunks to avoid memory issues:

- **Chunk size**: 1000 samples (configurable)
- **File naming**: `logits_{arch}_w{bits}bit_a{bits}bit_seed{seed}_quantized_chunk001_of_003.csv`
- **Metadata**: Includes chunk information in `_chunked_metadata.csv`

## File Structure Example

```
clustering/
├── initial_logits_resnet18_w4bit_a4bit_seed1001/
│   ├── logits_resnet18_w4bit_a4bit_seed1001_quantized.csv
│   ├── logits_resnet18_w4bit_a4bit_seed1001_fullprecision.csv
│   └── logits_resnet18_w4bit_a4bit_seed1001_metadata.csv
├── results_alpha0.4_clusters16_pca50_resnet18_w4bit_a4bit/
│   ├── logits_resnet18_w4bit_a4bit_seed1001_quantized.csv
│   ├── logits_resnet18_w4bit_a4bit_seed1001_fullprecision.csv
│   ├── logits_resnet18_w4bit_a4bit_seed1001_corrected.csv
│   └── logits_resnet18_w4bit_a4bit_seed1001_metadata.csv
└── logits_summary_resnet18_w4bit_a4bit_seed1001/
    └── logits_summary_resnet18_w4bit_a4bit_seed1001.csv
```

## CSV File Format

### Logits Data
Each CSV file contains:
- **Rows**: Individual samples
- **Columns**: Class logits (1000 columns for ImageNet)
- **Header**: No header (pure numerical data)

### Metadata Files
Contains information about:
- Model architecture
- Bit precision (weight/activation)
- Seed value
- Number of samples
- Number of classes
- Timestamp
- Chunking information (if applicable)

## Usage

The functionality is automatically enabled when running `main_imagenet.py`. No additional parameters are needed.

### Running Individual Models
```bash
python main_imagenet.py --data_path /path/to/imagenet --arch resnet18 --seed 1001 --n_bits_w 4 --n_bits_a 4
```

### Running All Models via Script
```bash
python run_script.py resnet18
python run_script.py resnet50
python run_script.py mobilenetv2
# etc.
```

## Testing

To test the functionality without running the full training:

```bash
python test_logits_saving.py
```

This will create dummy logits data and test all saving functions.

## Data Analysis

The saved CSV files can be used for:

1. **Comparative Analysis**: Compare quantized vs. full-precision vs. corrected logits
2. **Clustering Analysis**: Analyze the effectiveness of different clustering parameters
3. **Error Analysis**: Study quantization errors and correction improvements
4. **Statistical Analysis**: Compute distributions, correlations, and other metrics
5. **Visualization**: Create plots and charts for presentations

## Memory Considerations

- **Small datasets** (<1000 samples): Saved as single CSV files
- **Large datasets** (≥1000 samples): Automatically chunked to avoid memory issues
- **Chunk size**: Configurable via `chunk_size` parameter (default: 1000)

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write permissions in the output directory
2. **Disk Space**: Large datasets can generate many CSV files
3. **Memory Issues**: If chunking doesn't help, reduce `chunk_size`

### Error Messages

- Check the console output for detailed error messages
- All errors are logged with full tracebacks
- Failed saves don't stop the main execution

## Performance Impact

- **Minimal overhead**: CSV saving adds <1% to total execution time
- **Parallel processing**: CSV writing happens in the main thread
- **I/O optimization**: Files are written sequentially to avoid disk contention

## Future Enhancements

Potential improvements:
- Compression options (gzip, bzip2)
- Alternative formats (HDF5, Parquet)
- Database integration
- Cloud storage support
- Real-time streaming

