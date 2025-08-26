# t-SNE Clustering for Quantized Model Logits

This project provides comprehensive t-SNE clustering analysis for quantized neural network logits, specifically designed for analyzing the behavior of quantized models on ImageNet data.

## Overview

The t-SNE clustering representation helps visualize and analyze how quantized model logits cluster together, providing insights into:
- Model behavior patterns
- Quantization effects on feature representations
- Cluster quality and separation
- Potential areas for model improvement

## Features

### 1. **t-SNE Dimensionality Reduction**
- Reduces high-dimensional logits (e.g., 1000 classes) to 2D for visualization
- Configurable perplexity parameter for optimal results
- Automatic perplexity adjustment based on data size

### 2. **K-Means Clustering**
- Pre-clusters logits before t-SNE for better structure
- Configurable number of clusters
- Uses multiple initializations for robust results

### 3. **Comprehensive Visualization**
- Interactive scatter plots with color-coded clusters
- High-resolution PNG export (300 DPI)
- Cluster statistics and percentages

### 4. **Quality Metrics**
- **Silhouette Score**: Measures cluster cohesion and separation
- **Cluster Center Distances**: Average separation between clusters
- **Adjusted Rand Index**: Comparison with ground truth (if available)

### 5. **Data Export**
- CSV files with t-SNE coordinates and cluster assignments
- Cluster center coordinates
- Sample indices for traceability

## Usage

### Main Script (`main_imagenet_tsne.py`)

The main script automatically creates t-SNE clustering when run:

```bash
python main_imagenet_tsne.py --arch mobilenetv2 --n_bits_w 4 --n_bits_a 4 --seed 42
```

### Test Script (`test_tsne_clustering.py`)

Test the functionality with synthetic data:

```bash
python test_tsne_clustering.py
```

## Function Details

### `create_tsne_clustering_representation()`

```python
def create_tsne_clustering_representation(
    logits,           # Input logits tensor [N, C]
    num_clusters=10,  # Number of K-means clusters
    perplexity=30,    # t-SNE perplexity
    random_state=42   # Reproducibility seed
):
    """
    Returns:
    - tsne_coords: 2D coordinates [N, 2]
    - cluster_ids: Cluster assignments [N]
    - kmeans_model: Fitted K-means model
    - tsne_model: Fitted t-SNE model
    """
```

### `visualize_tsne_clustering()`

```python
def visualize_tsne_clustering(
    tsne_coords,      # t-SNE coordinates
    cluster_ids,      # Cluster assignments
    save_path=None,   # Optional save path
    title="Title"     # Plot title
):
```

## Output Files

The script generates several output files in the `results/` directory:

### 1. **Visualization**
- `tsne_clustering_W4A4_seed42.png`: High-resolution scatter plot

### 2. **Data Files**
- `tsne_clustering_data_W4A4_seed42.csv`: t-SNE coordinates + cluster IDs
- `cluster_centers_W4A4_seed42.csv`: K-means cluster centers

### 3. **Test Results**
- `test_results/test_tsne_clustering.png`: Test visualization
- `test_results/test_tsne_clustering_data.csv`: Test data
- `test_results/test_cluster_centers.csv`: Test cluster centers

## Parameters

### t-SNE Parameters
- **perplexity**: Controls neighborhood size (default: 30)
- **n_iter**: Maximum iterations (default: 1000)
- **learning_rate**: Learning rate (default: 'auto')

### Clustering Parameters
- **num_clusters**: Number of K-means clusters (default: 10)
- **n_init**: K-means initializations (default: 10)
- **random_state**: Reproducibility seed (default: 42)

## Example Output

```
============================================================
CREATING T-SNE CLUSTERING REPRESENTATION
============================================================
Creating t-SNE clustering representation for 50000 samples...
Applying K-means clustering with 10 clusters...
Applying t-SNE dimensionality reduction...
t-SNE completed. Final shape: (50000, 2)

============================================================
CLUSTER QUALITY ANALYSIS
============================================================
Silhouette Score: 0.4523
(Higher values indicate better-defined clusters)
Average distance between cluster centers: 2.8476

============================================================
T-SNE CLUSTERING COMPLETED SUCCESSFULLY!
============================================================
```

## Interpretation

### Silhouette Score
- **0.7-1.0**: Strong clustering structure
- **0.5-0.7**: Reasonable clustering
- **0.25-0.5**: Weak clustering
- **<0.25**: Poor clustering

### Cluster Visualization
- **Well-separated clusters**: Good model behavior
- **Overlapping clusters**: Potential quantization issues
- **Outliers**: Anomalous samples or model behavior

## Dependencies

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `num_samples` or use smaller perplexity
2. **Poor Clustering**: Adjust `num_clusters` or `perplexity`
3. **Slow Performance**: Reduce `n_iter` or use smaller dataset

### Performance Tips

- Use GPU for large datasets
- Adjust perplexity based on data size
- Consider PCA preprocessing for very high-dimensional data

## Advanced Usage

### Custom Clustering

```python
# Use different clustering algorithm
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=5)
cluster_ids = clustering.fit_predict(logits_np)
```

### Multiple t-SNE Runs

```python
# Compare different perplexity values
perplexities = [10, 30, 50]
for p in perplexities:
    tsne = TSNE(perplexity=p)
    coords = tsne.fit_transform(logits_np)
    # Analyze and compare results
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tsne_clustering_quantization,
  title={t-SNE Clustering Analysis for Quantized Neural Networks},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
