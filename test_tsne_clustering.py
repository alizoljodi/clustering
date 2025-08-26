#!/usr/bin/env python3
"""
Test script for t-SNE clustering functionality.
This script demonstrates how to use the t-SNE clustering functions with sample data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import os

def create_tsne_clustering_representation(logits, num_clusters=10, perplexity=30, random_state=42):
    """
    Create t-SNE clustering representation from logits.
    
    Args:
        logits: Tensor of shape [N, C] where N is number of samples, C is number of classes
        num_clusters: Number of clusters for K-means clustering
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
        
    Returns:
        tsne_coords: t-SNE coordinates [N, 2]
        cluster_ids: Cluster assignments [N]
        kmeans_model: Fitted K-means model
        tsne_model: Fitted t-SNE model
    """
    print(f"Creating t-SNE clustering representation for {logits.shape[0]} samples...")
    
    # Convert to numpy if it's a tensor
    if torch.is_tensor(logits):
        logits_np = logits.numpy()
    else:
        logits_np = logits
        
    # Apply K-means clustering first
    print(f"Applying K-means clustering with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(logits_np)
    
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE dimensionality reduction...")
    # Adjust perplexity based on data size
    adjusted_perplexity = min(perplexity, len(logits_np) // 4)
    tsne = TSNE(n_components=2, random_state=random_state, 
                perplexity=adjusted_perplexity, n_iter=1000, learning_rate='auto')
    tsne_coords = tsne.fit_transform(logits_np)
    
    print(f"t-SNE completed. Final shape: {tsne_coords.shape}")
    
    return tsne_coords, cluster_ids, kmeans, tsne

def visualize_tsne_clustering(tsne_coords, cluster_ids, save_path=None, title="t-SNE Clustering of Quantized Logits"):
    """
    Visualize t-SNE clustering results.
    
    Args:
        tsne_coords: t-SNE coordinates [N, 2]
        cluster_ids: Cluster assignments [N]
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with different colors for each cluster
    unique_clusters = np.unique(cluster_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=20)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to: {save_path}")
    
    plt.show()
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    for cluster_id in unique_clusters:
        cluster_size = np.sum(cluster_ids == cluster_id)
        cluster_percentage = (cluster_size / len(cluster_ids)) * 100
        print(f"Cluster {cluster_id}: {cluster_size} samples ({cluster_percentage:.1f}%)")

def main():
    """Main function to test t-SNE clustering with sample data."""
    print("Testing t-SNE Clustering Functionality")
    print("=" * 50)
    
    # Create sample logits data (simulating quantized model output)
    np.random.seed(42)
    num_samples = 1000
    num_classes = 1000  # ImageNet has 1000 classes
    
    # Create synthetic logits with some structure (clusters)
    print(f"Generating {num_samples} synthetic logits with {num_classes} classes...")
    
    # Create 5 clusters with different characteristics
    cluster_centers = np.array([
        [1.0, 0.5, 0.3, 0.1, 0.05] + [0.01] * (num_classes - 5),  # Cluster 0: high values in first 5
        [0.1, 1.0, 0.5, 0.3, 0.1] + [0.01] * (num_classes - 5),  # Cluster 1: high values in middle
        [0.05, 0.1, 0.3, 1.0, 0.5] + [0.01] * (num_classes - 5), # Cluster 2: high values in middle-late
        [0.01, 0.05, 0.1, 0.3, 1.0] + [0.01] * (num_classes - 5), # Cluster 3: high values in last 5
        [0.5, 0.5, 0.5, 0.5, 0.5] + [0.01] * (num_classes - 5)   # Cluster 4: uniform distribution
    ])
    
    # Assign samples to clusters
    cluster_assignments = np.random.choice(5, num_samples, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    
    # Generate logits based on cluster assignments
    logits = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        cluster_id = cluster_assignments[i]
        base_logits = cluster_centers[cluster_id].copy()
        # Add some noise
        noise = np.random.normal(0, 0.1, num_classes)
        logits[i] = base_logits + noise
    
    print(f"Generated logits shape: {logits.shape}")
    
    # Convert to tensor (simulating the actual use case)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    
    # Create t-SNE clustering representation
    print("\n" + "="*60)
    print("CREATING T-SNE CLUSTERING REPRESENTATION")
    print("="*60)
    
    # Apply t-SNE clustering
    tsne_coords, cluster_ids, kmeans_model, tsne_model = create_tsne_clustering_representation(
        logits_tensor, 
        num_clusters=5,  # We know we created 5 clusters
        perplexity=30,   
        random_state=42
    )
    
    # Visualize the clustering
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    tsne_plot_path = os.path.join(results_dir, "test_tsne_clustering.png")
    visualize_tsne_clustering(tsne_coords, cluster_ids, save_path=tsne_plot_path, 
                             title="Test t-SNE Clustering of Synthetic Logits")
    
    # Save t-SNE data and cluster assignments
    tsne_data = {
        'tsne_component_1': tsne_coords[:, 0],
        'tsne_component_2': tsne_coords[:, 1],
        'cluster_id': cluster_ids,
        'sample_index': np.arange(len(tsne_coords)),
        'true_cluster': cluster_assignments  # For comparison
    }
    
    tsne_df = pd.DataFrame(tsne_data)
    tsne_csv_path = os.path.join(results_dir, "test_tsne_clustering_data.csv")
    tsne_df.to_csv(tsne_csv_path, index=False)
    print(f"\nt-SNE clustering data saved to: {tsne_csv_path}")
    
    # Additional analysis: Cluster quality metrics
    print("\n" + "="*60)
    print("CLUSTER QUALITY ANALYSIS")
    print("="*60)
    
    # Calculate silhouette score for cluster quality
    try:
        silhouette_avg = silhouette_score(tsne_coords, cluster_ids)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print("(Higher values indicate better-defined clusters)")
    except Exception as e:
        print(f"Could not calculate silhouette score: {e}")
    
    # Calculate cluster separation (average distance between cluster centers)
    cluster_centers_kmeans = kmeans_model.cluster_centers_
    if len(cluster_centers_kmeans) > 1:
        center_distances = euclidean_distances(cluster_centers_kmeans)
        avg_center_distance = np.mean(center_distances[center_distances > 0])
        print(f"Average distance between cluster centers: {avg_center_distance:.4f}")
    
    # Save cluster centers
    centers_df = pd.DataFrame(cluster_centers_kmeans, 
                            columns=[f'feature_{i}' for i in range(cluster_centers_kmeans.shape[1])])
    centers_df['cluster_id'] = range(len(cluster_centers_kmeans))
    centers_csv_path = os.path.join(results_dir, "test_cluster_centers.csv")
    centers_df.to_csv(centers_csv_path, index=False)
    print(f"Cluster centers saved to: {centers_csv_path}")
    
    # Compare with true cluster assignments
    print("\n" + "="*60)
    print("CLUSTER COMPARISON ANALYSIS")
    print("="*60)
    
    # Calculate accuracy of clustering compared to true assignments
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari_score = adjusted_rand_score(cluster_assignments, cluster_ids)
    nmi_score = normalized_mutual_info_score(cluster_assignments, cluster_ids)
    
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Normalized Mutual Information: {nmi_score:.4f}")
    print("(Higher values indicate better agreement with true clusters)")
    
    print("\n" + "="*60)
    print("TEST T-SNE CLUSTERING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
