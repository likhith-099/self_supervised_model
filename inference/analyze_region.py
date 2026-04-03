"""
Regional analysis using extracted features
Performs clustering, visualization, and pattern analysis on regional data
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class RegionalAnalyzer:
    """Analyze regional patterns from extracted features"""
    
    def __init__(self, features: np.ndarray, image_paths: List[str] = None):
        """
        Initialize analyzer
        
        Args:
            features: Feature array (N, D) where N=samples, D=dimensions
            image_paths: Optional paths to corresponding images
        """
        self.features = features
        self.image_paths = image_paths
        self.n_samples, self.n_dims = features.shape
        
        print(f"Loaded features: {features.shape}")
    
    def cluster_kmeans(
        self,
        n_clusters: int = 5,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Apply K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed
        
        Returns:
            Cluster labels
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.features)
        
        # Print cluster sizes
        print(f"\nK-Means clustering ({n_clusters} clusters):")
        for i in range(n_clusters):
            count = (labels == i).sum()
            pct = count / len(labels) * 100
            print(f"  Cluster {i}: {count} samples ({pct:.1f}%)")
        
        return labels
    
    def cluster_dbscan(
        self,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        Apply DBSCAN clustering
        
        Args:
            eps: Maximum distance between points
            min_samples: Minimum samples per cluster
        
        Returns:
            Cluster labels (-1 for noise)
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        print(f"\nDBSCAN clustering:")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        return labels
    
    def reduce_dimensions_pca(
        self,
        n_components: int = 2,
        explained_variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, float]:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            n_components: Target dimensions
            explained_variance_threshold: Variance to preserve
        
        Returns:
            Tuple of (reduced features, explained variance ratio)
        """
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.features)
        
        total_variance = pca.explained_variance_ratio_.sum()
        
        print(f"\nPCA reduction:")
        print(f"  Original dims: {self.n_dims}")
        print(f"  Reduced dims: {n_components}")
        print(f"  Explained variance: {total_variance:.2%}")
        
        return reduced, total_variance
    
    def reduce_dimensions_tsne(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000
    ) -> np.ndarray:
        """
        Apply t-SNE for visualization
        
        Args:
            n_components: Target dimensions
            perplexity: t-SNE perplexity
            n_iter: Number of iterations
        
        Returns:
            Reduced features
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        reduced = tsne.fit_transform(self.features)
        
        print(f"\nt-SNE reduction completed")
        return reduced
    
    def visualize_clusters(
        self,
        reduced_features: np.ndarray,
        labels: np.ndarray,
        save_path: str = None,
        title: str = "Feature Clusters"
    ):
        """
        Visualize clusters in 2D
        
        Args:
            reduced_features: 2D reduced features
            labels: Cluster labels
            save_path: Path to save plot
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = 'gray' if label == -1 else colors[i]
            marker = 'x' if label == -1 else 'o'
            
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                c=[color],
                marker=marker,
                label=f'Cluster {label}' if label != -1 else 'Noise',
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def compute_cluster_statistics(
        self,
        labels: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics for each cluster
        
        Args:
            labels: Cluster labels
        
        Returns:
            Dictionary of cluster statistics
        """
        stats = {}
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_features = self.features[mask]
            
            stats[cluster_id] = {
                'n_samples': mask.sum(),
                'percentage': mask.sum() / len(labels) * 100,
                'mean_feature_norm': np.mean(np.linalg.norm(cluster_features, axis=1)),
                'std_feature_norm': np.std(np.linalg.norm(cluster_features, axis=1)),
                'centroid': cluster_features.mean(axis=0)
            }
        
        return stats
    
    def find_similar_images(
        self,
        query_idx: int,
        top_k: int = 10
    ) -> Tuple[List[int], np.ndarray]:
        """
        Find most similar images to a query
        
        Args:
            query_idx: Index of query image
            top_k: Number of similar images to return
        
        Returns:
            Tuple of (indices, distances)
        """
        query_feature = self.features[query_idx:query_idx+1]
        
        # Compute distances
        distances = np.linalg.norm(self.features - query_feature, axis=1)
        
        # Get top-k (excluding query itself)
        sorted_indices = np.argsort(distances)
        top_indices = sorted_indices[1:top_k+1]
        top_distances = distances[top_indices]
        
        return top_indices.tolist(), top_distances


def load_features(feature_file: str) -> Tuple[np.ndarray, List[str]]:
    """Load features from .npy file"""
    features = np.load(feature_file)
    
    # Try to load corresponding image paths if available
    image_paths_file = Path(feature_file).with_suffix('.txt')
    image_paths = None
    
    if image_paths_file.exists():
        with open(image_paths_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    
    return features, image_paths


def main():
    parser = argparse.ArgumentParser(description='Analyze regional features')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to features file (.npy)')
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of clusters for K-Means')
    parser.add_argument('--pca-dims', type=int, default=2,
                        help='PCA target dimensions')
    parser.add_argument('--tsne', action='store_true',
                        help='Apply t-SNE visualization')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load features
    features, image_paths = load_features(args.features)
    
    # Create analyzer
    analyzer = RegionalAnalyzer(features, image_paths)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PCA reduction
    pca_reduced, variance = analyzer.reduce_dimensions_pca(n_components=args.pca_dims)
    
    # K-Means clustering
    kmeans_labels = analyzer.cluster_kmeans(n_clusters=args.n_clusters)
    
    # Visualize PCA clusters
    analyzer.visualize_clusters(
        pca_reduced,
        kmeans_labels,
        save_path=str(output_dir / 'pca_kmeans_clusters.png'),
        title=f'PCA + K-Means ({args.n_clusters} clusters)'
    )
    
    # t-SNE if requested
    if args.tsne:
        tsne_reduced = analyzer.reduce_dimensions_tsne()
        analyzer.visualize_clusters(
            tsne_reduced,
            kmeans_labels,
            save_path=str(output_dir / 'tsne_clusters.png'),
            title=f't-SNE + K-Means ({args.n_clusters} clusters)'
        )
    
    # Compute and save cluster statistics
    stats = analyzer.compute_cluster_statistics(kmeans_labels)
    
    stats_file = output_dir / 'cluster_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("Cluster Statistics\n")
        f.write("=" * 50 + "\n\n")
        for cluster_id, cluster_stats in stats.items():
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Samples: {cluster_stats['n_samples']} ({cluster_stats['percentage']:.1f}%)\n")
            f.write(f"  Mean feature norm: {cluster_stats['mean_feature_norm']:.4f}\n")
            f.write(f"  Std feature norm: {cluster_stats['std_feature_norm']:.4f}\n\n")
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
