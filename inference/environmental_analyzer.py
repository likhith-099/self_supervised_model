"""
Specialized environmental condition analyzer
Detects specific environmental changes: vegetation, water, urban, climate
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt


class EnvironmentalConditionAnalyzer:
    """Analyze specific environmental conditions from feature embeddings"""
    
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
    
    def detect_vegetation_change(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        threshold: float = 0.15
    ) -> Dict:
        """
        Detect vegetation loss or gain using NIR band characteristics
        
        Args:
            baseline_features: Features from reference period
            current_features: Features from current period
            threshold: Change detection threshold
        
        Returns:
            Dictionary with vegetation change metrics
        """
        # Vegetation has high NIR reflectance (B8 band)
        # MAE encoder should capture this in feature representations
        
        # Compute mean feature vectors
        baseline_mean = baseline_features.mean(axis=0).reshape(1, -1)
        current_mean = current_features.mean(axis=0).reshape(1, -1)
        
        # Cosine similarity (vegetation similarity)
        similarity = cosine_similarity(baseline_mean, current_mean)[0][0]
        
        # Euclidean distance (magnitude of change)
        distance = np.linalg.norm(
            baseline_features.mean(axis=0) - current_features.mean(axis=0)
        )
        
        # Cluster analysis to find vegetation-specific clusters
        all_features = np.vstack([baseline_features, current_features])
        kmeans = KMeans(n_clusters=8, random_state=42)
        labels = kmeans.fit_predict(all_features)
        
        baseline_labels = labels[:len(baseline_features)]
        current_labels = labels[len(baseline_features):]
        
        # Identify vegetation clusters (typically high NIR response)
        # Assuming cluster with highest mean represents dense vegetation
        cluster_means = [
            all_features[labels == i].mean() 
            for i in range(8)
        ]
        vegetation_cluster = np.argmax(cluster_means)
        
        baseline_veg_pct = (baseline_labels == vegetation_cluster).sum() / len(baseline_labels) * 100
        current_veg_pct = (current_labels == vegetation_cluster).sum() / len(current_labels) * 100
        veg_change = current_veg_pct - baseline_veg_pct
        
        result = {
            'condition': 'vegetation',
            'similarity': float(similarity),
            'distance': float(distance),
            'baseline_vegetation_coverage': float(baseline_veg_pct),
            'current_vegetation_coverage': float(current_veg_pct),
            'vegetation_change_percent': float(veg_change),
            'status': 'LOSS' if veg_change < -threshold else 'GAIN' if veg_change > threshold else 'STABLE',
            'severity': 'HIGH' if abs(veg_change) > 20 else 'MEDIUM' if abs(veg_change) > 10 else 'LOW'
        }
        
        return result
    
    def detect_water_expansion(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """
        Detect water body expansion or contraction
        
        Water has low reflectance in visible and NIR bands
        
        Args:
            baseline_features: Features from reference period
            current_features: Features from current period
            threshold: Change detection threshold
        
        Returns:
            Dictionary with water change metrics
        """
        # Water absorbs NIR, so low NIR response indicates water
        all_features = np.vstack([baseline_features, current_features])
        kmeans = KMeans(n_clusters=8, random_state=42)
        labels = kmeans.fit_predict(all_features)
        
        baseline_labels = labels[:len(baseline_features)]
        current_labels = labels[len(baseline_features):]
        
        # Identify water clusters (lowest reflectance)
        cluster_means = [
            all_features[labels == i].mean() 
            for i in range(8)
        ]
        water_cluster = np.argmin(cluster_means)
        
        baseline_water_pct = (baseline_labels == water_cluster).sum() / len(baseline_labels) * 100
        current_water_pct = (current_labels == water_cluster).sum() / len(current_labels) * 100
        water_change = current_water_pct - baseline_water_pct
        
        # Compute spatial distribution if coordinates available
        expansion_metric = water_change / baseline_water_pct if baseline_water_pct > 0 else 0
        
        result = {
            'condition': 'water',
            'baseline_water_coverage': float(baseline_water_pct),
            'current_water_coverage': float(current_water_pct),
            'water_change_percent': float(water_change),
            'expansion_ratio': float(expansion_metric * 100),
            'status': 'EXPANSION' if water_change > threshold else 'CONTRACTION' if water_change < -threshold else 'STABLE',
            'severity': 'HIGH' if abs(water_change) > 15 else 'MEDIUM' if abs(water_change) > 5 else 'LOW'
        }
        
        return result
    
    def detect_urban_growth(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """
        Detect urban/built-up area expansion
        
        Urban areas have high albedo and distinct spectral signature
        
        Args:
            baseline_features: Features from reference period
            current_features: Features from current period
            threshold: Change detection threshold
        
        Returns:
            Dictionary with urban growth metrics
        """
        all_features = np.vstack([baseline_features, current_features])
        kmeans = KMeans(n_clusters=8, random_state=42)
        labels = kmeans.fit_predict(all_features)
        
        baseline_labels = labels[:len(baseline_features)]
        current_labels = labels[len(baseline_features):]
        
        # Urban clusters typically have moderate-high reflectance across bands
        # and lower variance (homogeneous surfaces)
        cluster_stats = []
        for i in range(8):
            cluster_data = all_features[labels == i]
            if len(cluster_data) > 0:
                mean_val = cluster_data.mean()
                std_val = cluster_data.std()
                # Urban score: high mean, low std
                urban_score = mean_val / (std_val + 1e-6)
                cluster_stats.append((i, urban_score))
        
        urban_cluster = max(cluster_stats, key=lambda x: x[1])[0]
        
        baseline_urban_pct = (baseline_labels == urban_cluster).sum() / len(baseline_labels) * 100
        current_urban_pct = (current_labels == urban_cluster).sum() / len(current_labels) * 100
        urban_change = current_urban_pct - baseline_urban_pct
        
        result = {
            'condition': 'urban',
            'baseline_urban_coverage': float(baseline_urban_pct),
            'current_urban_coverage': float(current_urban_pct),
            'urban_change_percent': float(urban_change),
            'growth_rate': float(urban_change / baseline_urban_pct * 100) if baseline_urban_pct > 0 else 0,
            'status': 'EXPANDING' if urban_change > threshold else 'DECLINING' if urban_change < -threshold else 'STABLE',
            'severity': 'HIGH' if urban_change > 20 else 'MEDIUM' if urban_change > 5 else 'LOW'
        }
        
        return result
    
    def detect_land_degradation(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        threshold: float = 0.2
    ) -> Dict:
        """
        Detect land degradation (loss of productive capacity)
        
        Combines multiple indicators:
        - Decreased vegetation
        - Increased bare soil
        - Reduced feature diversity
        
        Args:
            baseline_features: Features from reference period
            current_features: Features from current period
            threshold: Change detection threshold
        
        Returns:
            Dictionary with degradation metrics
        """
        # 1. Check vegetation loss
        veg_result = self.detect_vegetation_change(baseline_features, current_features)
        veg_loss = veg_result['vegetation_change_percent']
        
        # 2. Check feature diversity (biodiversity proxy)
        baseline_variance = baseline_features.var(axis=0).mean()
        current_variance = current_features.var(axis=0).mean()
        diversity_change = (current_variance - baseline_variance) / baseline_variance * 100
        
        # 3. Overall feature shift magnitude
        baseline_mean = baseline_features.mean(axis=0).reshape(1, -1)
        current_mean = current_features.mean(axis=0).reshape(1, -1)
        feature_shift = np.linalg.norm(baseline_mean - current_mean)
        
        # 4. Cluster homogenization check
        all_features = np.vstack([baseline_features, current_features])
        kmeans = KMeans(n_clusters=8, random_state=42)
        labels = kmeans.fit_predict(all_features)
        
        baseline_labels = labels[:len(baseline_features)]
        current_labels = labels[len(baseline_features):]
        
        # Count dominant clusters (>10% coverage)
        baseline_unique, baseline_counts = np.unique(baseline_labels, return_counts=True)
        current_unique, current_counts = np.unique(current_labels, return_counts=True)
        
        baseline_dominant = (baseline_counts / len(baseline_labels) > 0.1).sum()
        current_dominant = (current_counts / len(current_labels) > 0.1).sum()
        homogenization = current_dominant - baseline_dominant
        
        # Degradation score (weighted combination)
        degradation_score = (
            -veg_loss * 0.4 +  # Vegetation loss contributes positively
            -diversity_change * 0.3 +  # Diversity loss contributes positively
            -homogenization * 0.3  # Homogenization contributes positively
        )
        
        result = {
            'condition': 'land_degradation',
            'vegetation_loss_percent': float(-veg_loss),
            'diversity_change_percent': float(diversity_change),
            'feature_shift_magnitude': float(feature_shift),
            'homogenization_index': float(homogenization),
            'degradation_score': float(degradation_score),
            'status': 'DEGRADING' if degradation_score > threshold else 'IMPROVING' if degradation_score < -threshold else 'STABLE',
            'severity': 'CRITICAL' if degradation_score > 30 else 'HIGH' if degradation_score > 15 else 'MODERATE' if degradation_score > 5 else 'LOW'
        }
        
        return result
    
    def detect_environmental_stress(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        threshold: float = 0.15
    ) -> Dict:
        """
        Detect overall environmental stress (combined indicators)
        
        Integrates:
        - Climate stress signals
        - Ecosystem health
        - Resilience indicators
        
        Args:
            baseline_features: Features from reference period
            current_features: Features from current period
            threshold: Stress detection threshold
        
        Returns:
            Dictionary with stress metrics
        """
        # Multiple indicator analysis
        
        # 1. Overall ecosystem shift
        similarity = cosine_similarity(
            baseline_features.mean(axis=0).reshape(1, -1),
            current_features.mean(axis=0).reshape(1, -1)
        )[0][0]
        
        ecosystem_shift = 1 - similarity
        
        # 2. Variability increase (stress indicator)
        baseline_cv = baseline_features.std(axis=0).mean() / (baseline_features.mean(axis=0).mean() + 1e-6)
        current_cv = current_features.std(axis=0).mean() / (current_features.mean(axis=0).mean() + 1e-6)
        variability_change = (current_cv - baseline_cv) / baseline_cv * 100
        
        # 3. Extreme value frequency (outliers)
        baseline_median = np.median(baseline_features, axis=0)
        baseline_mad = np.median(np.abs(baseline_features - baseline_median), axis=0)
        
        outliers_baseline = (np.abs(baseline_features - baseline_median) > 3 * baseline_mad).any(axis=1).mean()
        outliers_current = (np.abs(current_features - baseline_median) > 3 * baseline_mad).any(axis=1).mean()
        outlier_increase = outliers_current - outliers_baseline
        
        # 4. Resilience metric (recovery capacity proxy)
        # Higher feature diversity = higher resilience
        baseline_diversity = np.linalg.matrix_rank(baseline_features)
        current_diversity = np.linalg.matrix_rank(current_features)
        resilience_change = (current_diversity - baseline_diversity) / baseline_diversity
        
        # Stress index (composite)
        stress_index = (
            ecosystem_shift * 0.3 +
            variability_change * 0.25 +
            outlier_increase * 100 * 0.25 +
            -resilience_change * 0.2
        )
        
        result = {
            'condition': 'environmental_stress',
            'ecosystem_shift': float(ecosystem_shift),
            'variability_change_percent': float(variability_change),
            'outlier_frequency_change': float(outlier_increase),
            'resilience_change': float(resilience_change),
            'stress_index': float(stress_index),
            'status': 'HIGH_STRESS' if stress_index > threshold else 'MODERATE_STRESS' if stress_index > 0 else 'LOW_STRESS',
            'severity': 'CRITICAL' if stress_index > 50 else 'HIGH' if stress_index > 20 else 'MODERATE' if stress_index > 5 else 'LOW'
        }
        
        return result
    
    def generate_report(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        condition: str = 'all'
    ) -> str:
        """
        Generate comprehensive environmental report
        
        Args:
            baseline_features: Baseline period features
            current_features: Current period features
            condition: Specific condition or 'all'
        
        Returns:
            Formatted report string
        """
        print("=" * 80)
        print("ENVIRONMENTAL CONDITION ANALYSIS REPORT")
        print("=" * 80)
        
        conditions = {
            'vegetation': self.detect_vegetation_change,
            'water': self.detect_water_expansion,
            'urban': self.detect_urban_growth,
            'land_degradation': self.detect_land_degradation,
            'environmental_stress': self.detect_environmental_stress
        }
        
        if condition == 'all':
            for cond_name, cond_func in conditions.items():
                print(f"\n{cond_name.upper().replace('_', ' ')} ANALYSIS")
                print("-" * 80)
                result = cond_func(baseline_features, current_features)
                self._print_result(result)
        elif condition in conditions:
            result = conditions[condition](baseline_features, current_features)
            self._print_result(result)
        else:
            print(f"Unknown condition: {condition}")
            print(f"Available conditions: {list(conditions.keys())}")
        
        print("\n" + "=" * 80)
    
    def _print_result(self, result: Dict):
        """Print analysis result in formatted way"""
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    def visualize_comparison(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        save_path: str = None
    ):
        """
        Visualize feature space comparison
        
        Args:
            baseline_features: Baseline features
            current_features: Current features
            save_path: Path to save plot
        """
        # Combine features
        all_features = np.vstack([baseline_features, current_features])
        labels = ['Baseline'] * len(baseline_features) + ['Current'] * len(current_features)
        
        # PCA reduction
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_features)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA scatter
        ax1 = axes[0]
        baseline_reduced = reduced[:len(baseline_features)]
        current_reduced = reduced[len(baseline_features):]
        
        ax1.scatter(baseline_reduced[:, 0], baseline_reduced[:, 1], 
                   alpha=0.5, c='blue', label='Baseline', s=50)
        ax1.scatter(current_reduced[:, 0], current_reduced[:, 1], 
                   alpha=0.5, c='red', label='Current', s=50)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('Feature Space Comparison (PCA)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature distribution
        ax2 = axes[1]
        ax2.hist([baseline_features.flatten(), current_features.flatten()], 
                bins=50, alpha=0.7, label=['Baseline', 'Current'], color=['blue', 'red'])
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Feature Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()


def main():
    """Test environmental condition analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Environmental Condition Analyzer')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline features file (.npy)')
    parser.add_argument('--current', type=str, required=True,
                        help='Current features file (.npy)')
    parser.add_argument('--condition', type=str, default='all',
                        choices=['vegetation', 'water', 'urban', 'land_degradation', 
                                'environmental_stress', 'all'],
                        help='Condition to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Load features
    baseline = np.load(args.baseline)
    current = np.load(args.current)
    
    # Create analyzer
    analyzer = EnvironmentalConditionAnalyzer(baseline)
    
    # Generate report
    analyzer.generate_report(baseline, current, args.condition)
    
    # Visualize
    if args.output:
        analyzer.visualize_comparison(baseline, current, save_path=args.output)


if __name__ == '__main__':
    main()
