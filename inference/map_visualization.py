"""
Real Map Visualization and Future Prediction System
Generates:
1. Before/After satellite map overlays
2. Change detection heatmaps
3. Future trend predictions
4. Interactive visualizations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import cv2
from PIL import Image


class MapVisualizer:
    """Generate real map visualizations and predictions"""
    
    def __init__(self, baseline_features: np.ndarray, current_features: np.ndarray,
                 baseline_images: List[str] = None, current_images: List[str] = None):
        """
        Initialize visualizer
        
        Args:
            baseline_features: Feature vectors from baseline period
            current_features: Feature vectors from current period
            baseline_images: Paths to baseline satellite images
            current_images: Paths to current satellite images
        """
        self.baseline_features = baseline_features
        self.current_features = current_features
        self.baseline_images = baseline_images or []
        self.current_images = current_images or []
    
    def create_before_after_map(self, save_path: str = None, show: bool = True):
        """
        Create side-by-side before/after map comparison
        
        Args:
            save_path: Path to save figure
            show: Whether to display
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Load and display baseline image
        if self.baseline_images:
            img_baseline = np.array(Image.open(self.baseline_images[0]))
            axes[0].imshow(img_baseline)
            axes[0].set_title('BEFORE (Baseline Period)', fontsize=16, fontweight='bold')
            axes[0].axis('off')
        
        # Load and display current image
        if self.current_images:
            img_current = np.array(Image.open(self.current_images[0]))
            axes[1].imshow(img_current)
            axes[1].set_title('AFTER (Current Period)', fontsize=16, fontweight='bold')
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved before/after map to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_change_detection_heatmap(self, change_type: str = 'vegetation',
                                       save_path: str = None, show: bool = True):
        """
        Create heatmap showing areas of change
        
        Args:
            change_type: Type of change (vegetation, water, urban)
            save_path: Path to save figure
            show: Whether to display
        """
        # Compute change magnitude for each location
        n_samples = min(len(self.baseline_features), len(self.current_features))
        
        # Calculate per-sample change
        changes = []
        for i in range(n_samples):
            baseline_vec = self.baseline_features[i]
            current_vec = self.current_features[i]
            
            # Euclidean distance (magnitude of change)
            change_magnitude = np.linalg.norm(baseline_vec - current_vec)
            changes.append(change_magnitude)
        
        changes = np.array(changes)
        
        # Normalize to 0-1
        changes_norm = (changes - changes.min()) / (changes.max() - changes.min() + 1e-8)
        
        # Create spatial grid (assuming tiles are arranged in grid)
        grid_size = int(np.sqrt(n_samples))
        change_grid = changes_norm[:grid_size*grid_size].reshape(grid_size, grid_size)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Custom colormap (green → yellow → red)
        colors = [(0, 'green'), (0.5, 'yellow'), (1, 'red')]
        cmap = LinearSegmentedColormap.from_list('change_map', colors, N=100)
        
        # Plot heatmap
        im = ax.imshow(change_grid, cmap=cmap, alpha=0.7)
        
        # Add colorbar with interpretation
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Change Magnitude', fontsize=12)
        
        # Add interpretation ticks
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['No Change\n(Green)', 'Moderate Change\n(Yellow)', 
                            'Significant Change\n(Red)'])
        
        ax.set_title(f'{change_type.upper()} CHANGE DETECTION MAP\n'
                    f'Red = Major Changes | Green = Stable Areas',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add statistics
        stats_text = (f'Statistics:\n'
                     f'Mean Change: {changes.mean():.4f}\n'
                     f'Max Change: {changes.max():.4f}\n'
                     f'Changed Areas: {(changes_norm > 0.5).sum()} / {n_samples} '
                     f'({(changes_norm > 0.5).mean()*100:.1f}%)')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved change detection map to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_trend_prediction(self, years_data: Dict[int, np.ndarray],
                               predict_years: int = 10,
                               save_path: str = None, show: bool = True):
        """
        Predict future environmental trends
        
        Args:
            years_data: Dictionary {year: feature_array}
            predict_years: Number of years to predict into future
            save_path: Path to save figure
            show: Whether to display
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Extract years and mean feature values
        years = sorted(years_data.keys())
        means = [years_data[y].mean() for y in years]
        stds = [years_data[y].std() for y in years]
        
        # Polynomial regression for trend prediction
        X = np.array(years).reshape(-1, 1)
        y = np.array(means)
        
        # Fit polynomial (degree 2 for non-linear trends)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        reg = LinearRegression()
        reg.fit(X_poly, y)
        
        # Generate predictions
        future_years = list(range(years[-1], years[-1] + predict_years + 1))
        X_future = np.array(future_years).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        predictions = reg.predict(X_future_poly)
        
        # Confidence intervals (simplified)
        residuals = y - reg.predict(X_poly)
        std_error = np.std(residuals)
        confidence_upper = predictions + 1.96 * std_error
        confidence_lower = predictions - 1.96 * std_error
        
        # Plot 1: Trend line with predictions
        ax1 = axes[0, 0]
        ax1.plot(years, means, 'o-', linewidth=2, markersize=8, label='Historical Data',
                color='blue')
        ax1.plot(future_years, predictions, 's--', linewidth=2, markersize=6,
                label='Predicted Trend', color='red')
        ax1.fill_between(future_years, confidence_lower, confidence_upper,
                        alpha=0.3, color='red', label='95% Confidence Interval')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Environmental Indicator', fontsize=12)
        ax1.set_title('ENVIRONMENTAL TREND & FUTURE PREDICTION',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add annotation for prediction
        pred_change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
        trend_direction = "INCREASING" if pred_change > 0 else "DECREASING"
        annotation = f'Predicted Change:\n{abs(pred_change):.1f}% {trend_direction}'
        ax1.annotate(annotation, xy=(0.98, 0.95), xycoords='axes fraction',
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Plot 2: Rate of change
        ax2 = axes[0, 1]
        year_indices = np.arange(len(years))
        rates = np.diff(means)
        ax2.bar(year_indices[:-1] + 0.5, rates, color='coral', alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(year_indices[:-1] + 0.5)
        ax2.set_xticklabels([f'{years[i]}-{years[i+1]}' for i in range(len(years)-1)],
                           rotation=45, ha='right')
        ax2.set_ylabel('Annual Rate of Change', fontsize=12)
        ax2.set_title('RATE OF ENVIRONMENTAL CHANGE',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Feature distribution shift (PCA)
        from sklearn.decomposition import PCA
        all_features = np.vstack([years_data[y] for y in years])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_features)
        
        ax3 = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        start_idx = 0
        for i, year in enumerate(years):
            n_samples = len(years_data[year])
            end_idx = start_idx + n_samples
            ax3.scatter(reduced[start_idx:end_idx, 0],
                       reduced[start_idx:end_idx, 1],
                       alpha=0.6, label=f'{year}', s=50, color=[colors[i]])
            start_idx = end_idx
        
        ax3.set_xlabel('PC1', fontsize=12)
        ax3.set_ylabel('PC2', fontsize=12)
        ax3.set_title('FEATURE SPACE EVOLUTION\n(Clustering shows environmental states)',
                     fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Severity gauge
        ax4 = axes[1, 1]
        
        # Calculate severity score
        overall_change = abs(means[-1] - means[0]) / means[0] * 100
        rate_of_change = abs(rates[-1]) if len(rates) > 0 else 0
        
        severity_score = min(100, overall_change * 0.7 + rate_of_change * 30)
        
        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        radius = 1
        
        # Background arcs
        colors_gauge = ['green', 'yellow', 'orange', 'red']
        thresholds = [0, 25, 50, 75, 100]
        
        for i in range(len(colors_gauge)):
            theta_start = thresholds[i] / 100 * np.pi
            theta_end = thresholds[i+1] / 100 * np.pi
            theta_arc = np.linspace(theta_start, theta_end, 20)
            
            x = radius * np.cos(theta_arc)
            y = radius * np.sin(theta_arc)
            
            ax4.plot(x, y, color=colors_gauge[i], linewidth=20, alpha=0.3)
        
        # Needle
        needle_angle = severity_score / 100 * np.pi
        needle_x = [0, radius * 0.8 * np.cos(needle_angle)]
        needle_y = [0, radius * 0.8 * np.sin(needle_angle)]
        ax4.plot(needle_x, needle_y, 'k-', linewidth=3, marker='o', markersize=8)
        
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-0.2, 1.2)
        ax4.axis('off')
        ax4.set_title(f'OVERALL SEVERITY GAUGE\nScore: {severity_score:.1f}/100',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add severity labels
        ax4.text(-1.1, 0.1, 'LOW', fontsize=10, color='green', fontweight='bold')
        ax4.text(0, 1.1, 'CRITICAL', fontsize=10, color='red', fontweight='bold',
                ha='center')
        ax4.text(1.1, 0.1, 'HIGH', fontsize=10, color='orange', fontweight='bold',
                ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved prediction analysis to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_comprehensive_report(self, condition: str, region_name: str,
                                     years_data: Dict[int, np.ndarray],
                                     save_path: str = None):
        """
        Generate comprehensive visual report with all analyses
        
        Args:
            condition: Environmental condition analyzed
            region_name: Region name
            years_data: Multi-year feature data
            save_path: Path to save report
        """
        print("=" * 80)
        print("GENERATING COMPREHENSIVE VISUAL REPORT")
        print("=" * 80)
        
        # Create output directory
        if save_path:
            from pathlib import Path
            output_dir = Path(save_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Before/After comparison
        print("\n[1/4] Creating before/after comparison map...")
        self.create_before_after_map(
            save_path=str(save_path).replace('.png', '_before_after.png') if save_path else None,
            show=False
        )
        
        # 2. Change detection heatmap
        print("[2/4] Creating change detection heatmap...")
        self.create_change_detection_heatmap(
            change_type=condition,
            save_path=str(save_path).replace('.png', '_change_map.png') if save_path else None,
            show=False
        )
        
        # 3. Trend prediction
        print("[3/4] Creating trend prediction analysis...")
        self.create_trend_prediction(
            years_data=years_data,
            predict_years=10,
            save_path=str(save_path).replace('.png', '_predictions.png') if save_path else None,
            show=False
        )
        
        print("\n" + "=" * 80)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 80)
        
        if save_path:
            print(f"\nReports saved to:")
            print(f"  - Before/After: {str(save_path).replace('.png', '_before_after.png')}")
            print(f"  - Change Map: {str(save_path).replace('.png', '_change_map.png')}")
            print(f"  - Predictions: {str(save_path).replace('.png', '_predictions.png')}")


def main():
    """Test visualization functions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map Visualization and Prediction System')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline features file (.npy)')
    parser.add_argument('--current', type=str, required=True,
                        help='Current features file (.npy)')
    parser.add_argument('--condition', type=str, default='vegetation',
                        help='Environmental condition')
    parser.add_argument('--output', type=str, default='analysis_report.png',
                        help='Output path for report')
    
    args = parser.parse_args()
    
    # Load features
    baseline = np.load(args.baseline)
    current = np.load(args.current)
    
    # Simulate multi-year data (for demo)
    years_data = {
        2019: baseline,
        2021: (baseline + current) / 2,
        2024: current
    }
    
    # Create visualizer
    visualizer = MapVisualizer(baseline, current)
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(
        condition=args.condition,
        region_name='test_region',
        years_data=years_data,
        save_path=args.output
    )


if __name__ == '__main__':
    main()
