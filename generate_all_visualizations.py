"""
Generate comprehensive visualizations for Mangalore multi-year analysis
Creates all required charts: before/after, change detection, trends, yearly patterns
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("📊 GENERATING COMPREHENSIVE VISUALIZATIONS FOR MANGALORE")
print("=" * 80)

output_dir = Path("analysis_output/mangalore_multiyear")
output_dir.mkdir(parents=True, exist_ok=True)

# Load feature files for all years
years = list(range(2018, 2025))
feature_files = {}
all_features_by_year = {}

print("\n[1/5] Loading feature data...")
for year in years:
    feature_file = output_dir / f"features_{year}.npy"
    if feature_file.exists():
        features = np.load(feature_file)
        all_features_by_year[year] = features
        feature_files[year] = str(feature_file)
        print(f"  ✓ {year}: {features.shape[0]} samples, {features.shape[1]} features")
    else:
        print(f"  ✗ {year}: File not found")

if not all_features_by_year:
    print("ERROR: No feature files found!")
    sys.exit(1)

print(f"\n[2/5] Creating Before/After Comparison (2018 vs 2024)...")

# Create before/after comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Feature Distribution Comparison
for year in [2018, 2024]:
    features = all_features_by_year[year]
    means = features.mean(axis=1)
    color = '#3498db' if year == 2018 else '#e74c3c'
    label = f'{year} (Before)' if year == 2018 else f'{year} (After)'
    axes[0, 0].hist(means, bins=30, alpha=0.6, color=color, label=label, edgecolor='black')

axes[0, 0].set_title('Before vs After: Feature Distribution\n2018 vs 2024', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Mean Feature Value', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Statistical Comparison
categories = ['Mean', 'Std', 'Min', 'Max']
values_2018 = [
    all_features_by_year[2018].mean(),
    all_features_by_year[2018].std(),
    all_features_by_year[2018].min(),
    all_features_by_year[2018].max()
]
values_2024 = [
    all_features_by_year[2024].mean(),
    all_features_by_year[2024].std(),
    all_features_by_year[2024].min(),
    all_features_by_year[2024].max()
]

x = np.arange(len(categories))
width = 0.35
bars1 = axes[0, 1].bar(x - width/2, values_2018, width, label='2018', color='#3498db', edgecolor='black')
bars2 = axes[0, 1].bar(x + width/2, values_2024, width, label='2024', color='#e74c3c', edgecolor='black')
axes[0, 1].set_title('Statistical Comparison: 2018 vs 2024', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(categories, fontsize=10)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 3: PCA Visualization (Before vs After)
features_2018 = all_features_by_year[2018]
features_2024 = all_features_by_year[2024]

# Combine for consistent PCA
combined = np.vstack([features_2018, features_2024])
pca = PCA(n_components=2)
combined_2d = pca.fit_transform(combined)

features_2018_2d = combined_2d[:len(features_2018)]
features_2024_2d = combined_2d[len(features_2018):]

axes[1, 0].scatter(features_2018_2d[:, 0], features_2018_2d[:, 1], 
                  c='#3498db', s=10, alpha=0.5, label='2018')
axes[1, 0].scatter(features_2024_2d[:, 0], features_2024_2d[:, 1], 
                  c='#e74c3c', s=10, alpha=0.5, label='2024')
axes[1, 0].set_title('PCA: Environmental Patterns (2018 vs 2024)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Change Summary
change_percent = ((values_2024[0] - values_2018[0]) / abs(values_2018[0])) * 100
axes[1, 1].axis('off')

summary_text = f"""
BEFORE vs AFTER SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━

Year 2018 (Before):
  Mean: {values_2018[0]:.4f}
  Std: {values_2018[1]:.4f}
  Samples: {len(features_2018):,}

Year 2024 (After):
  Mean: {values_2024[0]:.4f}
  Std: {values_2024[1]:.4f}
  Samples: {len(features_2024):,}

━━━━━━━━━━━━━━━━━━━━━━━
CHANGE DETECTED:
  {change_percent:+.1f}% {'increase' if change_percent > 0 else 'decrease'}
━━━━━━━━━━━━━━━━━━━━━━━

Interpretation:
{'⚠️ Environmental stress detected' if change_percent < 0 else '✓ Positive growth observed'}
"""

axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f7fafc', edgecolor='#2d3748', linewidth=2))

plt.tight_layout()
before_after_path = output_dir / "before_after_comparison.png"
plt.savefig(before_after_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {before_after_path}")
plt.close()

print(f"\n[3/5] Creating Change Detection Map...")

# Create change detection visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Year-over-year change heatmap
change_data = []
year_labels = []
for i in range(len(years)-1):
    year1, year2 = years[i], years[i+1]
    change = (all_features_by_year[year2].mean() - all_features_by_year[year1].mean()) / abs(all_features_by_year[year1].mean()) * 100
    change_data.append(change)
    year_labels.append(f'{year1}-{year2}')

colors_change = ['#e74c3c' if c < 0 else '#27ae60' for c in change_data]
bars = axes[0, 0].bar(year_labels, change_data, color=colors_change, edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Year-over-Year Environmental Change', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Change (%)', fontsize=12)
axes[0, 0].axhline(y=0, color='black', linewidth=1, linestyle='-')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, change_data):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

# Plot 2: Cumulative change over time
cumulative_changes = []
base_mean = all_features_by_year[years[0]].mean()
for year in years:
    change = (all_features_by_year[year].mean() - base_mean) / abs(base_mean) * 100
    cumulative_changes.append(change)

axes[0, 1].plot(years, cumulative_changes, 'b-', linewidth=3, marker='o', markersize=10)
axes[0, 1].fill_between(years, cumulative_changes, alpha=0.3, color='blue')
axes[0, 1].set_title('Cumulative Environmental Change from Baseline', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Year', fontsize=12)
axes[0, 1].set_ylabel('Cumulative Change (%)', fontsize=12)
axes[0, 1].axhline(y=0, color='black', linewidth=1, linestyle='--')
axes[0, 1].grid(True, alpha=0.3)

for year, change in zip(years, cumulative_changes):
    axes[0, 1].annotate(f'{change:+.1f}%', (year, change), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')

# Plot 3: Variance changes over years
variances = [all_features_by_year[year].var() for year in years]
axes[1, 0].plot(years, variances, 'g-', linewidth=3, marker='s', markersize=10)
axes[1, 0].set_title('Environmental Variance Over Time', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Year', fontsize=12)
axes[1, 0].set_ylabel('Variance', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].fill_between(years, variances, alpha=0.3, color='green')

# Plot 4: Change intensity visualization
axes[1, 1].axis('off')

intensity_text = f"""
CHANGE DETECTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━

Time Period: {years[0]} - {years[-1]}
Total Years: {len(years)}

Overall Change: {cumulative_changes[-1]:+.1f}%
Average Yearly Change: {np.mean(change_data):+.1f}%

Trend Direction: {'📈 Increasing' if cumulative_changes[-1] > 0 else '📉 Decreasing'}

Key Observations:
"""

for i, (label, change) in enumerate(zip(year_labels, change_data)):
    direction = '↓' if change < 0 else '↑'
    intensity_text += f"\n• {label}: {change:+.1f}% {direction}"

axes[1, 1].text(0.05, 0.95, intensity_text, transform=axes[1, 1].transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f7fafc', edgecolor='#2d3748', linewidth=2))

plt.tight_layout()
change_map_path = output_dir / "change_detection_map.png"
plt.savefig(change_map_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {change_map_path}")
plt.close()

print(f"\n[4/5] Creating Year-by-Year Pattern Analysis...")

# Create individual year analysis
for year in years:
    features = all_features_by_year[year]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # PCA for this year
    pca_year = PCA(n_components=2)
    features_2d = pca_year.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    # Plot 1: PCA clusters
    scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=clusters, cmap='viridis', s=15, alpha=0.6)
    axes[0, 0].set_title(f'{year} - Environmental Patterns\n(PCA Visualization)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(f'PC1 ({pca_year.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    axes[0, 0].set_ylabel(f'PC2 ({pca_year.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
    
    # Plot 2: Cluster distribution
    cluster_counts = [(clusters == i).sum() for i in range(5)]
    colors_cluster = ['#27ae60', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    axes[0, 1].bar(range(5), cluster_counts, color=colors_cluster, edgecolor='black', alpha=0.8)
    axes[0, 1].set_title(f'{year} - Pattern Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Cluster ID', fontsize=11)
    axes[0, 1].set_ylabel('Number of Samples', fontsize=11)
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Feature distribution
    axes[1, 0].hist(features.mean(axis=1), bins=30, edgecolor='black', 
                   alpha=0.7, color='steelblue')
    axes[1, 0].set_title(f'{year} - Environmental Indicator Distribution', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Mean Feature Value', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axvline(features.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {features.mean():.4f}')
    axes[1, 0].legend()
    
    # Plot 4: Statistics
    axes[1, 1].axis('off')
    stats = {
        'Samples': features.shape[0],
        'Features': features.shape[1],
        'Mean': features.mean(),
        'Std': features.std(),
        'Min': features.min(),
        'Max': features.max()
    }
    
    table_data = [[key, f"{value:.4f}" if isinstance(value, float) else f"{value:,}"] 
                  for key, value in stats.items()]
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Statistic', 'Value'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title(f'{year} - Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    year_viz_path = output_dir / f"year_{year}_analysis.png"
    plt.savefig(year_viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {year_viz_path}")
    plt.close()

print(f"\n[5/5] Creating Historical Trend Analysis...")

# Create comprehensive trend analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Multi-year trend
means = [all_features_by_year[year].mean() for year in years]
stds = [all_features_by_year[year].std() for year in years]

axes[0, 0].plot(years, means, 'b-', linewidth=3, marker='o', markersize=10, label='Mean')
axes[0, 0].fill_between(years, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue', label='±1 Std')
axes[0, 0].set_title('Historical Environmental Trend (2018-2024)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Year', fontsize=12)
axes[0, 0].set_ylabel('Mean Feature Value', fontsize=12)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

for year, mean in zip(years, means):
    axes[0, 0].annotate(f'{mean:.3f}', (year, mean), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')

# Plot 2: Variance trend
variances = [all_features_by_year[year].var() for year in years]
axes[0, 1].bar(years, variances, color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Yearly Variance Trend', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Year', fontsize=12)
axes[0, 1].set_ylabel('Variance', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Sample distribution over years
sample_counts = [len(all_features_by_year[year]) for year in years]
axes[1, 0].plot(years, sample_counts, 'g-', linewidth=3, marker='s', markersize=10)
axes[1, 0].fill_between(years, sample_counts, alpha=0.3, color='green')
axes[1, 0].set_title('Sample Count Over Years', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Year', fontsize=12)
axes[1, 0].set_ylabel('Number of Samples', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

for year, count in zip(years, sample_counts):
    axes[1, 0].annotate(f'{count:,}', (year, count), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')

# Plot 4: Summary
axes[1, 1].axis('off')

trend_summary = f"""
HISTORICAL TREND ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━

Period: {years[0]} - {years[-1]} ({len(years)} years)

Overall Statistics:
  Mean: {np.mean(means):.4f}
  Std: {np.mean(stds):.4f}
  Total Samples: {sum(sample_counts):,}

Trend Analysis:
  Start ({years[0]}): {means[0]:.4f}
  End ({years[-1]}): {means[-1]:.4f}
  Change: {((means[-1] - means[0]) / abs(means[0]) * 100):+.1f}%

Rate of Change:
  {(means[-1] - means[0]) / (years[-1] - years[0]):.6f} per year

Assessment:
{'⚠️ Decreasing trend - Monitor closely' if means[-1] < means[0] else '✓ Increasing trend - Positive sign'}
"""

axes[1, 1].text(0.05, 0.95, trend_summary, transform=axes[1, 1].transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f7fafc', edgecolor='#2d3748', linewidth=2))

plt.tight_layout()
trend_path = output_dir / "historical_trend_analysis.png"
plt.savefig(trend_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {trend_path}")
plt.close()

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📁 Generated Files:")
print(f"  ✓ {before_after_path.name}")
print(f"  ✓ {change_map_path.name}")
print(f"  ✓ {trend_path.name}")
print(f"  ✓ year_2018_analysis.png to year_2024_analysis.png (7 files)")
print(f"\n📊 Total visualizations: 11 files")
print("=" * 80)
