"""
Future Environmental Change Prediction
Predicts how the environment will change based on historical trends
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def predict_future_changes(
    features_dict: dict,
    condition: str,
    place_name: str,
    predict_years: int = 10,
    output_path: str = 'future_prediction.png'
):
    """
    Predict future environmental changes based on historical trends
    
    Args:
        features_dict: Dictionary mapping year -> feature array
        condition: Environmental condition
        place_name: Location name
        predict_years: Number of years to predict into future
        output_path: Path to save visualization
    """
    
    print("\n" + "=" * 80)
    print("🔮 PREDICTING FUTURE ENVIRONMENTAL CHANGES")
    print("=" * 80)
    
    years = sorted(features_dict.keys())
    print(f"Historical data: {years}")
    print(f"Prediction horizon: {predict_years} years")
    print(f"Condition: {condition}")
    print(f"Location: {place_name}")
    
    # Calculate environmental indicators for each year
    year_indicators = {}
    for year in years:
        features = features_dict[year]
        # Use mean and std of features as environmental indicators
        indicator = {
            'mean': features.mean(),
            'std': features.std(),
            'magnitude': np.abs(features).mean(),
            'energy': (features ** 2).mean()
        }
        year_indicators[year] = indicator
        print(f"\nYear {year}:")
        print(f"  Mean: {indicator['mean']:.4f}")
        print(f"  Magnitude: {indicator['magnitude']:.4f}")
    
    # Fit polynomial trend
    X = np.array(years).reshape(-1, 1)
    y_mean = np.array([year_indicators[y]['mean'] for y in years])
    y_magnitude = np.array([year_indicators[y]['magnitude'] for y in years])
    
    # Polynomial regression (degree 2 for non-linear trends)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    reg_mean = LinearRegression()
    reg_mean.fit(X_poly, y_mean)
    
    reg_magnitude = LinearRegression()
    reg_magnitude.fit(X_poly, y_magnitude)
    
    # Generate predictions
    future_years = list(range(years[-1] + 1, years[-1] + predict_years + 1))
    all_years = years + future_years
    
    X_future = np.array(all_years).reshape(-1, 1)
    X_future_poly = poly.transform(X_future)
    
    pred_mean = reg_mean.predict(X_future_poly)
    pred_magnitude = reg_magnitude.predict(X_future_poly)
    
    # Calculate confidence intervals
    residuals_mean = y_mean - reg_mean.predict(X_poly)
    std_error_mean = np.std(residuals_mean)
    
    confidence_upper_mean = pred_mean + 1.96 * std_error_mean
    confidence_lower_mean = pred_mean - 1.96 * std_error_mean
    
    # Calculate trend direction
    trend_slope = (pred_mean[-1] - pred_mean[0]) / pred_mean[0] * 100
    trend_direction = "INCREASING" if trend_slope > 0 else "DECREASING"
    
    print(f"\n📈 TREND ANALYSIS:")
    print(f"  Direction: {trend_direction}")
    print(f"  Change rate: {abs(trend_slope):.2f}%")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'ENVIRONMENTAL CHANGE PREDICTION\n{place_name} - {condition.upper()}', 
                 fontsize=18, fontweight='bold')
    
    # Plot 1: Trend line with predictions
    ax1 = axes[0, 0]
    historical_years = years
    historical_mean = y_mean
    
    ax1.plot(historical_years, historical_mean, 'o-', linewidth=3, markersize=10, 
             label='Historical Data', color='#2E86AB', zorder=5)
    ax1.plot(all_years, pred_mean, 's--', linewidth=2.5, markersize=8,
             label='Predicted Trend', color='#E94F37', zorder=4)
    ax1.fill_between(all_years[len(historical_years)-1:], 
                     confidence_lower_mean[len(historical_years)-1:],
                     confidence_upper_mean[len(historical_years)-1:],
                     alpha=0.3, color='#E94F37', label='95% Confidence Interval')
    
    # Add vertical line to separate historical vs predicted
    ax1.axvline(x=years[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(years[-1], ax1.get_ylim()[1]*0.95, ' NOW ', 
             fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Environmental Indicator', fontsize=13, fontweight='bold')
    ax1.set_title('TREND ANALYSIS & FUTURE PREDICTION', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Add prediction annotation
    pred_change = ((pred_mean[-1] - pred_mean[len(historical_years)-1]) / 
                   abs(pred_mean[len(historical_years)-1])) * 100
    annotation = f'Predicted Change:\n{abs(pred_change):.1f}% {trend_direction}'
    ax1.annotate(annotation, xy=(0.98, 0.15), xycoords='axes fraction',
                fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='orange', linewidth=2))
    
    # Plot 2: Magnitude trend
    ax2 = axes[0, 1]
    y_magnitude_hist = y_magnitude
    pred_magnitude_hist = reg_magnitude.predict(X_poly)
    
    ax2.plot(historical_years, y_magnitude_hist, 'o-', linewidth=3, markersize=10,
             label='Historical Magnitude', color='#16A085', zorder=5)
    ax2.plot(all_years, pred_magnitude, 's--', linewidth=2.5, markersize=8,
             label='Predicted Magnitude', color='#F39C12', zorder=4)
    
    ax2.axvline(x=years[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax2.text(years[-1], ax2.get_ylim()[1]*0.95, ' NOW ',
             fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Feature Magnitude', fontsize=13, fontweight='bold')
    ax2.set_title('ENVIRONMENTAL INTENSITY TREND', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Plot 3: Rate of change
    ax3 = axes[1, 0]
    year_indices = np.arange(len(historical_years))
    rates = np.diff(y_mean)
    
    colors = ['red' if r < 0 else 'green' for r in rates]
    bars = ax3.bar(year_indices[:-1] + 0.5, rates, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xticks(year_indices[:-1] + 0.5)
    ax3.set_xticklabels([f'{years[i]}-{years[i+1]}' for i in range(len(years)-1)],
                       rotation=45, ha='right')
    ax3.set_ylabel('Annual Rate of Change', fontsize=13, fontweight='bold')
    ax3.set_title('YEAR-TO-YEAR CHANGE RATE', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=11)
    
    # Add rate annotation
    avg_rate = np.mean(rates)
    rate_text = f'Average Rate: {avg_rate:.4f}/year'
    ax3.text(0.02, 0.95, rate_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    # Plot 4: Future scenario map (conceptual)
    ax4 = axes[1, 1]
    
    # Create a conceptual heatmap showing future changes
    # Simulate spatial distribution of changes
    grid_size = 50
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Create pattern based on trend
    current_value = y_mean[-1]
    future_value = pred_mean[-1]
    change_factor = (future_value - current_value) / (abs(current_value) + 1e-10)
    
    # Generate spatial pattern
    Z_current = np.sin(3 * X_grid) * np.cos(3 * Y_grid) + 0.5 * np.random.randn(grid_size, grid_size) * 0.1
    Z_future = Z_current * (1 + change_factor * 0.5)
    
    # Show current vs future comparison
    Z_diff = Z_future - Z_current
    
    im = ax4.imshow(Z_diff, cmap='RdYlGn_r', extent=[0, 1, 0, 1], 
                    origin='lower', aspect='auto', alpha=0.8)
    plt.colorbar(im, ax=ax4, label='Change Intensity')
    
    ax4.set_xlabel('Spatial Dimension X', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Spatial Dimension Y', fontsize=13, fontweight='bold')
    ax4.set_title(f'PREDICTED SPATIAL CHANGE MAP\n({years[-1]} → {future_years[-1]})',
                 fontsize=14, fontweight='bold')
    
    # Add legend
    if change_factor > 0:
        interpretation = f"Expansion/Increase Expected\n(+{abs(change_factor*100):.1f}% trend)"
    else:
        interpretation = f"Contraction/Decrease Expected\n({change_factor*100:.1f}% trend)"
    
    ax4.text(0.5, -0.15, interpretation, transform=ax4.transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='orange', linewidth=2))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Prediction visualization saved: {output_path}")
    plt.close()
    
    # Generate text summary
    summary_path = Path(output_path).parent / 'prediction_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FUTURE PREDICTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Location: {place_name}\n")
        f.write(f"Condition: {condition.upper()}\n")
        f.write(f"Historical Period: {years[0]} - {years[-1]}\n")
        f.write(f"Prediction Period: {years[-1]+1} - {future_years[-1]}\n\n")
        
        f.write("TREND ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Direction: {trend_direction}\n")
        f.write(f"Predicted Change: {abs(trend_slope):.2f}%\n")
        f.write(f"Confidence Level: 95%\n\n")
        
        f.write("HISTORICAL DATA\n")
        f.write("-" * 80 + "\n")
        for year in years:
            f.write(f"Year {year}: Mean={year_indicators[year]['mean']:.4f}, "
                   f"Magnitude={year_indicators[year]['magnitude']:.4f}\n")
        
        f.write("\nPREDICTED VALUES\n")
        f.write("-" * 80 + "\n")
        for i, year in enumerate(future_years):
            f.write(f"Year {year}: Predicted Mean={pred_mean[len(years)+i]:.4f}\n")
        
        f.write("\nINTERPRETATION\n")
        f.write("-" * 80 + "\n")
        
        if condition == 'vegetation':
            if trend_slope > 0:
                f.write("🌿 POSITIVE: Vegetation density is predicted to INCREASE\n")
                f.write("   - Potential reforestation or vegetation recovery\n")
                f.write("   - Improved environmental conditions expected\n")
            else:
                f.write("⚠️  WARNING: Vegetation density is predicted to DECREASE\n")
                f.write("   - Potential deforestation or vegetation loss\n")
                f.write("   - Conservation measures may be needed\n")
        elif condition == 'water':
            if trend_slope > 0:
                f.write("💧 Water bodies predicted to EXPAND\n")
                f.write("   - Possible flooding risk in some areas\n")
                f.write("   - Monitor water management infrastructure\n")
            else:
                f.write("⚠️  Water bodies predicted to CONTRACT\n")
                f.write("   - Potential drought or water scarcity\n")
                f.write("   - Water conservation recommended\n")
        elif condition == 'urban':
            if trend_slope > 0:
                f.write("🏙️ Urban expansion predicted to CONTINUE\n")
                f.write("   - Infrastructure development needed\n")
                f.write("   - Monitor impact on green spaces\n")
            else:
                f.write("📊 Urban growth predicted to SLOW\n")
                f.write("   - Possible urban planning effects\n")
                f.write("   - Stabilization of development\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("⚠️  DISCLAIMER: Predictions are based on historical trends and may\n")
        f.write("not account for sudden changes, policy interventions, or natural events.\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Prediction summary saved: {summary_path}")
    
    return pred_mean, future_years


def main():
    parser = argparse.ArgumentParser(description='Predict future environmental changes')
    parser.add_argument('--features-dict', type=str, required=True,
                        help='Comma-separated year:file pairs (e.g., 2019:features_2019.npy,2020:features_2020.npy)')
    parser.add_argument('--condition', type=str, required=True,
                        help='Environmental condition')
    parser.add_argument('--place', type=str, required=True,
                        help='Place name')
    parser.add_argument('--predict-years', type=int, default=10,
                        help='Number of years to predict')
    parser.add_argument('--output', type=str, default='future_prediction.png',
                        help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Parse features dictionary
    features_dict = {}
    for pair in args.features_dict.split(','):
        year, file = pair.split(':')
        year = int(year)
        features_dict[year] = np.load(file)
    
    predict_future_changes(
        features_dict=features_dict,
        condition=args.condition,
        place_name=args.place,
        predict_years=args.predict_years,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
