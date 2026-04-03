"""
Interactive CLI for environmental condition analysis
Usage: python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_region_config
from inference.environmental_analyzer import EnvironmentalConditionAnalyzer
from inference.map_visualization import MapVisualizer


def analyze_environmental_change(
    region: str,
    condition: str,
    baseline_year: int,
    current_year: int,
    baseline_dates: tuple = None,
    current_dates: tuple = None
):
    """
    Analyze specific environmental condition change for a region
    
    Args:
        region: Region name (assam, mangalore, etc.)
        condition: Condition to analyze (vegetation, water, urban, land_degradation, environmental_stress)
        baseline_year: Baseline year
        current_year: Current year
        baseline_dates: Optional (start_date, end_date) for baseline
        current_dates: Optional (start_date, end_date) for current
    """
    
    print("=" * 80)
    print("ENVIRONMENTAL CONDITION ANALYSIS")
    print("=" * 80)
    print(f"Region: {region.upper()}")
    print(f"Condition: {condition.upper().replace('_', ' ')}")
    print(f"Time Period: {baseline_year} → {current_year}")
    print("=" * 80)
    
    # Get region configuration
    region_config = get_region_config(region)
    if not region_config:
        print(f"ERROR: Region '{region}' not found!")
        print("Available regions: assam, mangalore, amazon, congo")
        return
    
    # Default date ranges (dry season for better analysis)
    if baseline_dates is None:
        baseline_dates = (f"{baseline_year}-01-01", f"{baseline_year}-03-31")
    if current_dates is None:
        current_dates = (f"{current_year}-01-01", f"{current_year}-03-31")
    
    # Check if feature files exist
    baseline_feature_file = PROJECT_ROOT / f"features_{region}_{baseline_year}.npy"
    current_feature_file = PROJECT_ROOT / f"features_{region}_{current_year}.npy"
    
    if not baseline_feature_file.exists():
        print(f"\nWARNING: Baseline features not found: {baseline_feature_file}")
        print(f"To generate features for {baseline_year}, run:")
        print(f"  python run_full_workflow.py --region {region} --start-date {baseline_dates[0]} --end-date {baseline_dates[1]}")
        return
    
    if not current_feature_file.exists():
        print(f"\nWARNING: Current features not found: {current_feature_file}")
        print(f"To generate features for {current_year}, run:")
        print(f"  python run_full_workflow.py --region {region} --start-date {current_dates[0]} --end-date {current_dates[1]}")
        return
    
    # Load features
    print(f"\nLoading baseline features ({baseline_year})...")
    baseline_features = np.load(baseline_feature_file)
    print(f"  Loaded {len(baseline_features)} samples")
    
    print(f"\nLoading current features ({current_year})...")
    current_features = np.load(current_feature_file)
    print(f"  Loaded {len(current_features)} samples")
    
    # Create analyzer
    analyzer = EnvironmentalConditionAnalyzer(baseline_features)
    
    # Generate analysis report
    print("\n")
    analyzer.generate_report(baseline_features, current_features, condition)
    
    # Create multi-year data for predictions
    years_data = {
        baseline_year: baseline_features,
        (baseline_year + current_year) // 2: (baseline_features + current_features) / 2,
        current_year: current_features
    }
    
    # Generate comprehensive visualizations
    print("\n" + "=" * 80)
    print("GENERATING MAP VISUALIZATIONS AND PREDICTIONS")
    print("=" * 80)
    
    visualizer = MapVisualizer(baseline_features, current_features)
    
    output_dir = PROJECT_ROOT / f"analysis_{region}" / f"{condition}_change"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    report_path = output_dir / f"comprehensive_analysis_{condition}.png"
    visualizer.generate_comprehensive_report(
        condition=condition,
        region_name=region,
        years_data=years_data,
        save_path=str(report_path)
    )
    
    # Save visualization
    output_dir = PROJECT_ROOT / f"analysis_{region}" / f"{condition}_change"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = output_dir / f"{condition}_{baseline_year}_vs_{current_year}.png"
    analyzer.visualize_comparison(baseline_features, current_features, save_path=str(viz_path))
    
    # Save detailed results
    result_file = output_dir / f"analysis_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Environmental Condition Analysis Report\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Region: {region}\n")
        f.write(f"Condition: {condition}\n")
        f.write(f"Baseline Year: {baseline_year}\n")
        f.write(f"Current Year: {current_year}\n")
        f.write(f"Analysis Date: {datetime.now()}\n\n")
        
        # Run specific analysis and save results
        if condition == 'all':
            analyses = {
                'vegetation': analyzer.detect_vegetation_change,
                'water': analyzer.detect_water_expansion,
                'urban': analyzer.detect_urban_growth,
                'land_degradation': analyzer.detect_land_degradation,
                'environmental_stress': analyzer.detect_environmental_stress
            }
            
            for cond_name, cond_func in analyses.items():
                f.write(f"\n{'=' * 60}\n")
                f.write(f"{cond_name.upper().replace('_', ' ')} ANALYSIS\n")
                f.write(f"{'=' * 60}\n")
                result = cond_func(baseline_features, current_features)
                for key, value in result.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        else:
            cond_map = {
                'vegetation': analyzer.detect_vegetation_change,
                'water': analyzer.detect_water_expansion,
                'urban': analyzer.detect_urban_growth,
                'land_degradation': analyzer.detect_land_degradation,
                'environmental_stress': analyzer.detect_environmental_stress
            }
            result = cond_map[condition](baseline_features, current_features)
            for key, value in result.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    print(f"\nResults saved to:")
    print(f"  - Visualization: {viz_path}")
    print(f"  - Report: {result_file}")
    print(f"  - Comprehensive Analysis: {report_path}")
    print(f"  - Change Map: {output_dir / f'{condition}_change_map.png'}")
    print(f"  - Predictions: {output_dir / f'{condition}_predictions.png'}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Analyze specific environmental conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze vegetation change in Assam from 2019 to 2024
  python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
  
  # Analyze water expansion in Mangalore
  python analyze_condition.py --region mangalore --condition water --year1 2020 --year2 2024
  
  # Analyze all conditions for a region
  python analyze_condition.py --region assam --condition all --year1 2019 --year2 2024
        """
    )
    
    parser.add_argument('--region', type=str, required=True,
                        help='Region name (assam, mangalore, amazon, congo)')
    parser.add_argument('--condition', type=str, default='all',
                        choices=['vegetation', 'water', 'urban', 'land_degradation', 
                                'environmental_stress', 'all'],
                        help='Environmental condition to analyze')
    parser.add_argument('--year1', type=int, required=True,
                        help='Baseline year')
    parser.add_argument('--year2', type=int, required=True,
                        help='Current year')
    parser.add_argument('--date-range1', type=str, nargs=2, default=None,
                        help='Baseline date range (YYYY-MM-DD YYYY-MM-DD)')
    parser.add_argument('--date-range2', type=str, nargs=2, default=None,
                        help='Current date range (YYYY-MM-DD YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    analyze_environmental_change(
        region=args.region,
        condition=args.condition,
        baseline_year=args.year1,
        current_year=args.year2,
        baseline_dates=args.date_range1,
        current_dates=args.date_range2
    )


if __name__ == '__main__':
    import numpy as np
    main()
