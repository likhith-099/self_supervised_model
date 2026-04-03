"""
Fully Automated Environmental Analysis System
User inputs: Place name, Condition, Timeline
System outputs: Complete environmental analysis report

Usage: python auto_analyze.py --place "Mangalore, India" --condition vegetation --start 2019-01-01 --end 2024-01-01
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.geocoder import Geocoder


def run_complete_analysis(place_name: str, condition: str, 
                         start_date: str, end_date: str,
                         timeline_years: list = None):
    """
    Complete automated environmental analysis
    
    Args:
        place_name: Location name (e.g., "Mangalore, India")
        condition: Environmental condition (vegetation, water, urban, land_degradation, environmental_stress)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeline_years: List of years for time-series analysis (default: [start_year, end_year])
    """
    
    print("=" * 80)
    print("FULLY AUTOMATED ENVIRONMENTAL ANALYSIS SYSTEM")
    print("=" * 80)
    print(f"Location: {place_name}")
    print(f"Condition: {condition.upper()}")
    print(f"Timeline: {start_date} to {end_date}")
    print("=" * 80)
    
    # Step 1: Geocode location
    print("\n[PHASE 1] Converting place name to coordinates...")
    print("-" * 80)
    geocoder = Geocoder()
    location_info = geocoder.get_region_coordinates(place_name)
    
    if not location_info:
        print(f"ERROR: Could not locate '{place_name}'")
        print("Please try a more specific location name")
        return
    
    bbox = location_info['bbox']
    region_name = place_name.replace(" ", "_").replace(",", "").lower()
    
    print(f"✓ Location: {location_info['display_name']}")
    print(f"✓ Bounding Box: {bbox}")
    print()
    
    # Step 2: Download satellite images
    print("[PHASE 2] Downloading Sentinel-2 satellite imagery...")
    print("-" * 80)
    download_cmd = (
        f"python preprocessing/download_sentinel_api.py "
        f"--place \"{place_name}\" "
        f"--condition {condition} "
        f"--start {start_date} "
        f"--end {end_date} "
        f"--limit 50"
    )
    print(f"Executing: {download_cmd}")
    os.system(download_cmd)
    print()
    
    # Step 3: Preprocess and generate tiles
    print("[PHASE 3] Preprocessing and generating tiles...")
    print("-" * 80)
    preprocess_cmd = (
        f"python preprocessing/cloud_filter.py "
        f"--input data/raw/sentinel "
        f"--output data/raw/filtered "
        f"--max-cloud 30.0; "
        f"python preprocessing/tile_generator.py "
        f"--input data/raw/filtered "
        f"--output data/tiles "
        f"--tile-size 128 "
        f"--train-ratio 0.8; "
        f"python preprocessing/normalize.py "
        f"--input data/tiles/train "
        f"--output data/tiles_normalized "
        f"--method sentinel2"
    )
    os.system(preprocess_cmd)
    print()
    
    # Step 4: Train MAE model (first time period only)
    print("[PHASE 4] Training MAE model...")
    print("-" * 80)
    train_cmd = (
        f"python training/train_mae.py "
        f"--train-dir data/tiles/train "
        f"--val-dir data/tiles/val "
        f"--batch-size 64 "
        f"--epochs 200 "
        f"--lr 1e-4 "
        f"--img-size 128 "
        f"--device cuda"
    )
    os.system(train_cmd)
    print()
    
    # Step 5: Extract features for all time periods
    print("[PHASE 5] Extracting feature embeddings...")
    print("-" * 80)
    
    if timeline_years is None:
        # Parse years from dates
        start_year = start_date.split('-')[0]
        end_year = end_date.split('-')[0]
        timeline_years = [start_year, end_year]
    
    feature_files = []
    for year in timeline_years:
        print(f"\nExtracting features for year {year}...")
        extract_cmd = (
            f"python inference/extract_features.py "
            f"--checkpoint checkpoints/mae_encoder.pth "
            f"--input data/tiles/val "
            f"--output features_{region_name}_{year}.npy "
            f"--device cuda"
        )
        os.system(extract_cmd)
        feature_files.append(f"features_{region_name}_{year}.npy")
    
    print()
    
    # Step 6: Analyze environmental changes
    print("[PHASE 6] Analyzing environmental changes...")
    print("-" * 80)
    
    if len(feature_files) >= 2:
        analyze_cmd = (
            f"python analyze_condition.py "
            f"--region {region_name} "
            f"--condition {condition} "
            f"--year1 {timeline_years[0]} "
            f"--year2 {timeline_years[-1]}"
        )
        os.system(analyze_cmd)
    else:
        print("WARNING: Need at least 2 time periods for comparison")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in:")
    print(f"  - Features: {feature_files}")
    print(f"  - Analysis: analysis_{region_name}/{condition}_change/")
    print(f"  - Visualizations: PNG charts and graphs")
    print(f"\nReport location: analysis_{region_name}/{condition}_change/analysis_results.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Fully Automated Environmental Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze vegetation change in Mangalore from 2019 to 2024
  python auto_analyze.py --place "Mangalore, India" --condition vegetation --start 2019-01-01 --end 2024-01-01
  
  # Analyze water expansion in Assam
  python auto_analyze.py --place "Assam, India" --condition water --start 2020-01-01 --end 2024-01-01
  
  # Analyze urban growth with custom timeline
  python auto_analyze.py --place "New York, USA" --condition urban --start 2018-06-01 --end 2024-06-01
        """
    )
    
    parser.add_argument('--place', type=str, required=True,
                        help='Place name (e.g., "Mangalore, India", "Assam, India")')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['vegetation', 'water', 'urban', 'land_degradation', 
                                'environmental_stress', 'all'],
                        help='Environmental condition to analyze')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeline', type=str, nargs='+', default=None,
                        help='Specific years for analysis (e.g., --timeline 2019 2021 2024)')
    parser.add_argument('--images', type=int, default=50,
                        help='Number of satellite images to download (default: 50)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    
    args = parser.parse_args()
    
    run_complete_analysis(
        place_name=args.place,
        condition=args.condition,
        start_date=args.start,
        end_date=args.end,
        timeline_years=args.timeline
    )


if __name__ == '__main__':
    main()
