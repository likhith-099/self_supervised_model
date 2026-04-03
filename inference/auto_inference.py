"""
Inference-only Environmental Analysis System
Uses pre-trained MAE model for analysis
No training required during inference

Usage: python inference/auto_inference.py --place "Mangalore, India" --condition vegetation --start 2019-01-01 --end 2024-01-01
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.geocoder import Geocoder


def check_model_exists(checkpoint_path: str) -> bool:
    """Check if pre-trained model exists"""
    return Path(checkpoint_path).exists()


def run_inference_analysis(
    place_name: str, 
    condition: str, 
    start_date: str, 
    end_date: str,
    checkpoint_path: str = "checkpoints/mae_encoder.pth",
    timeline_years: list = None
):
    """
    Run inference using pre-trained MAE model
    
    Args:
        place_name: Location name
        condition: Environmental condition
        start_date: Start date
        end_date: End date
        checkpoint_path: Path to pre-trained model
        timeline_years: List of years for analysis
    """
    
    print("=" * 80)
    print("AUTOMATED ENVIRONMENTAL ANALYSIS (INFERENCE MODE)")
    print("=" * 80)
    print(f"Location: {place_name}")
    print(f"Condition: {condition.upper()}")
    print(f"Timeline: {start_date} to {end_date}")
    print("=" * 80)
    
    # Check if model exists
    if not check_model_exists(checkpoint_path):
        print(f"\n❌ ERROR: Pre-trained model not found at {checkpoint_path}")
        print("\nPlease train the model first using one of these methods:")
        print("  1. Run: python training/train_mae.py --train-dir data/tiles_normalized --val-dir data/tiles/val --epochs 200")
        print("  2. Or use the full workflow: python run_full_workflow.py --place \"Your Location\" --condition vegetation")
        sys.exit(1)
    
    print(f"\n✓ Using pre-trained model: {checkpoint_path}")
    
    # Step 1: Geocode location
    print("\n[STEP 1] Converting place name to coordinates...")
    print("-" * 80)
    geocoder = Geocoder()
    location_info = geocoder.get_region_coordinates(place_name)
    
    if not location_info:
        print(f"ERROR: Could not locate '{place_name}'")
        return
    
    bbox = location_info['bbox']
    region_name = place_name.replace(" ", "_").replace(",", "").lower()
    
    print(f"✓ Location: {location_info['display_name']}")
    print(f"✓ Bounding Box: {bbox}")
    
    # Step 2: Download satellite images for this region
    print("\n[STEP 2] Downloading Sentinel-2 imagery for analysis...")
    print("-" * 80)
    
    # Create clean directory for this analysis
    analysis_dir = f"data/inference_{region_name}"
    if Path(analysis_dir).exists():
        shutil.rmtree(analysis_dir)
    
    download_cmd = (
        f"python preprocessing/download_sentinel_api.py "
        f"--place \"{place_name}\" "
        f"--condition {condition} "
        f"--start {start_date} "
        f"--end {end_date} "
        f"--limit 50 "
        f"--output-dir {analysis_dir}"
    )
    print(f"Executing: {download_cmd}")
    os.system(download_cmd)
    
    # Step 3: Preprocess downloaded images
    print("\n[STEP 3] Preprocessing images...")
    print("-" * 80)
    
    filtered_dir = f"{analysis_dir}_filtered"
    tiles_dir = f"{analysis_dir}_tiles"
    
    preprocess_cmd = (
        f"python preprocessing/cloud_filter.py "
        f"--input {analysis_dir} "
        f"--output {filtered_dir} "
        f"--max-cloud 30.0; "
        f"python preprocessing/tile_generator.py "
        f"--input {filtered_dir} "
        f"--output {tiles_dir} "
        f"--tile-size 128 "
        f"--train-ratio 0.0"  # No train/val split needed for inference
    )
    print(f"Preprocessing: {preprocess_cmd}")
    os.system(preprocess_cmd)
    
    # Step 4: Normalize tiles
    print("\n[STEP 4] Normalizing tiles...")
    print("-" * 80)
    
    normalized_dir = f"{tiles_dir}_normalized"
    normalize_cmd = (
        f"python preprocessing/normalize.py "
        f"--input {tiles_dir} "
        f"--output {normalized_dir} "
        f"--method sentinel2"
    )
    os.system(normalize_cmd)
    
    # Step 5: Extract features using pre-trained model
    print("\n[STEP 5] Extracting features with pre-trained MAE...")
    print("-" * 80)
    
    if timeline_years is None:
        start_year = start_date.split('-')[0]
        end_year = end_date.split('-')[0]
        timeline_years = [start_year, end_year]
    
    feature_files = []
    for year in timeline_years:
        print(f"\nExtracting features for year {year}...")
        
        # For single period analysis, use all tiles
        feature_file = f"features_{region_name}_{year}.npy"
        
        extract_cmd = (
            f"python inference/extract_features.py "
            f"--checkpoint {checkpoint_path} "
            f"--input {normalized_dir} "
            f"--output {feature_file} "
            f"--device cuda"
        )
        os.system(extract_cmd)
        feature_files.append(feature_file)
    
    # Step 6: Analyze features
    print("\n[STEP 6] Analyzing environmental patterns...")
    print("-" * 80)
    
    output_dir = f"analysis_{region_name}/{condition}_inference"
    
    if len(feature_files) >= 1:
        # Use most recent features for analysis
        latest_features = feature_files[-1]
        
        analyze_cmd = (
            f"python inference/analyze_region.py "
            f"--features {latest_features} "
            f"--n-clusters 5 "
            f"--pca-dims 2 "
            f"--tsne "
            f"--output-dir {output_dir}"
        )
        print(f"Analyzing: {analyze_cmd}")
        os.system(analyze_cmd)
    
    # Cleanup temporary directories
    print("\n[CLEANUP] Removing temporary files...")
    for dir_to_remove in [analysis_dir, filtered_dir, tiles_dir, normalized_dir]:
        if Path(dir_to_remove).exists():
            try:
                shutil.rmtree(dir_to_remove)
                print(f"  ✓ Removed {dir_to_remove}")
            except Exception as e:
                print(f"  ⚠ Could not remove {dir_to_remove}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in:")
    print(f"  - Features: {feature_files}")
    print(f"  - Analysis: {output_dir}/")
    print(f"  - Visualizations: {output_dir}/pca_kmeans_clusters.png")
    print(f"  - Statistics: {output_dir}/cluster_statistics.txt")
    print(f"\n✨ Used pre-trained model - no training required!")


def main():
    parser = argparse.ArgumentParser(
        description='Inference-based Environmental Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze vegetation in Mangalore (uses pre-trained model)
  python inference/auto_inference.py --place "Mangalore, India" --condition vegetation --start 2024-01-01 --end 2024-06-01
  
  # Analyze water bodies in Assam
  python inference/auto_inference.py --place "Assam, India" --condition water --start 2024-06-01 --end 2024-09-01
  
  # Multi-year comparison
  python inference/auto_inference.py --place "Dubai, UAE" --condition urban --start 2020-01-01 --end 2024-01-01 --timeline 2020 2022 2024
        """
    )
    
    parser.add_argument('--place', type=str, required=True,
                        help='Place name (e.g., "Mangalore, India")')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['vegetation', 'water', 'urban', 'land_degradation', 
                                'environmental_stress', 'all'],
                        help='Environmental condition to analyze')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeline', type=str, nargs='+', default=None,
                        help='Specific years for analysis')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mae_encoder.pth',
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--images', type=int, default=50,
                        help='Number of satellite images to download')
    
    args = parser.parse_args()
    
    run_inference_analysis(
        place_name=args.place,
        condition=args.condition,
        start_date=args.start,
        end_date=args.end,
        checkpoint_path=args.checkpoint,
        timeline_years=args.timeline
    )


if __name__ == '__main__':
    main()
