"""
Complete workflow automation script
Runs all preprocessing, training, and inference steps sequentially
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config, get_region_config


def run_workflow(
    region_name: str,
    start_date: str,
    end_date: str,
    cloud_threshold: float = None,
    tile_size: int = 128,
    train_epochs: int = 400,
    skip_download: bool = False,
    skip_training: bool = False
):
    """
    Execute complete ML pipeline workflow
    
    Args:
        region_name: Target region (e.g., 'assam', 'mangalore')
        start_date: Start date for image search (YYYY-MM-DD)
        end_date: End date for image search (YYYY-MM-DD)
        cloud_threshold: Maximum cloud coverage percentage
        tile_size: Size of tiles in pixels
        train_epochs: Number of training epochs
        skip_download: Skip download step if True
        skip_training: Skip training step if True
    """
    
    print("=" * 80)
    print("ML WORKFLOW AUTOMATION")
    print("=" * 80)
    
    # Get region configuration
    region_config = get_region_config(region_name)
    if not region_config:
        print(f"ERROR: Region '{region_name}' not found in configuration!")
        print("Available regions: assam, mangalore, amazon, congo")
        return
    
    bbox = region_config['bbox']
    if cloud_threshold is None:
        cloud_threshold = region_config['cloud_threshold']
    
    print(f"Region: {region_config['name']}")
    print(f"Bounding Box: {bbox}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Cloud Threshold: {cloud_threshold}%")
    print(f"Tile Size: {tile_size}x{tile_size}")
    print("=" * 80)
    
    # Step 1: Download satellite images
    if not skip_download:
        print("\n[STEP 1/6] Downloading Satellite Images...")
        print("-" * 80)
        os.system(f"python preprocessing/download_sentinel.py "
                  f"--output data/raw/sentinel "
                  f"--bbox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} "
                  f"--start-date {start_date} "
                  f"--end-date {end_date} "
                  f"--cloud-coverage {cloud_threshold} "
                  f"--limit 100")
        print()
    
    # Step 2: Filter cloudy images
    print("\n[STEP 2/6] Filtering Cloudy Images...")
    print("-" * 80)
    os.system(f"python preprocessing/cloud_filter.py "
              f"--input data/raw/sentinel "
              f"--output data/raw/filtered "
              f"--max-cloud {cloud_threshold}")
    print()
    
    # Step 3: Generate tiles
    print("\n[STEP 3/6] Generating Training Tiles...")
    print("-" * 80)
    os.system(f"python preprocessing/tile_generator.py "
              f"--input data/raw/filtered "
              f"--output data/tiles "
              f"--tile-size {tile_size} "
              f"--train-ratio 0.8")
    print()
    
    # Step 4: Normalize tiles
    print("\n[STEP 4/6] Normalizing Tiles...")
    print("-" * 80)
    os.system(f"python preprocessing/normalize.py "
              f"--input data/tiles/train "
              f"--output data/tiles_normalized "
              f"--method sentinel2")
    print()
    
    # Step 5: Train MAE model
    if not skip_training:
        print("\n[STEP 5/6] Training MAE Model...")
        print("-" * 80)
        os.system(f"python training/train_mae.py "
                  f"--train-dir data/tiles/train "
                  f"--val-dir data/tiles/val "
                  f"--batch-size 64 "
                  f"--epochs {train_epochs} "
                  f"--lr 1e-4 "
                  f"--img-size {tile_size} "
                  f"--device cuda")
        print()
    
    # Step 6: Extract features
    print("\n[STEP 6/6] Extracting Features...")
    print("-" * 80)
    checkpoint_path = PROJECT_ROOT / 'checkpoints' / 'mae_encoder.pth'
    if checkpoint_path.exists():
        os.system(f"python inference/extract_features.py "
                  f"--checkpoint checkpoints/mae_encoder.pth "
                  f"--input data/tiles/val "
                  f"--output features_{region_name}.npy "
                  f"--device cuda")
        
        # Analyze region
        os.system(f"python inference/analyze_region.py "
                  f"--features features_{region_name}.npy "
                  f"--n-clusters 5 "
                  f"--tsne "
                  f"--output-dir analysis_{region_name}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Skipping feature extraction (model not trained yet)")
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved in:")
    print(f"  - Features: features_{region_name}.npy")
    print(f"  - Analysis: analysis_{region_name}/")
    print(f"  - Checkpoints: checkpoints/")
    print("\nNext steps:")
    print("  1. Review cluster analysis in analysis_{region_name}/")
    print("  2. Compare features across different time periods")
    print("  3. Track environmental changes over time")


def main():
    parser = argparse.ArgumentParser(description='Automated ML workflow for satellite analysis')
    parser.add_argument('--region', type=str, required=True,
                        help='Region name (assam, mangalore, amazon, congo)')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date YYYY-MM-DD')
    parser.add_argument('--cloud-threshold', type=float, default=None,
                        help='Maximum cloud coverage percentage')
    parser.add_argument('--tile-size', type=int, default=128,
                        help='Tile size in pixels')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training step')
    
    args = parser.parse_args()
    
    run_workflow(
        region_name=args.region,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_threshold=args.cloud_threshold,
        tile_size=args.tile_size,
        train_epochs=args.epochs,
        skip_download=args.skip_download,
        skip_training=args.skip_training
    )


if __name__ == '__main__':
    main()
