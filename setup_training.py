"""
One-time MAE Model Training Script
Train once, use for inference forever

This script:
1. Downloads sample training data
2. Trains the MAE model
3. Saves the checkpoint
4. Ready for inference!
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_model_once(
    training_location: str = "Mangalore, India",
    condition: str = "vegetation",
    epochs: int = 200,
    batch_size: int = 32,
    img_size: int = 128,
    device: str = "cuda"
):
    """
    Train MAE model once
    
    Args:
        training_location: Location to download training data from
        condition: Type of imagery for training
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        img_size: Image size
        device: Training device
    """
    
    print("=" * 80)
    print("ONE-TIME MAE MODEL TRAINING")
    print("=" * 80)
    print(f"Training location: {training_location}")
    print(f"Condition: {condition}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)
    
    # Check if model already exists
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "mae_encoder.pth"
    if checkpoint_path.exists():
        print(f"\n⚠️  Model already exists at: {checkpoint_path}")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training. Using existing model.")
            return str(checkpoint_path)
    
    # Step 1: Download training data
    print("\n[PHASE 1] Downloading training data...")
    print("-" * 80)
    
    train_data_dir = PROJECT_ROOT / "data" / "training_data"
    
    download_cmd = (
        f"python preprocessing/download_sentinel_api.py "
        f"--place \"{training_location}\" "
        f"--condition {condition} "
        f"--start 2023-01-01 "
        f"--end 2024-01-01 "
        f"--limit 100 "
        f"--output-dir {train_data_dir}"
    )
    print(f"Downloading: {download_cmd}")
    os.system(download_cmd)
    
    # Step 2: Preprocess training data
    print("\n[PHASE 2] Preprocessing training data...")
    print("-" * 80)
    
    filtered_dir = PROJECT_ROOT / "data" / "training_filtered"
    tiles_dir = PROJECT_ROOT / "data" / "training_tiles"
    normalized_dir = PROJECT_ROOT / "data" / "tiles_normalized"
    
    # Cloud filtering
    cloud_filter_cmd = (
        f"python preprocessing/cloud_filter.py "
        f"--input {train_data_dir} "
        f"--output {filtered_dir} "
        f"--max-cloud 30.0"
    )
    print("Cloud filtering...")
    os.system(cloud_filter_cmd)
    
    # Tile generation
    tile_gen_cmd = (
        f"python preprocessing/tile_generator.py "
        f"--input {filtered_dir} "
        f"--output {tiles_dir} "
        f"--tile-size {img_size} "
        f"--train-ratio 0.8 "
        f"--val-ratio 0.2"
    )
    print("Generating tiles...")
    os.system(tile_gen_cmd)
    
    # Normalization
    normalize_cmd = (
        f"python preprocessing/normalize.py "
        f"--input {tiles_dir}/train "
        f"--output {normalized_dir} "
        f"--method sentinel2"
    )
    print("Normalizing...")
    os.system(normalize_cmd)
    
    # Step 3: Train MAE model
    print("\n[PHASE 3] Training MAE model...")
    print("-" * 80)
    print("This will take some time. Please wait...")
    
    train_cmd = (
        f"python training/train_mae.py "
        f"--train-dir {normalized_dir} "
        f"--val-dir {tiles_dir}/val "
        f"--batch-size {batch_size} "
        f"--epochs {epochs} "
        f"--lr 1e-4 "
        f"--img-size {img_size} "
        f"--device {device}"
    )
    print(f"Training: {train_cmd}")
    os.system(train_cmd)
    
    # Verify checkpoint was saved
    if checkpoint_path.exists():
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\nModel saved to: {checkpoint_path}")
        print(f"\nYou can now run inference on ANY location using:")
        print(f'  python inference/auto_inference.py --place "Your Location" --condition vegetation')
        print("=" * 80)
        return str(checkpoint_path)
    else:
        print("\n❌ ERROR: Training failed - checkpoint not saved")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='One-time MAE model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python setup_training.py
  
  # Train with custom location
  python setup_training.py --location "Amazon Rainforest" --condition vegetation
  
  # Train with smaller batch size (if GPU runs out of memory)
  python setup_training.py --batch-size 16
        """
    )
    
    parser.add_argument('--location', type=str, default="Mangalore, India",
                        help='Location to download training data from')
    parser.add_argument('--condition', type=str, default='vegetation',
                        choices=['vegetation', 'water', 'urban', 'land_degradation'],
                        help='Type of imagery for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (reduce if CUDA OOM)')
    parser.add_argument('--img-size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_model_once(
        training_location=args.location,
        condition=args.condition,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
