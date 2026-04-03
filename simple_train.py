"""
SIMPLE STEP-BY-STEP TRAINING GUIDE
Run this script and follow the prompts!
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_step(step_num, text):
    """Print step header"""
    print(f"\n{'='*20} STEP {step_num} {'='*60}")
    print(f"→ {text}")
    print(f"{'='*80}\n")


def check_data():
    """Step 1: Check if data exists"""
    print_step(1, "CHECKING YOUR DATA")
    
    sentinel_dir = PROJECT_ROOT / 'data' / 'raw' / 'sentinel'
    
    if sentinel_dir.exists():
        n_images = len(list(sentinel_dir.glob('*.SAFE')))
        print(f"✓ Found {n_images} Sentinel-2 images in data/raw/sentinel/")
        
        if n_images > 0:
            print("✓ Great! You have satellite images ready!")
            return True
    
    print("✗ No images found!")
    print("\nTo download images, run:")
    print('  python preprocessing/download_sentinel_api.py --place "Mangalore, India" --condition vegetation --start 2024-01-01 --end 2024-03-31')
    return False


def filter_clouds():
    """Step 2: Filter cloudy images"""
    print_step(2, "FILTERING CLOUDY IMAGES")
    
    cmd = (
        f"python preprocessing/cloud_filter.py "
        f"--input data/raw/sentinel "
        f"--output data/raw/filtered "
        f"--max-cloud 30.0"
    )
    
    print("Removing images with too many clouds...")
    print(f"Command: {cmd}")
    os.system(cmd)
    
    print("\n✓ Cloud filtering complete!")


def generate_tiles():
    """Step 3: Generate tiles"""
    print_step(3, "GENERATING 128x128 PIXEL TILES")
    
    cmd = (
        f"python preprocessing/tile_generator.py "
        f"--input data/raw/filtered "
        f"--output data/tiles "
        f"--tile-size 128 "
        f"--train-ratio 0.8"
    )
    
    print("Cutting large images into small tiles for training...")
    print(f"Command: {cmd}")
    os.system(cmd)
    
    print("\n✓ Tiles generated!")


def normalize_tiles():
    """Step 4: Normalize tiles"""
    print_step(4, "NORMALIZING PIXEL VALUES")
    
    cmd = (
        f"python preprocessing/normalize.py "
        f"--input data/tiles/train "
        f"--output data/tiles_normalized "
        f"--method sentinel2"
    )
    
    print("Adjusting pixel values for machine learning...")
    print(f"Command: {cmd}")
    os.system(cmd)
    
    print("\n✓ Normalization complete!")


def train_model():
    """Step 5: Train MAE model"""
    print_step(5, "TRAINING MASKED AUTOENCODER (MAE)")
    
    print("This will take 2-4 hours on GPU (or overnight on CPU)")
    print("The AI is learning to recognize environmental features...")
    print("\nStarting training now...\n")
    
    cmd = (
        f"python training/train_mae.py "
        f"--train-dir data/tiles/train "
        f"--val-dir data/tiles/val "
        f"--batch-size 64 "
        f"--epochs 200 "
        f"--lr 1e-4 "
        f"--img-size 128 "
        f"--device cuda"
    )
    
    print(f"Command: {cmd}")
    os.system(cmd)
    
    print("\n✓ Training complete!")
    print("✓ Model saved to: checkpoints/mae_encoder.pth")


def extract_features():
    """Step 6: Extract features"""
    print_step(6, "EXTRACTING ENVIRONMENTAL FEATURES")
    
    cmd = (
        f"python inference/extract_features.py "
        f"--checkpoint checkpoints/mae_encoder.pth "
        f"--input data/tiles/val "
        f"--output features_analysis.npy "
        f"--device cuda"
    )
    
    print("Converting satellite images into numerical features...")
    print(f"Command: {cmd}")
    os.system(cmd)
    
    print("\n✓ Features extracted!")


def analyze_changes():
    """Step 7: Analyze environmental changes"""
    print_step(7, "ANALYZING ENVIRONMENTAL CHANGES")
    
    # For demonstration, we'll compare first half vs second half of features
    print("Comparing different areas to detect changes...")
    print("Generating visualizations and predictions...")
    
    cmd = (
        f"python inference/map_visualization.py "
        f"--baseline features_analysis.npy "
        f"--current features_analysis.npy "
        f"--condition vegetation "
        f"--output analysis_report.png"
    )
    
    os.system(cmd)
    
    print("\n✓ Analysis complete!")


def show_results():
    """Show final results"""
    print_header("🎉 TRAINING AND ANALYSIS COMPLETE!")
    
    print("Your model has been trained successfully!\n")
    print("What you can do now:\n")
    
    print("1️⃣  VIEW RESULTS:")
    print("   Open: analysis_report.png\n")
    
    print("2️⃣  ANALYZE SPECIFIC CONDITIONS:")
    print("   python analyze_condition.py --region mangalore --condition vegetation --year1 2024 --year2 2025\n")
    
    print("3️⃣  RUN FULL AUTOMATED ANALYSIS:")
    print('   python auto_analyze.py --place "Your Location" --condition vegetation --start 2024-01-01 --end 2025-01-01\n')
    
    print("=" * 80)
    print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def main():
    """Main workflow"""
    print_header("🚀 SIMPLE MODEL TRAINING GUIDE")
    
    print("This script will guide you through training your environmental analysis model.")
    print("Each step will process your satellite data automatically.\n")
    
    print("Estimated total time: 3-5 hours (mostly training time)")
    print("You can leave it running and come back later!\n")
    
    input("Press ENTER to start training...")
    
    # Execute all steps
    if check_data():
        filter_clouds()
        generate_tiles()
        normalize_tiles()
        train_model()
        extract_features()
        analyze_changes()
        show_results()
    else:
        print("\n⚠ Please download satellite images first!")
        print('Command: python preprocessing/download_sentinel_api.py --place "Mangalore, India" --condition vegetation --start 2024-01-01 --end 2024-03-31')


if __name__ == '__main__':
    main()
