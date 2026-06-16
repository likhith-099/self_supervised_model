"""
Create a smaller training subset for faster experimentation
"""
import os
import shutil
import random
from pathlib import Path

def create_subset(source_dir, target_dir, num_images=30000, seed=42):
    """
    Create a subset of training data
    
    Args:
        source_dir: Original training data directory
        target_dir: New subset directory
        num_images: Number of images to select
        seed: Random seed for reproducibility
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    print(f"Scanning {source_dir}...")
    all_files = [f for f in os.listdir(source_path) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    print(f"Found {len(all_files)} images")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select random subset
    if len(all_files) <= num_images:
        print(f"Using all {len(all_files)} images (less than requested {num_images})")
        selected_files = all_files
    else:
        print(f"Randomly selecting {num_images} images...")
        selected_files = random.sample(all_files, num_images)
    
    # Copy files
    print(f"Copying {len(selected_files)} images to {target_dir}...")
    for i, filename in enumerate(selected_files):
        src = source_path / filename
        dst = target_path / filename
        shutil.copy2(src, dst)
        
        # Progress update
        if (i + 1) % 5000 == 0:
            print(f"  Copied {i+1}/{len(selected_files)} images...")
    
    print(f"\n✅ Done! Created subset with {len(selected_files)} images")
    print(f"📁 Location: {target_dir}")

if __name__ == '__main__':
    # Create 30k training subset
    create_subset(
        source_dir='data/tiles/train',
        target_dir='data/tiles/train_30k',
        num_images=30000,
        seed=42
    )
    
    # Optionally create smaller validation subset too
    print("\nCreating validation subset...")
    create_subset(
        source_dir='data/tiles/val',
        target_dir='data/tiles/val_5k',
        num_images=5000,
        seed=42
    )
