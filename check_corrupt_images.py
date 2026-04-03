"""
Find corrupt images in dataset
"""
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from training.dataset_loader import SatelliteTileDataset

def check_dataset(data_dir: str, img_size: int = 128):
    """Check all images in dataset for dimension issues"""
    print(f"Checking dataset: {data_dir}")
    
    dataset = SatelliteTileDataset(data_dir, img_size=img_size, augment=False)
    
    problematic_images = []
    
    for idx in range(len(dataset)):
        try:
            img_path = dataset.image_paths[idx]
            img_tensor, _ = dataset[idx]
            
            # Check dimensions
            if len(img_tensor.shape) != 3:
                print(f"❌ Image {idx}: {img_path.name} - Wrong dimensions: {img_tensor.shape}")
                problematic_images.append((idx, img_path, "Wrong dimensions"))
            elif img_tensor.shape[0] != 3:
                print(f"❌ Image {idx}: {img_path.name} - Wrong channels: {img_tensor.shape[0]}")
                problematic_images.append((idx, img_path, f"Wrong channels: {img_tensor.shape[0]}"))
            elif torch.isnan(img_tensor).any():
                print(f"❌ Image {idx}: {img_path.name} - Contains NaN values")
                problematic_images.append((idx, img_path, "NaN values"))
            elif torch.isinf(img_tensor).any():
                print(f"❌ Image {idx}: {img_path.name} - Contains Inf values")
                problematic_images.append((idx, img_path, "Inf values"))
                
        except Exception as e:
            print(f"❌ Image {idx}: {dataset.image_paths[idx].name} - Error: {str(e)}")
            problematic_images.append((idx, dataset.image_paths[idx], str(e)))
        
        if (idx + 1) % 1000 == 0:
            print(f"Checked {idx + 1}/{len(dataset)} images...")
    
    print(f"\n{'='*60}")
    print(f"Total images checked: {len(dataset)}")
    print(f"Problematic images found: {len(problematic_images)}")
    
    if problematic_images:
        print(f"\nProblematic images:")
        for idx, path, reason in problematic_images:
            print(f"  - Index {idx}: {path.name} ({reason})")
    else:
        print("✅ All images are valid!")
    
    return problematic_images

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find corrupt images')
    parser.add_argument('--data-dir', type=str, default='data/tiles_normalized',
                        help='Dataset directory to check')
    parser.add_argument('--img-size', type=int, default=128,
                        help='Image size')
    
    args = parser.parse_args()
    
    check_dataset(args.data_dir, args.img_size)
