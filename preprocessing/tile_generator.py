"""
Tile generator for satellite imagery
Splits large satellite images into smaller tiles for training
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


def generate_tiles(
    image_path: str,
    output_dir: str,
    tile_size: int = 256,
    stride: int = None,
    overlap: float = 0.0
) -> List[str]:
    """
    Split a large image into smaller tiles
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save tiles
        tile_size: Size of each tile (pixels)
        stride: Step size between tiles (default: tile_size)
        overlap: Overlap ratio between tiles (0.0-1.0)
    
    Returns:
        List of generated tile paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    
    if stride is None:
        stride = tile_size
    elif overlap > 0:
        stride = int(tile_size * (1 - overlap))
    
    tiles = []
    tile_count = 0
    
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract tile
            box = (x, y, x + tile_size, y + tile_size)
            tile = img.crop(box)
            
            # Save tile
            tile_filename = f"{Path(image_path).stem}_tile_{tile_count:04d}.png"
            tile_path = output_path / tile_filename
            tile.save(tile_path)
            
            tiles.append(str(tile_path))
            tile_count += 1
    
    print(f"Generated {tile_count} tiles from {image_path}")
    print(f"Saved to: {output_dir}")
    
    return tiles


def generate_train_val_split(
    input_dir: str,
    output_base: str,
    train_ratio: float = 0.8,
    tile_size: int = 256
) -> Tuple[List[str], List[str]]:
    """
    Generate tiles and split into training and validation sets
    
    Args:
        input_dir: Directory containing input images
        output_base: Base directory for output (will create train/ and val/ subdirs)
        train_ratio: Ratio of training tiles (0.0-1.0)
        tile_size: Size of tiles
    
    Returns:
        Tuple of (train_tiles, val_tiles)
    """
    input_path = Path(input_dir)
    train_dir = Path(output_base) / 'train'
    val_dir = Path(output_base) / 'val'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    all_tiles = []
    
    # Generate tiles for all images (support both TIF and JP2)
    for img_file in input_path.rglob('*R10m\*.jp2'):
        if any(band in img_file.name.upper() for band in ['TCI', '_B02_', '_B03_', '_B04_', '_B08_']):
            tiles = generate_tiles(
                str(img_file),
                output_dir=str(train_dir),  # Temporarily save all to train
                tile_size=tile_size
            )
            all_tiles.extend(tiles)
    
    # Split into train/val
    np.random.shuffle(all_tiles)
    split_idx = int(len(all_tiles) * train_ratio)
    
    train_tiles = all_tiles[:split_idx]
    val_tiles = all_tiles[split_idx:]
    
    # Move validation tiles
    for tile_path in val_tiles:
        src = Path(tile_path)
        dst = val_dir / src.name
        src.rename(dst)
    
    print(f"\nSplit summary:")
    print(f"  Training tiles: {len(train_tiles)}")
    print(f"  Validation tiles: {len(val_tiles)}")
    
    return train_tiles, val_tiles


def main():
    parser = argparse.ArgumentParser(description='Generate tiles from satellite images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with satellite images')
    parser.add_argument('--output', type=str, default='data/tiles',
                        help='Output directory for tiles')
    parser.add_argument('--tile-size', type=int, default=256,
                        help='Tile size in pixels')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Overlap ratio between tiles')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training/validation split ratio')
    
    args = parser.parse_args()
    
    generate_train_val_split(
        input_dir=args.input,
        output_base=args.output,
        train_ratio=args.train_ratio,
        tile_size=args.tile_size
    )


if __name__ == '__main__':
    main()
