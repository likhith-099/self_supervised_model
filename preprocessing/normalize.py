"""
Pixel normalization for satellite imagery
Applies various normalization techniques to prepare data for ML models
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image


class Normalizer:
    """Handles pixel normalization for satellite imagery"""
    
    def __init__(
        self,
        method: str = 'sentinel2',
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ('sentinel2', 'minmax', 'zscore', 'custom')
            stats: Pre-computed (mean, std) for z-score normalization
        """
        self.method = method
        self.stats = stats
        
        # Sentinel-2 Level-2A specific normalization
        if method == 'sentinel2':
            # Scale factors for Sentinel-2 surface reflectance bands
            # B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
            self.scale_factors = np.array([10000.0, 10000.0, 10000.0, 10000.0])
            self.min_values = np.array([0.0, 0.0, 0.0, 0.0])
            self.max_values = np.array([3000.0, 3000.0, 4000.0, 4000.0])  # Typical max values
    
    def compute_stats(self, image_paths: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std from a set of images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            Tuple of (mean, std) arrays
        """
        all_pixels = []
        
        for img_path in image_paths[:100]:  # Sample first 100 images
            img = np.array(Image.open(img_path))
            all_pixels.append(img.flatten())
        
        all_pixels = np.concatenate(all_pixels)
        mean = np.mean(all_pixels)
        std = np.std(all_pixels)
        
        self.stats = (mean, std)
        return self.stats
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
        
        Returns:
            Normalized image
        """
        if self.method == 'sentinel2':
            # Sentinel-2 specific normalization
            # Scale by 10000 and clip to valid range
            img_norm = image.astype(np.float32) / 10000.0
            img_norm = np.clip(img_norm, 0.0, 1.0)
            return img_norm
        
        elif self.method == 'minmax':
            return (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        elif self.method == 'zscore':
            if self.stats is None:
                raise ValueError("Stats not computed. Call compute_stats() first.")
            mean, std = self.stats
            return (image - mean) / (std + 1e-8)
        
        elif self.method == 'custom':
            # Custom normalization (e.g., based on sensor specifications)
            return image / 10000.0  # For Sentinel-2 surface reflectance
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


def normalize_images(
    input_dir: str,
    output_dir: str,
    method: str = 'zscore',
    compute_stats_from: str = None
) -> None:
    """
    Normalize all images in a directory
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for normalized images
        method: Normalization method
        compute_stats_from: Directory to compute statistics from
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize normalizer
    normalizer = Normalizer(method=method)
    
    # Compute statistics if needed
    if method == 'zscore' and compute_stats_from:
        print(f"Computing statistics from {compute_stats_from}...")
        stats_images = list(Path(compute_stats_from).rglob('*.png'))
        normalizer.compute_stats([str(p) for p in stats_images])
        print(f"Mean: {normalizer.stats[0]:.4f}, Std: {normalizer.stats[1]:.4f}")
    
    # Normalize images
    normalized_count = 0
    for img_file in input_path.rglob('*.png'):
        img = np.array(Image.open(img_file)).astype(np.float32)
        img_norm = normalizer.normalize(img)
        
        # Save normalized image
        output_file = output_path / img_file.name
        Image.fromarray((img_norm * 255).astype(np.uint8)).save(output_file)
        normalized_count += 1
    
    print(f"\nNormalized {normalized_count} images")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Normalize satellite imagery')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with images')
    parser.add_argument('--output', type=str, default='data/tiles_normalized',
                        help='Output directory for normalized images')
    parser.add_argument('--method', type=str, default='zscore',
                        choices=['minmax', 'zscore', 'custom'],
                        help='Normalization method')
    parser.add_argument('--stats-from', type=str, default=None,
                        help='Directory to compute statistics from')
    
    args = parser.parse_args()
    
    normalize_images(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        compute_stats_from=args.stats_from
    )


if __name__ == '__main__':
    main()
