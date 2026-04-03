"""
Cloud filtering module for satellite imagery
Removes images with excessive cloud coverage
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


def calculate_cloud_coverage(image_path: str) -> float:
    """
    FAST cloud estimation using file size and basic metadata
    Avoids loading massive JP2 files
    
    Args:
        image_path: Path to satellite image
    
    Returns:
        Estimated cloud coverage percentage (0-100)
    """
    import os
    
    try:
        # Get file size as proxy for cloud content
        file_size = os.path.getsize(image_path)
        
        # Sentinel-2 TCI 10m files are typically 5-30 MB
        # Cloudier scenes compress better (smaller files)
        # Clear scenes are larger (more detail)
        
        if file_size > 25 * 1024 * 1024:  # >25 MB = mostly clear
            return 5.0
        elif file_size > 15 * 1024 * 1024:  # 15-25 MB = some clouds
            return 15.0
        elif file_size > 8 * 1024 * 1024:  # 8-15 MB = cloudy
            return 35.0
        else:  # <8 MB = very cloudy
            return 60.0
            
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return 0.0  # Assume clear if error


def filter_cloudy_images(
    input_dir: str,
    output_dir: str,
    max_cloud_coverage: float = 20.0
) -> Tuple[List[str], List[str]]:
    """
    Filter out images with cloud coverage above threshold
    
    Args:
        input_dir: Directory containing raw satellite images
        output_dir: Directory to save filtered images
        max_cloud_coverage: Maximum acceptable cloud coverage (%)
    
    Returns:
        Tuple of (kept_files, removed_files)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    kept_files = []
    removed_files = []
    
    # Look for TCI (True Color) or B08 (NIR) bands at 10m resolution
    for img_file in input_path.rglob('*R10m\*.jp2'):
        # Only process True Color or main visible bands
        if not any(band in img_file.name.upper() for band in ['TCI', '_B02_', '_B03_', '_B04_', '_B08_']):
            continue
        
        cloud_pct = calculate_cloud_coverage(str(img_file))
        
        if cloud_pct <= max_cloud_coverage:
            # Create symlink to avoid copying huge files
            dest = output_path / img_file.name
            try:
                dest.symlink_to(img_file)
            except:
                # If symlinks fail, skip this file
                print(f"  Skipping {img_file.name} (symlink error)")
                continue
            kept_files.append(str(dest))
            print(f"✓ Keeping {img_file.name} (cloud: {cloud_pct:.1f}%)")
        else:
            removed_files.append(str(img_file))
            print(f"✗ Removing {img_file.name} (cloud: {cloud_pct:.1f}%)")
    
    print(f"\nSummary:")
    print(f"  Kept: {len(kept_files)} images")
    print(f"  Removed: {len(removed_files)} images")
    
    return kept_files, removed_files


def main():
    parser = argparse.ArgumentParser(description='Filter cloudy satellite images')
    parser.add_argument('--input', type=str, default='data/raw/sentinel',
                        help='Input directory with raw images')
    parser.add_argument('--output', type=str, default='data/raw/filtered',
                        help='Output directory for filtered images')
    parser.add_argument('--max-cloud', type=float, default=20.0,
                        help='Maximum cloud coverage percentage')
    
    args = parser.parse_args()
    
    filter_cloudy_images(
        input_dir=args.input,
        output_dir=args.output,
        max_cloud_coverage=args.max_cloud
    )


if __name__ == '__main__':
    main()
