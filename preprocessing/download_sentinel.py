"""
Sentinel satellite image downloader
Downloads satellite scenes from Copernicus Open Science Hub
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional


def download_sentinel_images(
    output_dir: str,
    bounding_box: tuple,
    date_range: tuple,
    cloud_coverage: float = 20.0,
    limit: int = 100
) -> List[str]:
    """
    Download Sentinel-2 images using Copernicus Open Science Hub API
    
    Args:
        output_dir: Directory to save downloaded images
        bounding_box: (min_lon, min_lat, max_lon, max_lat)
        date_range: (start_date, end_date) in 'YYYY-MM-DD' format
        cloud_coverage: Maximum cloud coverage percentage
        limit: Maximum number of images to download
    
    Returns:
        List of downloaded file paths
    """
    try:
        from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    except ImportError:
        raise ImportError("Please install sentinelsat: pip install sentinelsat")
    
    # API credentials - Configure these in utils/config.py or environment variables
    api_user = os.getenv('COPERNICUS_USER', 'your_username_here')
    api_password = os.getenv('COPERNICUS_PASSWORD', 'your_password_here')
    
    if api_user == 'your_username_here':
        print("ERROR: Copernicus API credentials not configured!")
        print("Please set environment variables:")
        print("  COPERNICUS_USER=your_username")
        print("  COPERNICUS_PASSWORD=your_password")
        print("\nOr edit utils/config.py and add your credentials.")
        return []
    
    # Connect to API
    api = SentinelAPI(api_user, api_password, 'https://apihub.copernicus.eu/apihub')
    
    # Convert bounding box to WKT format
    wkt = f"POLYGON(({bounding_box[0]} {bounding_box[1]}, " \
          f"{bounding_box[2]} {bounding_box[1]}, " \
          f"{bounding_box[2]} {bounding_box[3]}, " \
          f"{bounding_box[0]} {bounding_box[3]}, " \
          f"{bounding_box[0]} {bounding_box[1]}))"
    
    # Query products
    products = api.query(
        wkt,
        date=date_range,
        platformname='Sentinel-2',
        producttype='S2MSI2A',  # Level-2A surface reflectance
        cloudcoverpercentage=(0, cloud_coverage)
    )
    
    print(f"Found {len(products)} products matching criteria")
    
    # Download products
    downloaded_files = []
    for uuid, product_info in list(products.items())[:limit]:
        print(f"Downloading: {product_info['title']}")
        filepath = api.download(uuid, directory_path=output_dir)
        downloaded_files.append(filepath)
        print(f"  Saved to: {filepath}")
    
    api.logout()
    print(f"\nDownloaded {len(downloaded_files)} images")
    
    return downloaded_files


def main():
    parser = argparse.ArgumentParser(description='Download Sentinel satellite images')
    parser.add_argument('--output', type=str, default='data/raw/sentinel',
                        help='Output directory for downloaded images')
    parser.add_argument('--bbox', type=float, nargs=4, required=True,
                        help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--cloud-coverage', type=float, default=20.0,
                        help='Maximum cloud coverage percentage')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of images to download')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bounding_box = tuple(args.bbox)
    date_range = (args.start_date, args.end_date)
    
    downloaded = download_sentinel_images(
        output_dir=str(output_path),
        bounding_box=bounding_box,
        date_range=date_range,
        cloud_coverage=args.cloud_coverage,
        limit=args.limit
    )
    
    print(f"\nDownloaded {len(downloaded)} images")


if __name__ == '__main__':
    main()
