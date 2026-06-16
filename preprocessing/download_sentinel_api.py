"""
Automated Sentinel-2 Image Downloader
Converts place names to coordinates and downloads satellite imagery
"""

import requests
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geocoder import Geocoder

# Copernicus STAC search endpoint
SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"

# Your access token for Copernicus API
ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJYVUh3VWZKaHVDVWo0X3k4ZF8xM0hxWXBYMFdwdDd2anhob2FPLUxzREZFIn0.eyJleHAiOjE3NzU5MzQwMjEsImlhdCI6MTc3NTkzMjIyMSwianRpIjoib25ydHJvOjlmNTFhZjQwLWRmZTgtYjlhZS1jZjNlLWFhMTQ4YjhjNmRlMyIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkuZGF0YXNwYWNlLmNvcGVybmljdXMuZXUvYXV0aC9yZWFsbXMvQ0RTRSIsImF1ZCI6WyJDTE9VREZFUlJPX1BVQkxJQyIsImFjY291bnQiXSwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiY2RzZS1wdWJsaWMiLCJzaWQiOiI4MWUxZGIyMC1lNDVlLTJiOTgtODBiOS0yNDE5NWRiMjA5NDMiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9sb2NhbGhvc3Q6NDIwMCIsIioiLCJodHRwczovL3dvcmtzcGFjZS5zdGFnaW5nLWNkc2UtZGF0YS1leHBsb3Jlci5hcHBzLnN0YWdpbmcuaW50cmEuY2xvdWRmZXJyby5jb20iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNvcGVybmljdXMtZ2VuZXJhbC1xdW90YSIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLWNkYXMiLCJjb3Blcm5pY3VzLWdlbmVyYWwiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6IkFVRElFTkNFX1BVQkxJQyBvcGVuaWQgZW1haWwgcHJvZmlsZSBvbmRlbWFuZF9wcm9jZXNzaW5nIHVzZXItY29udGV4dCIsImdyb3VwX21lbWJlcnNoaXAiOlsiL2FjY2Vzc19ncm91cHMvdXNlcl90eXBvbG9neS9jb3Blcm5pY3VzX2dlbmVyYWwiLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIiXSwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJTbmVoYSBTaGV0dHkiLCJvcmdhbml6YXRpb25zIjpbImRlZmF1bHQtNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIl0sInVzZXJfY29udGV4dF9pZCI6Ijc3N2NhMDhjLWY0OTUtNDM5Ni04ODkzLTNiZmI3YjU0N2JlNiIsImNvbnRleHRfcm9sZXMiOnt9LCJjb250ZXh0X2dyb3VwcyI6WyIvYWNjZXNzX2dyb3Vwcy91c2VyX3R5cG9sb2d5L2NvcGVybmljdXNfZ2VuZXJhbC8iLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIvIl0sInByZWZlcnJlZF91c2VybmFtZSI6InNuZWhhc2hldHR5LjE4MDVAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNuZWhhIiwiZmFtaWx5X25hbWUiOiJTaGV0dHkiLCJ1c2VyX2NvbnRleHQiOiJkZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4ZiIsImVtYWlsIjoic25laGFzaGV0dHkuMTgwNUBnbWFpbC5jb20ifQ.mW_A2GGI0cNr6n7uskKTR_yugxj_YILmclSnzNJ6oeA1BcnO4nURIwV2EcmfWdifx3P3LRYDugrKRYYHJGwWpAKYhHezJon8zD4nPExncH3-WQ2O2q6A_MK--bwU6NQa-kHk-fDfKOvSwINer9nkOFcvMnvapsDiWgIitHHOZfxpGhQJMGVAAyLPTGlB3d_rE9wyGErirV5V7cBI_nMgIxzI8Zi6WonAm_BGTkjhdRhutsTwdhioQNBpdosr7Y3tlzupbvYNLwU0xWQIfoaV96paMj4A_9DCAeX0ul5HszPoPFBZPNDrpGbgdFol5eFSXQx4NDEFOmBItxHYYsUyug"

# Initialize geocoder
geocoder = Geocoder()


def search_satellite_images(bbox: list, date_range: str, limit: int = 20) -> dict:
    """Search Sentinel-2 images using Copernicus STAC API
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        date_range: Date range string "YYYY-MM-DD/YYYY-MM-DD"
        limit: Number of images to fetch
    
    Returns:
        Search results dictionary
    """

    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "limit": limit
    }

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(SEARCH_URL, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()


def download_images(results: dict, save_dir: str):
    """Download preview images from the API results
    
    Args:
        results: Search results from API
        save_dir: Directory to save images
    """

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    for i, feature in enumerate(results["features"]):

        assets = feature["assets"]

        img_url = None

        # Prefer preview image
        if "preview" in assets:
            img_url = assets["preview"]["href"]

        elif "thumbnail" in assets:
            img_url = assets["thumbnail"]["href"]

        if img_url is None:
            print("Skipping item, no preview found")
            continue

        print(f"Downloading image {i+1}/{len(results['features'])}...")

        img_data = requests.get(img_url, headers=headers).content

        file_path = os.path.join(save_dir, f"image_{i:03d}.jpg")

        with open(file_path, "wb") as f:
            f.write(img_data)
        
        print(f"  ✓ Saved: {file_path}")


def main(place_name: str, condition: str, start_date: str, end_date: str, 
         limit: int = 20, save_dir: str = None):
    """
    Complete automated download workflow
    
    Args:
        place_name: Location name (e.g., "Mangalore, India")
        condition: Environmental condition to study (vegetation, water, urban, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Number of images to download
        save_dir: Directory to save images
    """
    print("=" * 80)
    print("AUTOMATED SATELLITE IMAGE DOWNLOAD")
    print("=" * 80)
    print(f"Location: {place_name}")
    print(f"Condition to Study: {condition.upper()}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Number of Images: {limit}")
    print("=" * 80)
    
    # Step 1: Convert place name to coordinates
    print("\n[STEP 1/3] Converting location to coordinates...")
    print("-" * 80)
    location_info = geocoder.get_region_coordinates(place_name)
    
    if not location_info:
        print(f"ERROR: Could not find location '{place_name}'")
        return
    
    bbox = location_info['bbox']
    print(f"✓ Location found: {location_info['display_name']}")
    print(f"✓ Bounding Box: {bbox}")
    print()
    
    # Step 2: Search for satellite images
    print("[STEP 2/3] Searching for Sentinel-2 satellite images...")
    print("-" * 80)
    # Fix datetime format: use T separator
    date_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    results = search_satellite_images(bbox, date_range, limit)
    
    n_found = len(results.get('features', []))
    print(f"✓ Found {n_found} satellite images")
    print()
    
    # Step 3: Download images
    print("[STEP 3/3] Downloading images...")
    print("-" * 80)
    
    if save_dir is None:
        save_dir = str(Path(__file__).parent.parent / 'data' / 'raw' / 'sentinel')
    
    os.makedirs(save_dir, exist_ok=True)
    download_images(results, save_dir)
    
    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Images saved to: {save_dir}")
    print(f"Total images downloaded: {n_found}")
    print(f"\nNext step: Run preprocessing and analysis")
    print("  python run_full_workflow.py --region custom --start-date {} --end-date {}".format(
        start_date, end_date))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Sentinel-2 Image Downloader')
    parser.add_argument('--place', type=str, required=True,
                        help='Place name (e.g., "Mangalore, India")')
    parser.add_argument('--condition', type=str, required=True,
                        help='Environmental condition to study')
    parser.add_argument('--start', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=20,
                        help='Number of images to download')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for images')
    
    args = parser.parse_args()
    
    main(
        place_name=args.place,
        condition=args.condition,
        start_date=args.start,
        end_date=args.end,
        limit=args.limit,
        save_dir=args.output
    )