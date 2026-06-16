"""
Universal Environmental Analysis Pipeline
Use this for ANY location, ANY condition, ANY time period!

Usage:
python universal_pipeline.py --location "Karwar, India" --condition water --start-year 2005 --end-year 2010
python universal_pipeline.py --location "Mumbai, India" --condition vegetation --start-year 2015 --end-year 2020
python universal_pipeline.py --location "Goa, India" --condition climate --start-year 2018 --end-year 2023
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.copernicus_client import CopernicusClient
from utils.geocoder import Geocoder
from utils.visual_report import create_interactive_report
from models.encoder import EncoderLoader
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Copernicus API tokens
ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJYVUh3VWZKaHVDVWo0X3k4ZF8xM0hxWXBYMFdwdDd2anhob2FPLUxzREZFIn0.eyJleHAiOjE3NzcwOTMzMjksImlhdCI6MTc3NzA5MTUyOSwianRpIjoib25ydHJvOjQzMWM5NTY0LWFjNzItNTExNy1lYzA4LTdjNjJiMDVlNjM3YiIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkuZGF0YXNwYWNlLmNvcGVybmljdXMuZXUvYXV0aC9yZWFsbXMvQ0RTRSIsImF1ZCI6WyJDTE9VREZFUlJPX1BVQkxJQyIsImFjY291bnQiXSwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiY2RzZS1wdWJsaWMiLCJzaWQiOiI5MzgzOWY2Zi1iNDk5LWIwYjYtYjBkNS0zOGY4YjI0ZDAzNjMiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9sb2NhbGhvc3Q6NDIwMCIsIioiLCJodHRwczovL3dvcmtzcGFjZS5zdGFnaW5nLWNkc2UtZGF0YS1leHBsb3Jlci5hcHBzLnN0YWdpbmcuaW50cmEuY2xvdWRmZXJyby5jb20iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNvcGVybmljdXMtZ2VuZXJhbC1xdW90YSIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLWNkYXMiLCJjb3Blcm5pY3VzLWdlbmVyYWwiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6IkFVRElFTkNFX1BVQkxJQyBvcGVuaWQgZW1haWwgcHJvZmlsZSBvbmRlbWFuZF9wcm9jZXNzaW5nIHVzZXItY29udGV4dCIsImdyb3VwX21lbWJlcnNoaXAiOlsiL2FjY2Vzc19ncm91cHMvdXNlcl90eXBvbG9neS9jb3Blcm5pY3VzX2dlbmVyYWwiLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIiXSwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJTbmVoYSBTaGV0dHkiLCJvcmdhbml6YXRpb25zIjpbImRlZmF1bHQtNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIl0sInVzZXJfY29udGV4dF9pZCI6Ijc3N2NhMDhjLWY0OTUtNDM5Ni04ODkzLTNiZmI3YjU0N2JlNiIsImNvbnRleHRfcm9sZXMiOnt9LCJjb250ZXh0X2dyb3VwcyI6WyIvYWNjZXNzX2dyb3Vwcy91c2VyX3R5cG9sb2d5L2NvcGVybmljdXNfZ2VuZXJhbC8iLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIvIl0sInByZWZlcnJlZF91c2VybmFtZSI6InNuZWhhc2hldHR5LjE4MDVAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNuZWhhIiwiZmFtaWx5X25hbWUiOiJTaGV0dHkiLCJ1c2VyX2NvbnRleHQiOiJkZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4ZiIsImVtYWlsIjoic25laGFzaGV0dHkuMTgwNUBnbWFpbC5jb20ifQ.lFBEkBkkHrwmUPrTrJRJFvstpVaD_cKxu-78LTkafenzcnAG4-gU3gK2S9EbA6sTuwoefC6lg_uKCyAIZo1EhSeHiqUeW4ON_LflyWTSLhUa4yDh7joux3I53zxdiZOE-x5P20taxc4XMjn0nDJjOhNAzXKiFUfB6hrx8jUmRKkDfh4r3plmGi12NaG56OfsJ94K-hu1ZyPmA9jvkNJeK-C-j81g9rnDijBVXhhsuhf0bBmcD9VjUvERRXToNvrSjcw7AjcOVWWbmEbzImmIuRYM63k8C_CxjnYIXKCXNGgKoZVazAi703vAFDATEPzXpb4a6rdRYiXkYLp8TBJf0Q"
REFRESH_TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJjODg0ZDhkYi0wNjY2LTQ5OWYtOTJkYi1hNTQwODZmMDQ5YjkifQ.eyJleHAiOjE3NzcwOTUxMjksImlhdCI6MTc3NzA5MTUyOSwianRpIjoiNjg2Nzg3OWItNjJhOS0zZDc0LTk3MTgtMDEyMjQ5ZjkyZGVhIiwiaXNzIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwiYXVkIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiUmVmcmVzaCIsImF6cCI6ImNkc2UtcHVibGljIiwic2lkIjoiOTM4MzlmNmYtYjQ5OS1iMGI2LWIwZDUtMzhmOGIyNGQwMzYzIiwic2NvcGUiOiJBVURJRU5DRV9QVUJMSUMgYmFzaWMgd2ViLW9yaWdpbnMgb3BlbmlkIGVtYWlsIHJvbGVzIHByb2ZpbGUgb25kZW1hbmRfcHJvY2Vzc2luZyB1c2VyLWNvbnRleHQifQ.cCGSFgVfqkEUhQY2e9f9dY2M38sNzj0cPbImtPzFMM8O8049WV_YguBHYr4MWRRjlRWKHU_L6t4uFsRTypQj-g"

def get_bbox_from_location(place_name):
    """Get approximate bounding box for a location using geocoding"""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="universal_pipeline")
        location = geolocator.geocode(place_name, timeout=10)
        if location:
            # Create a small bounding box around the location (approx 10km radius)
            lat, lon = location.latitude, location.longitude
            bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
            print(f"   📍 Geocoded: {place_name} -> ({lat:.4f}, {lon:.4f})")
            print(f"   📐 BBox: {bbox}")
            return bbox
    except Exception as e:
        print(f"   ⚠️ Geocoding failed: {e}")
    
    # Default fallback (you can add more locations here)
    print(f"   ⚠️ Using default bbox, please verify coordinates")
    return [74.0, 12.0, 75.0, 13.0]  # Default: Karnataka region

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Universal Environmental Analysis Pipeline')
    parser.add_argument('--location', type=str, required=True,
                        help='Location name (e.g., "Karwar, India")')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['water', 'vegetation', 'climate', 'urban', 'land_degradation'],
                        help='Analysis condition: water, vegetation, climate, urban, or land_degradation')
    parser.add_argument('--start-year', type=int, required=True,
                        help='Start year (e.g., 2005)')
    parser.add_argument('--end-year', type=int, required=True,
                        help='End year (e.g., 2010)')
    parser.add_argument('--bbox', type=float, nargs=4,
                        help='Custom bounding box: lon_min lat_min lon_max lat_max (optional)')
    
    args = parser.parse_args()
    
    # Get bounding box
    if args.bbox:
        bbox = args.bbox
        print(f"📐 Using custom bbox: {bbox}")
    else:
        bbox = get_bbox_from_location(args.location)
    
    print("=" * 80)
    print(f"🌍 UNIVERSAL ENVIRONMENTAL ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"📍 Location: {args.location}")
    print(f"📅 Period: {args.start_year} - {args.end_year}")
    print(f"🔍 Condition: {args.condition.title()}")
    print(f"📐 BBox: {bbox}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Configuration
    location_slug = args.location.lower().replace(' ', '_').replace(',', '')
    output_dir = Path(f"analysis_output/{location_slug}_{args.condition}_{args.start_year}_{args.end_year}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    years = list(range(args.start_year, args.end_year + 1))
    feature_files = {}
    
    # PHASE 1: Initialize API
    print("\n" + "=" * 80)
    print("PHASE 1: Initializing Copernicus API Client")
    print("=" * 80)
    client = CopernicusClient(ACCESS_TOKEN, REFRESH_TOKEN)
    print("✅ API client initialized successfully")

    # PHASE 2: Download satellite data
    print("\n" + "=" * 80)
    print("PHASE 2: Downloading Satellite Data")
    print("=" * 80)
    
    for year in years:
        print(f"\n{'─' * 60}")
        print(f"📅 Processing Year: {year}")
        print(f"{'─' * 60}")
        
        year_dir = output_dir / f"raw_{year}"
        year_dir.mkdir(exist_ok=True)
        
        existing_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
        
        if len(existing_images) >= 5:
            print(f"  ✓ Using {len(existing_images)} existing images from {year}")
        else:
            print(f"  📥 Downloading satellite images for {year}...")
            try:
                results = client.search_images(
                    bbox=bbox,
                    start_date=f"{year}-01-01",
                    end_date=f"{year}-12-31",
                    limit=8
                )
                
                if results and len(results.get('features', [])) > 0:
                    downloaded = client.download_images(results, str(year_dir))
                    print(f"  ✓ Downloaded {downloaded} images for {year}")
                else:
                    print(f"  ❌ No images available for {year}")
                    continue
            except Exception as e:
                print(f"  ❌ Download failed for {year}: {e}")
                continue
        
        all_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
        if len(all_images) == 0:
            print(f"  ⚠️ No images to process for {year}")
            continue
            
        print(f"  📊 Total images for {year}: {len(all_images)}")

    # PHASE 3: Preprocess images
    print("\n" + "=" * 80)
    print("PHASE 3: Preprocessing - Tiling and Normalization")
    print("=" * 80)
    
    tile_size = 128
    stride = 128
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for year in years:
        print(f"\n{'─' * 60}")
        print(f"🔄 Preprocessing Year: {year}")
        print(f"{'─' * 60}")
        
        year_dir = output_dir / f"raw_{year}"
        tiles_dir = output_dir / f"tiles_{year}"
        normalized_dir = output_dir / f"normalized_{year}"
        
        tiles_dir.mkdir(exist_ok=True)
        normalized_dir.mkdir(exist_ok=True)
        
        all_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
        
        if len(all_images) == 0:
            print(f"  ⚠️ No images to process for {year}")
            continue
        
        print(f"  🔲 Creating {tile_size}×{tile_size} tiles...")
        tile_count = 0
        
        for img_path in all_images[:5]:
            try:
                img = Image.open(img_path).convert('RGB')
                img_width, img_height = img.size
                
                for y in range(0, img_height - tile_size + 1, stride):
                    for x in range(0, img_width - tile_size + 1, stride):
                        tile = img.crop((x, y, x + tile_size, y + tile_size))
                        tile.save(tiles_dir / f"tile_{tile_count:04d}.png")
                        tile_count += 1
            except Exception as e:
                print(f"    ⚠️ Error processing {img_path}: {e}")
        
        print(f"  ✓ Created {tile_count} tiles")
        
        print(f"  🎨 Normalizing tiles...")
        tile_files = list(tiles_dir.glob("*.png"))
        
        for tile_path in tqdm(tile_files, desc=f"  Normalizing {year}"):
            try:
                img = Image.open(tile_path).convert('RGB')
                tensor = transform(img)
                np.save(normalized_dir / f"{tile_path.stem}.npy", tensor.numpy())
            except Exception as e:
                pass
        
        print(f"  ✓ Normalized {len(tile_files)} tiles")

    # PHASE 4: Extract features
    print("\n" + "=" * 80)
    print("PHASE 4: Feature Extraction with MAE Model")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    checkpoint_path = "checkpoints_30k/checkpoint_final.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📦 Loading MAE model from {checkpoint_path}...")
    encoder = EncoderLoader(checkpoint_path=checkpoint_path, device=device, model_type='full')
    print("✅ Model loaded successfully")
    
    for year in years:
        print(f"\n{'─' * 60}")
        print(f"🧠 Extracting Features: {year}")
        print(f"{'─' * 60}")
        
        tiles_dir = output_dir / f"tiles_{year}"
        
        if not tiles_dir.exists():
            print(f"  ⚠️ No tiles directory for {year}")
            continue
        
        tile_files = list(tiles_dir.glob("*.png"))
        if len(tile_files) == 0:
            print(f"  ⚠️ No tiles to process for {year}")
            continue
        
        max_tiles = min(100, len(tile_files))
        tile_paths = [str(p) for p in tile_files[:max_tiles]]
        
        print(f"  📊 Processing {len(tile_paths)} tiles...")
        
        try:
            features = encoder.extract_features_batch(tile_paths)
            
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            
            feature_file = output_dir / f"features_{year}.npy"
            np.save(feature_file, features)
            feature_files[year] = str(feature_file)
            
            print(f"  ✓ Features extracted: {features.shape}")
            print(f"  💾 Saved to: {feature_file}")
            
        except Exception as e:
            print(f"  ❌ Feature extraction failed for {year}: {e}")
            continue

    # PHASE 5: Analysis and Visualization
    print("\n" + "=" * 80)
    print("PHASE 5: Analysis & Visualization Generation")
    print("=" * 80)
    
    if not feature_files:
        print("❌ No features extracted. Cannot perform analysis.")
        return
    
    print(f"📊 Years with features: {list(feature_files.keys())}")
    
    all_features_by_year = {}
    for year in sorted(feature_files.keys()):
        all_features_by_year[year] = np.load(feature_files[year])
    
    years_list = sorted(all_features_by_year.keys())
    
    # Calculate indicators based on condition
    print(f"\n{'─' * 60}")
    print(f"📊 Calculating {args.condition.title()} Indicators...")
    print(f"{'─' * 60}")
    
    tile_area_km2 = 1.6384
    indicators_by_year = {}
    
    for year in years_list:
        features = all_features_by_year[year]
        feature_means = features.mean(axis=1)
        feature_variance = features.var(axis=1)
        
        # Different thresholding based on condition
        if args.condition == 'water':
            threshold = np.percentile(feature_means, 30)
            detected = (feature_means < threshold).sum()
            label = "Water Bodies"
        elif args.condition == 'vegetation':
            threshold = np.percentile(feature_means, 60)
            detected = (feature_means > threshold).sum()
            label = "Vegetation"
        elif args.condition == 'climate' or args.condition == 'urban':
            threshold = np.percentile(feature_variance, 70)
            detected = (feature_variance > threshold).sum()
            label = "Urban/Complex"
        else:
            threshold = np.percentile(feature_means, 20)
            detected = (feature_means < threshold).sum()
            label = "Target Area"
        
        detected_area = detected * tile_area_km2
        total_tiles = len(features)
        coverage_pct = detected / total_tiles * 100
        
        indicators_by_year[year] = {
            'detected_tiles': int(detected),
            'detected_area_km2': float(detected_area),
            'coverage_percentage': float(coverage_pct),
            'total_tiles': total_tiles
        }
        
        print(f"\n  📅 Year {year}:")
        print(f"     {label}: {detected_area:.2f} km² ({coverage_pct:.1f}%)")
    
    # Generate visualizations (year-by-year)
    print(f"\n{'─' * 60}")
    print(f"📊 Creating Year-by-Year Visualizations...")
    print(f"{'─' * 60}")
    
    for year in years_list:
        features = all_features_by_year[year]
        print(f"\n  📈 Analysis for Year: {year}")
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        variance_explained = pca.explained_variance_ratio_.sum() * 100
        
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: PCA clusters
        scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=clusters, cmap='viridis', s=30, alpha=0.7)
        axes[0, 0].set_title(f'{year} - Pattern Analysis', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # Plot 2: Cluster distribution
        zone_counts = [(clusters == i).sum() for i in range(n_clusters)]
        axes[0, 1].bar(range(n_clusters), zone_counts, color='#2196F3', edgecolor='black', alpha=0.8)
        axes[0, 1].set_title(f'{year} - Zone Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Zone ID', fontsize=11)
        axes[0, 1].set_ylabel('Samples', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Feature distribution
        axes[1, 0].hist(features.mean(axis=1), bins=20, edgecolor='black', alpha=0.7, color='#4CAF50')
        axes[1, 0].set_title(f'{year} - Feature Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Mean Feature Value', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Statistics
        axes[1, 1].axis('off')
        climate_info = indicators_by_year[year]
        
        stats = {
            'Total Samples': features.shape[0],
            'Feature Dimensions': features.shape[1],
            'Detected Area': f"{climate_info['detected_area_km2']:.2f} km²",
            'Coverage': f"{climate_info['coverage_percentage']:.1f}%",
            'Mean Feature': f"{features.mean():.4f}",
            'Std Feature': f"{features.std():.4f}"
        }
        
        table_data = [[key, str(value)] for key, value in stats.items()]
        table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'Value'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title(f'{year} - Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        year_viz_path = output_dir / f"year_{year}_analysis.png"
        plt.savefig(year_viz_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {year_viz_path}")
        plt.close()
    
    # Before/After comparison
    print(f"\n{'─' * 60}")
    print(f"📊 Creating Comparison Analysis...")
    print(f"{'─' * 60}")
    
    if len(years_list) >= 2:
        first_year = years_list[0]
        last_year = years_list[-1]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Feature distribution comparison
        for year in [first_year, last_year]:
            features = all_features_by_year[year]
            means = features.mean(axis=1)
            color = '#2196F3' if year == first_year else '#F44336'
            label = f'{year} (Before)' if year == first_year else f'{year} (After)'
            axes[0].hist(means, bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
        
        axes[0].set_title(f'Before vs After: {first_year} vs {last_year}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Mean Feature Value', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Area comparison
        categories = ['Detected Area (km²)', 'Coverage (%)']
        values_first = [indicators_by_year[first_year]['detected_area_km2'],
                       indicators_by_year[first_year]['coverage_percentage']]
        values_last = [indicators_by_year[last_year]['detected_area_km2'],
                      indicators_by_year[last_year]['coverage_percentage']]
        
        x = np.arange(len(categories))
        width = 0.35
        axes[1].bar(x - width/2, values_first, width, label=str(first_year), color='#2196F3', edgecolor='black')
        axes[1].bar(x + width/2, values_last, width, label=str(last_year), color='#F44336', edgecolor='black')
        axes[1].set_title(f'Area Comparison: {first_year} vs {last_year}', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories, fontsize=10)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        comparison_path = output_dir / "before_after_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {comparison_path}")
        plt.close()
    
    # Trend analysis
    print(f"\n{'─' * 60}")
    print(f"📊 Creating Trend Analysis...")
    print(f"{'─' * 60}")
    
    if len(years_list) >= 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        areas = [indicators_by_year[year]['detected_area_km2'] for year in years_list]
        ax.plot(years_list, areas, 'b-', linewidth=3, marker='o', markersize=10, label='Detected Area')
        ax.set_title(f'{args.condition.title()} Trend ({years_list[0]}-{years_list[-1]})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Area (km²)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        for year, area in zip(years_list, areas):
            ax.annotate(f'{area:.1f}', (year, area), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        trend_path = output_dir / "historical_trend_analysis.png"
        plt.savefig(trend_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {trend_path}")
        plt.close()

    # PHASE 6: Generate interactive report
    print("\n" + "=" * 80)
    print("PHASE 6: Generating Interactive Visual Report")
    print("=" * 80)
    
    if feature_files:
        try:
            report_path = create_interactive_report(
                analysis_dir=str(output_dir),
                place_name=args.location,
                condition=args.condition,
                start_year=min(feature_files.keys()),
                end_year=max(feature_files.keys()),
                feature_files=feature_files
            )
            print(f"✅ Interactive report generated: {report_path}")
        except Exception as e:
            print(f"⚠️ Report generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 80)
    print(f"📍 Location: {args.location}")
    print(f"📅 Period: {args.start_year} - {args.end_year}")
    print(f"🔍 Condition: {args.condition.title()}")
    print(f"📊 Years Processed: {sorted(feature_files.keys())}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print(f"\n🚀 To open the report, run:")
    print(f"   start {output_dir}\\VISUAL_REPORT.html")
    print("=" * 80)

if __name__ == "__main__":
    main()
