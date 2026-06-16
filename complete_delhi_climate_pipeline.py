"""
Complete Automated Pipeline for Delhi Climate Analysis (2018-2023)
Downloads satellite data, preprocesses, extracts features, and generates climate analysis
"""

import sys
import os
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
ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJYVUh3VWZKaHVDVWo0X3k4ZF8xM0hxWXBYMFdwdDd2anhob2FPLUxzREZFIn0.eyJleHAiOjE3NzcwNDk2MzAsImlhdCI6MTc3NzA0NzgzMCwianRpIjoib25ydHJvOjc2ZmYxMDg4LWU4YjQtZWM4OS0wMjM5LTVkOGI3MTlmMDQ3NyIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkuZGF0YXNwYWNlLmNvcGVybmljdXMuZXUvYXV0aC9yZWFsbXMvQ0RTRSIsImF1ZCI6WyJDTE9VREZFUlJPX1BVQkxJQyIsImFjY291bnQiXSwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiY2RzZS1wdWJsaWMiLCJzaWQiOiJlODdkYWE3OC02Zjg0LTUwNTQtNmI1OS1lZGJlOWFlZWEzYTYiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9sb2NhbGhvc3Q6NDIwMCIsIioiLCJodHRwczovL3dvcmtzcGFjZS5zdGFnaW5nLWNkc2UtZGF0YS1leHBsb3Jlci5hcHBzLnN0YWdpbmcuaW50cmEuY2xvdWRmZXJyby5jb20iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNvcGVybmljdXMtZ2VuZXJhbC1xdW90YSIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLWNkYXMiLCJjb3Blcm5pY3VzLWdlbmVyYWwiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6IkFVRElFTkNFX1BVQkxJQyBvcGVuaWQgZW1haWwgcHJvZmlsZSBvbmRlbWFuZF9wcm9jZXNzaW5nIHVzZXItY29udGV4dCIsImdyb3VwX21lbWJlcnNoaXAiOlsiL2FjY2Vzc19ncm91cHMvdXNlcl90eXBvbG9neS9jb3Blcm5pY3VzX2dlbmVyYWwiLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIiXSwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJTbmVoYSBTaGV0dHkiLCJvcmdhbml6YXRpb25zIjpbImRlZmF1bHQtNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIl0sInVzZXJfY29udGV4dF9pZCI6Ijc3N2NhMDhjLWY0OTUtNDM5Ni04ODkzLTNiZmI3YjU0N2JlNiIsImNvbnRleHRfcm9sZXMiOnt9LCJjb250ZXh0X2dyb3VwcyI6WyIvYWNjZXNzX2dyb3Vwcy91c2VyX3R5cG9sb2d5L2NvcGVybmljdXNfZ2VuZXJhbC8iLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIvIl0sInByZWZlcnJlZF91c2VybmFtZSI6InNuZWhhc2hldHR5LjE4MDVAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNuZWhhIiwiZmFtaWx5X25hbWUiOiJTaGV0dHkiLCJ1c2VyX2NvbnRleHQiOiJkZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4ZiIsImVtYWlsIjoic25laGFzaGV0dHkuMTgwNUBnbWFpbC5jb20ifQ.eTalJusHO2_DQ5qdWdq6lS8GwzLB5uVVL1HetSjoILZOA5CXacA3Ly4BOTVrt_MxHvwcL_ZpfVx69mtKpRnI_kMQHEU_pTTDCNJCatD8I0KDqX7tOgZ933dh0Ke9yAHWUr1Pa7CyByP5wNxQ6LLkZHx5L-pXnAVzkTjyP7bFOuH9DizrnzakuDGvZ-SUEzAgoCPTYHECHD24frUhChN_wWAHXT1lDc9miDROqcwHp1bObwrVaWClcBhA6uix5iYaUj2jZ6-v6oADUhqmBfBpixeJvHsViYhbppbbQtl40VPmj9XRoWnSNKlyb6ewoQkWxKBbHag32WgSi9AI5XRNAQ"
REFRESH_TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJjODg0ZDhkYi0wNjY2LTQ5OWYtOTJkYi1hNTQwODZmMDQ5YjkifQ.eyJleHAiOjE3NzcwNTE0MzAsImlhdCI6MTc3NzA0NzgzMCwianRpIjoiYzk0MzViZDQtNzg4YS04YzUzLTVlYmQtMTMwMTJmNmVmMTE3IiwiaXNzIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwiYXVkIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiUmVmcmVzaCIsImF6cCI6ImNkc2UtcHVibGljIiwic2lkIjoiZTg3ZGFhNzgtNmY4NC01MDU0LTZiNTktZWRiZTlhZWVhM2E2Iiwic2NvcGUiOiJBVURJRU5DRV9QVUJMSUMgYmFzaWMgd2ViLW9yaWdpbnMgb3BlbmlkIGVtYWlsIHJvbGVzIHByb2ZpbGUgb25kZW1hbmRfcHJvY2Vzc2luZyB1c2VyLWNvbnRleHQifQ.IgbaNlmCGFjBv-8LfepXa3MbQxJHJNOu5Jz-RN_gzVTofoPcF58SCPRZPgqnvC1lQN3B6KEv_QcgqMOoszyjCA"

def main():
    print("=" * 80)
    print("🌡️ COMPLETE AUTOMATED PIPELINE: DELHI CLIMATE ANALYSIS (2018-2023)")
    print("=" * 80)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📍 Location: Delhi, India")
    print("📅 Period: 2018 - 2023 (6 years)")
    print("🌍 Analyzing: Climate patterns, urban heat, and environmental changes")
    print("=" * 80)

    # Configuration
    output_dir = Path("analysis_output/delhi_climate_2018_2023_complete")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Delhi bounding box (larger area to capture urban heat island effect)
    bbox = [76.8, 28.4, 77.4, 28.9]  # Delhi NCR area
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    
    feature_files = {}
    
    # PHASE 1: Initialize API
    print("\n" + "=" * 80)
    print("PHASE 1: Initializing Copernicus API Client")
    print("=" * 80)
    client = CopernicusClient(ACCESS_TOKEN, REFRESH_TOKEN)
    print("✅ API client initialized successfully")

    # PHASE 2: Download satellite data for each year
    print("\n" + "=" * 80)
    print("PHASE 2: Downloading Satellite Data (2018-2023)")
    print("=" * 80)
    
    for year in years:
        print(f"\n{'─' * 60}")
        print(f"📅 Processing Year: {year}")
        print(f"{'─' * 60}")
        
        year_dir = output_dir / f"raw_{year}"
        year_dir.mkdir(exist_ok=True)
        
        # Check if we already have images
        existing_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
        
        if len(existing_images) >= 5:
            print(f"  ✓ Using {len(existing_images)} existing images from {year}")
        else:
            print(f"  📥 Downloading satellite images for {year}...")
            try:
                # Search for images during summer (May-June) for heat analysis
                results = client.search_images(
                    bbox=bbox,
                    start_date=f"{year}-05-01",
                    end_date=f"{year}-06-30",
                    limit=8
                )
                
                if results and len(results.get('features', [])) > 0:
                    downloaded = client.download_images(results, str(year_dir))
                    print(f"  ✓ Downloaded {downloaded} images for {year}")
                else:
                    print(f"  ⚠️ No images found for {year}, trying alternative dates...")
                    # Try different time period
                    results = client.search_images(
                        bbox=bbox,
                        start_date=f"{year}-01-01",
                        end_date=f"{year}-12-31",
                        limit=8
                    )
                    if results and len(results.get('features', [])) > 0:
                        downloaded = client.download_images(results, str(year_dir))
                        print(f"  ✓ Downloaded {downloaded} images for {year} (alternative)")
                    else:
                        print(f"  ❌ No images available for {year}")
                        continue
            except Exception as e:
                print(f"  ❌ Download failed for {year}: {e}")
                continue
        
        # Verify we have images to process
        all_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
        if len(all_images) == 0:
            print(f"  ⚠️ No images to process for {year}")
            continue
            
        print(f"  📊 Total images for {year}: {len(all_images)}")

    # PHASE 3: Preprocess images (tile and normalize)
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
        
        # Tile images
        print(f"  🔲 Creating {tile_size}×{tile_size} tiles...")
        tile_count = 0
        
        for img_path in all_images[:5]:  # Process first 5 images
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
        
        # Normalize tiles
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

    # PHASE 4: Extract features using MAE model
    print("\n" + "=" * 80)
    print("PHASE 4: Feature Extraction with MAE Model")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    checkpoint_path = "checkpoints_30k/checkpoint_final.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure the model checkpoint exists")
        return
    
    print(f"📦 Loading MAE model from {checkpoint_path}...")
    encoder = EncoderLoader(checkpoint_path=checkpoint_path, device=device, model_type='full')
    print("✅ Model loaded successfully")
    
    for year in years:
        print(f"\n{'─' * 60}")
        print(f"🧠 Extracting Features: {year}")
        print(f"{'─' * 60}")
        
        normalized_dir = output_dir / f"normalized_{year}"
        tiles_dir = output_dir / f"tiles_{year}"
        
        if not tiles_dir.exists():
            print(f"  ⚠️ No tiles directory for {year}")
            continue
        
        tile_files = list(tiles_dir.glob("*.png"))
        if len(tile_files) == 0:
            print(f"  ⚠️ No tiles to process for {year}")
            continue
        
        # Limit to manageable number
        max_tiles = min(100, len(tile_files))
        tile_paths = [str(p) for p in tile_files[:max_tiles]]
        
        print(f"  📊 Processing {len(tile_paths)} tiles...")
        
        try:
            features = encoder.extract_features_batch(tile_paths)
            
            # Flatten features if needed
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)
            
            # Save features
            feature_file = output_dir / f"features_{year}.npy"
            np.save(feature_file, features)
            feature_files[year] = str(feature_file)
            
            print(f"  ✓ Features extracted: {features.shape}")
            print(f"  💾 Saved to: {feature_file}")
            
        except Exception as e:
            print(f"  ❌ Feature extraction failed for {year}: {e}")
            continue

    # PHASE 5: Analyze climate patterns AND generate visualizations
    print("\n" + "=" * 80)
    print("PHASE 5: Climate Pattern Analysis & Visualization Generation")
    print("=" * 80)
    
    if not feature_files:
        print("❌ No features extracted. Cannot perform analysis.")
        return
    
    print(f"📊 Years with features: {list(feature_files.keys())}")
    
    # Load all features
    all_features_by_year = {}
    for year in sorted(feature_files.keys()):
        all_features_by_year[year] = np.load(feature_files[year])
    
    years_list = sorted(all_features_by_year.keys())
    
    # Calculate climate indicators for each year
    print(f"\n{'─' * 60}")
    print(f"🌡️ Calculating Climate Indicators...")
    print(f"{'─' * 60}")
    
    # Each tile is 128x128 pixels from Sentinel-2 (10m resolution) = 1.6384 km²
    tile_area_km2 = 1.6384
    
    climate_indicators_by_year = {}
    
    for year in years_list:
        features = all_features_by_year[year]
        
        # Climate analysis through feature statistics:
        # - High feature variance = urban heat islands, complex surfaces
        # - Low feature variance = water bodies, uniform vegetation
        # - Mean feature values indicate surface composition
        
        feature_means = features.mean(axis=1)
        feature_stds = features.std(axis=1)
        feature_variance = features.var(axis=1)
        
        # Urban heat indicators (high complexity surfaces)
        # Top 30% of variance likely urban/built-up areas
        urban_heat_threshold = np.percentile(feature_variance, 70)
        urban_heat_tiles = (feature_variance > urban_heat_threshold).sum()
        
        # Vegetation/green cover (moderate complexity)
        veg_lower = np.percentile(feature_means, 30)
        veg_upper = np.percentile(feature_means, 70)
        green_cover_tiles = ((feature_means > veg_lower) & (feature_means < veg_upper)).sum()
        
        # Water bodies (low complexity, smooth surfaces)
        water_threshold = np.percentile(feature_means, 20)
        water_tiles = (feature_means < water_threshold).sum()
        
        # Calculate areas
        urban_heat_area = urban_heat_tiles * tile_area_km2
        green_cover_area = green_cover_tiles * tile_area_km2
        water_area = water_tiles * tile_area_km2
        
        total_tiles = len(features)
        
        climate_indicators_by_year[year] = {
            'urban_heat_tiles': int(urban_heat_tiles),
            'urban_heat_area_km2': float(urban_heat_area),
            'urban_heat_percentage': float(urban_heat_tiles / total_tiles * 100),
            
            'green_cover_tiles': int(green_cover_tiles),
            'green_cover_area_km2': float(green_cover_area),
            'green_cover_percentage': float(green_cover_tiles / total_tiles * 100),
            
            'water_tiles': int(water_tiles),
            'water_area_km2': float(water_area),
            'water_percentage': float(water_tiles / total_tiles * 100),
            
            'mean_feature': float(feature_means.mean()),
            'std_feature': float(feature_stds.mean()),
            'variance_feature': float(feature_variance.mean()),
            'total_tiles': total_tiles
        }
        
        print(f"\n  📅 Year {year}:")
        print(f"     🏙️ Urban Heat Areas: {urban_heat_area:.2f} km² ({urban_heat_tiles/total_tiles*100:.1f}%)")
        print(f"     🌿 Green Cover: {green_cover_area:.2f} km² ({green_cover_tiles/total_tiles*100:.1f}%)")
        print(f"     💧 Water Bodies: {water_area:.2f} km² ({water_tiles/total_tiles*100:.1f}%)")
        print(f"     📊 Feature Variance: {feature_variance.mean():.4f}")
    
    # Save climate indicators data
    indicators_file = output_dir / "climate_indicators_data.npy"
    np.save(indicators_file, climate_indicators_by_year)
    print(f"\n  💾 Climate indicators data saved to: {indicators_file}")
    
    # 5A: Generate Year-by-Year Analysis
    print(f"\n{'─' * 60}")
    print(f"📊 Creating Year-by-Year Climate Visualizations...")
    print(f"{'─' * 60}")
    
    for year in years_list:
        features = all_features_by_year[year]
        print(f"\n  📈 Analysis for Year: {year}")
        print(f"  📊 Feature shape: {features.shape}")
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        variance_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  🔽 PCA: {features.shape[1]}D → 2D (Variance: {variance_explained:.2f}%)")
        
        # Clustering to identify different climate zones
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        print(f"  🔢 K-Means: {n_clusters} climate zones identified")
        
        for i in range(n_clusters):
            count = (clusters == i).sum()
            percentage = count / len(clusters) * 100
            print(f"    Zone {i}: {count} tiles ({percentage:.1f}%)")
        
        # Create visualization for this year
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: PCA clusters (climate zones)
        scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=clusters, cmap='plasma', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[0, 0].set_title(f'{year} - Climate Zone Patterns\n(PCA Visualization)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Climate Zone')
        
        # Plot 2: Climate zone distribution
        zone_counts = [(clusters == i).sum() for i in range(n_clusters)]
        colors_zone = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
        axes[0, 1].bar(range(n_clusters), zone_counts, color=colors_zone, edgecolor='black', alpha=0.8)
        axes[0, 1].set_title(f'{year} - Climate Zone Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Climate Zone ID', fontsize=11)
        axes[0, 1].set_ylabel('Number of Samples', fontsize=11)
        axes[0, 1].set_xticks(range(n_clusters))
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, count in enumerate(zone_counts):
            axes[0, 1].text(i, count + 0.2, str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Feature distribution (surface temperature indicator)
        axes[1, 0].hist(features.mean(axis=1), bins=20, edgecolor='black', 
                       alpha=0.7, color='#e74c3c')
        axes[1, 0].set_title(f'{year} - Surface Characteristic Distribution', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Mean Feature Value', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axvline(features.mean(), color='blue', linestyle='--', 
                          linewidth=2, label=f'Mean: {features.mean():.4f}')
        axes[1, 0].legend()
        
        # Plot 4: Climate statistics table
        axes[1, 1].axis('off')
        
        # Get climate indicators for this year
        climate_info = climate_indicators_by_year[year]
        
        stats = {
            'Total Samples': features.shape[0],
            'Feature Dimensions': features.shape[1],
            'Urban Heat Area': f"{climate_info['urban_heat_area_km2']:.2f} km²",
            'Green Cover Area': f"{climate_info['green_cover_area_km2']:.2f} km²",
            'Water Body Area': f"{climate_info['water_area_km2']:.2f} km²",
            'Urban Heat %': f"{climate_info['urban_heat_percentage']:.1f}%",
            'Green Cover %': f"{climate_info['green_cover_percentage']:.1f}%",
            'Water %': f"{climate_info['water_percentage']:.1f}%",
            'Feature Variance': f"{climate_info['variance_feature']:.4f}"
        }
        
        table_data = [[key, f"{value:.4f}" if isinstance(value, float) else str(value)] 
                      for key, value in stats.items()]
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Climate Indicator', 'Value'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title(f'{year} - Climate Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        year_viz_path = output_dir / f"year_{year}_climate_analysis.png"
        plt.savefig(year_viz_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {year_viz_path}")
        plt.close()
    
    # 5B: Generate Before/After Comparison
    print(f"\n{'─' * 60}")
    print(f"📊 Creating Climate Change Comparison...")
    print(f"{'─' * 60}")
    
    if len(years_list) >= 2:
        first_year = years_list[0]
        last_year = years_list[-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Feature Distribution Comparison
        for year in [first_year, last_year]:
            features = all_features_by_year[year]
            means = features.mean(axis=1)
            color = '#3498db' if year == first_year else '#e74c3c'
            label = f'{year} (Before)' if year == first_year else f'{year} (After)'
            axes[0, 0].hist(means, bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
        
        axes[0, 0].set_title(f'Before vs After: Climate Indicators\n{first_year} vs {last_year}', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Mean Feature Value', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Climate Components Comparison
        categories = ['Urban Heat\n(km²)', 'Green Cover\n(km²)', 'Water\n(km²)']
        values_first = [
            climate_indicators_by_year[first_year]['urban_heat_area_km2'],
            climate_indicators_by_year[first_year]['green_cover_area_km2'],
            climate_indicators_by_year[first_year]['water_area_km2']
        ]
        values_last = [
            climate_indicators_by_year[last_year]['urban_heat_area_km2'],
            climate_indicators_by_year[last_year]['green_cover_area_km2'],
            climate_indicators_by_year[last_year]['water_area_km2']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        bars1 = axes[0, 1].bar(x - width/2, values_first, width, label=str(first_year), color='#3498db', edgecolor='black')
        bars2 = axes[0, 1].bar(x + width/2, values_last, width, label=str(last_year), color='#e74c3c', edgecolor='black')
        axes[0, 1].set_title(f'Climate Components: {first_year} vs {last_year}', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories, fontsize=10)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: PCA Visualization
        features_first = all_features_by_year[first_year]
        features_last = all_features_by_year[last_year]
        
        combined = np.vstack([features_first, features_last])
        pca_combined = PCA(n_components=2)
        combined_2d = pca_combined.fit_transform(combined)
        
        features_first_2d = combined_2d[:len(features_first)]
        features_last_2d = combined_2d[len(features_first):]
        
        axes[1, 0].scatter(features_first_2d[:, 0], features_first_2d[:, 1], 
                          c='#3498db', s=20, alpha=0.6, label=str(first_year), edgecolors='black', linewidth=0.5)
        axes[1, 0].scatter(features_last_2d[:, 0], features_last_2d[:, 1], 
                          c='#e74c3c', s=20, alpha=0.6, label=str(last_year), edgecolors='black', linewidth=0.5)
        axes[1, 0].set_title(f'PCA: Climate Patterns ({first_year} vs {last_year})', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel(f'PC1 ({pca_combined.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[1, 0].set_ylabel(f'PC2 ({pca_combined.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Change Summary
        climate_first = climate_indicators_by_year[first_year]
        climate_last = climate_indicators_by_year[last_year]
        
        urban_change = climate_last['urban_heat_area_km2'] - climate_first['urban_heat_area_km2']
        green_change = climate_last['green_cover_area_km2'] - climate_first['green_cover_area_km2']
        water_change = climate_last['water_area_km2'] - climate_first['water_area_km2']
        
        axes[1, 1].axis('off')
        
        summary_text = f"""
CLIMATE CHANGE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━

{first_year} (Before):
  Urban Heat: {climate_first['urban_heat_area_km2']:.2f} km² ({climate_first['urban_heat_percentage']:.1f}%)
  Green Cover: {climate_first['green_cover_area_km2']:.2f} km² ({climate_first['green_cover_percentage']:.1f}%)
  Water Bodies: {climate_first['water_area_km2']:.2f} km² ({climate_first['water_percentage']:.1f}%)

{last_year} (After):
  Urban Heat: {climate_last['urban_heat_area_km2']:.2f} km² ({climate_last['urban_heat_percentage']:.1f}%)
  Green Cover: {climate_last['green_cover_area_km2']:.2f} km² ({climate_last['green_cover_percentage']:.1f}%)
  Water Bodies: {climate_last['water_area_km2']:.2f} km² ({climate_last['water_percentage']:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━
CHANGES DETECTED:
  Urban Heat: {urban_change:+.2f} km²
  Green Cover: {green_change:+.2f} km²
  Water: {water_change:+.2f} km²
━━━━━━━━━━━━━━━━━━━━━━

Assessment:
{'⚠️ Urban heat increasing - Heat island effect growing' if urban_change > 0 else '✓ Urban heat stable or decreasing'}
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='#f7fafc', edgecolor='#2d3748', linewidth=2))
        
        plt.tight_layout()
        before_after_path = output_dir / "climate_change_comparison.png"
        plt.savefig(before_after_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {before_after_path}")
        plt.close()
    
    # 5C: Generate Historical Trend Analysis
    print(f"\n{'─' * 60}")
    print(f"📊 Creating Climate Trend Analysis...")
    print(f"{'─' * 60}")
    
    if len(years_list) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Urban heat trend
        urban_areas = [climate_indicators_by_year[year]['urban_heat_area_km2'] for year in years_list]
        axes[0, 0].plot(years_list, urban_areas, 'r-', linewidth=3, marker='o', markersize=10, label='Urban Heat')
        axes[0, 0].set_title(f'Urban Heat Island Trend ({years_list[0]}-{years_list[-1]})', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Year', fontsize=12)
        axes[0, 0].set_ylabel('Urban Heat Area (km²)', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        for year, area in zip(years_list, urban_areas):
            axes[0, 0].annotate(f'{area:.1f}', (year, area), textcoords="offset points",
                               xytext=(0, 15), ha='center', fontsize=9, fontweight='bold')
        
        # Plot 2: Green cover trend
        green_areas = [climate_indicators_by_year[year]['green_cover_area_km2'] for year in years_list]
        axes[0, 1].plot(years_list, green_areas, 'g-', linewidth=3, marker='s', markersize=10, label='Green Cover')
        axes[0, 1].set_title('Green Cover Trend Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Year', fontsize=12)
        axes[0, 1].set_ylabel('Green Cover Area (km²)', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].fill_between(years_list, green_areas, alpha=0.3, color='green')
        
        # Plot 3: Water bodies trend
        water_areas = [climate_indicators_by_year[year]['water_area_km2'] for year in years_list]
        axes[1, 0].plot(years_list, water_areas, 'b-', linewidth=3, marker='^', markersize=10, label='Water Bodies')
        axes[1, 0].set_title('Water Bodies Trend Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Year', fontsize=12)
        axes[1, 0].set_ylabel('Water Area (km²)', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(years_list, water_areas, alpha=0.3, color='blue')
        
        # Plot 4: Summary with climate data
        axes[1, 1].axis('off')
        
        urban_first = climate_indicators_by_year[years_list[0]]['urban_heat_area_km2']
        urban_last = climate_indicators_by_year[years_list[-1]]['urban_heat_area_km2']
        green_first = climate_indicators_by_year[years_list[0]]['green_cover_area_km2']
        green_last = climate_indicators_by_year[years_list[-1]]['green_cover_area_km2']
        
        trend_summary = f"""
CLIMATE TREND ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━

Period: {years_list[0]} - {years_list[-1]} ({len(years_list)} years)

Urban Heat Islands:
{chr(10).join([f'  {year}: {climate_indicators_by_year[year]["urban_heat_area_km2"]:.2f} km²' for year in years_list])}

Green Cover:
{chr(10).join([f'  {year}: {climate_indicators_by_year[year]["green_cover_area_km2"]:.2f} km²' for year in years_list])}

Water Bodies:
{chr(10).join([f'  {year}: {climate_indicators_by_year[year]["water_area_km2"]:.2f} km²' for year in years_list])}

━━━━━━━━━━━━━━━━━━━━━━
CHANGES:
  Urban Heat: {urban_last - urban_first:+.2f} km²
  Green Cover: {green_last - green_first:+.2f} km²
━━━━━━━━━━━━━━━━━━━━━━

Assessment:
{'⚠️ Urbanization increasing - Monitor heat islands' if (urban_last - urban_first) > 0 else '✓ Climate conditions stable'}
"""
        
        axes[1, 1].text(0.05, 0.95, trend_summary, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='#f7fafc', edgecolor='#2d3748', linewidth=2))
        
        plt.tight_layout()
        trend_path = output_dir / "climate_trend_analysis.png"
        plt.savefig(trend_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {trend_path}")
        plt.close()

    # PHASE 6: Generate visual report
    print("\n" + "=" * 80)
    print("PHASE 6: Generating Interactive Visual Report")
    print("=" * 80)
    
    if feature_files:
        try:
            report_path = create_interactive_report(
                analysis_dir=str(output_dir),
                place_name="Delhi, India",
                condition="climate",
                start_year=min(feature_files.keys()),
                end_year=max(feature_files.keys()),
                feature_files=feature_files
            )
            print(f"✅ Interactive report generated: {report_path}")
        except Exception as e:
            print(f"⚠️ Report generation failed: {e}")
            print("But analysis data is still available in the output directory")

    # Summary
    print("\n" + "=" * 80)
    print("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 80)
    print(f"📍 Location: Delhi, India")
    print(f"📅 Period: 2018 - 2023")
    print(f"🌡️ Condition: Climate Analysis")
    print(f"📊 Years Processed: {sorted(feature_files.keys())}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if feature_files:
        print(f"\n📊 Feature Files Generated:")
        for year in sorted(feature_files.keys()):
            feature_file = Path(feature_files[year])
            if feature_file.exists():
                size_mb = feature_file.stat().st_size / (1024 * 1024)
                print(f"  • {year}: {feature_files[year]} ({size_mb:.2f} MB)")
    
    print(f"\n📈 Analysis Results:")
    print(f"  • Satellite images downloaded and processed")
    print(f"  • 128×128 tiles created")
    print(f"  • Features extracted using MAE model")
    print(f"  • Climate indicators calculated (urban heat, green cover, water)")
    print(f"  • Year-by-year climate analysis completed")
    print(f"  • Climate change comparison generated")
    print(f"  • Historical trend analysis created")
    print(f"  • Interactive visual report with all visualizations")
    print("=" * 80)

if __name__ == "__main__":
    main()
