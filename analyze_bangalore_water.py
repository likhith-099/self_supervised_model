"""
Bangalore Water Body Analysis 2020-2023
Downloads real satellite imagery and analyzes water bodies
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.copernicus_client import CopernicusClient
from utils.visual_report import create_interactive_report

# New API tokens
ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJYVUh3VWZKaHVDVWo0X3k4ZF8xM0hxWXBYMFdwdDd2anhob2FPLUxzREZFIn0.eyJleHAiOjE3NzcwNDk2MzAsImlhdCI6MTc3NzA0NzgzMCwianRpIjoib25ydHJvOjc2ZmYxMDg4LWU4YjQtZWM4OS0wMjM5LTVkOGI3MTlmMDQ3NyIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkuZGF0YXNwYWNlLmNvcGVybmljdXMuZXUvYXV0aC9yZWFsbXMvQ0RTRSIsImF1ZCI6WyJDTE9VREZFUlJPX1BVQkxJQyIsImFjY291bnQiXSwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiY2RzZS1wdWJsaWMiLCJzaWQiOiJlODdkYWE3OC02Zjg0LTUwNTQtNmI1OS1lZGJlOWFlZWEzYTYiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9sb2NhbGhvc3Q6NDIwMCIsIioiLCJodHRwczovL3dvcmtzcGFjZS5zdGFnaW5nLWNkc2UtZGF0YS1leHBsb3Jlci5hcHBzLnN0YWdpbmcuaW50cmEuY2xvdWRmZXJyby5jb20iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNvcGVybmljdXMtZ2VuZXJhbC1xdW90YSIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLWNkYXMiLCJjb3Blcm5pY3VzLWdlbmVyYWwiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6IkFVRElFTkNFX1BVQkxJQyBvcGVuaWQgZW1haWwgcHJvZmlsZSBvbmRlbWFuZF9wcm9jZXNzaW5nIHVzZXItY29udGV4dCIsImdyb3VwX21lbWJlcnNoaXAiOlsiL2FjY2Vzc19ncm91cHMvdXNlcl90eXBvbG9neS9jb3Blcm5pY3VzX2dlbmVyYWwiLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIiXSwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJTbmVoYSBTaGV0dHkiLCJvcmdhbml6YXRpb25zIjpbImRlZmF1bHQtNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIl0sInVzZXJfY29udGV4dF9pZCI6Ijc3N2NhMDhjLWY0OTUtNDM5Ni04ODkzLTNiZmI3YjU0N2JlNiIsImNvbnRleHRfcm9sZXMiOnt9LCJjb250ZXh0X2dyb3VwcyI6WyIvYWNjZXNzX2dyb3Vwcy91c2VyX3R5cG9sb2d5L2NvcGVybmljdXNfZ2VuZXJhbC8iLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIvIl0sInByZWZlcnJlZF91c2VybmFtZSI6InNuZWhhc2hldHR5LjE4MDVAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNuZWhhIiwiZmFtaWx5X25hbWUiOiJTaGV0dHkiLCJ1c2VyX2NvbnRleHQiOiJkZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4ZiIsImVtYWlsIjoic25laGFzaGV0dHkuMTgwNUBnbWFpbC5jb20ifQ.eTalJusHO2_DQ5qdWdq6lS8GwzLB5uVVL1HetSjoILZOA5CXacA3Ly4BOTVrt_MxHvwcL_ZpfVx69mtKpRnI_kMQHEU_pTTDCNJCatD8I0KDqX7tOgZ933dh0Ke9yAHWUr1Pa7CyByP5wNxQ6LLkZHx5L-pXnAVzkTjyP7bFOuH9DizrnzakuDGvZ-SUEzAgoCPTYHECHD24frUhChN_wWAHXT1lDc9miDROqcwHp1bObwrVaWClcBhA6uix5iYaUj2jZ6-v6oADUhqmBfBpixeJvHsViYhbppbbQtl40VPmj9XRoWnSNKlyb6ewoQkWxKBbHag32WgSi9AI5XRNAQ"
REFRESH_TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJjODg0ZDhkYi0wNjY2LTQ5OWYtOTJkYi1hNTQwODZmMDQ5YjkifQ.eyJleHAiOjE3NzcwNTE0MzAsImlhdCI6MTc3NzA0NzgzMCwianRpIjoiYzk0MzViZDQtNzg4YS04YzUzLTVlYmQtMTMwMTJmNmVmMTE3IiwiaXNzIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwiYXVkIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiUmVmcmVzaCIsImF6cCI6ImNkc2UtcHVibGljIiwic2lkIjoiZTg3ZGFhNzgtNmY4NC01MDU0LTZiNTktZWRiZTlhZWVhM2E2Iiwic2NvcGUiOiJBVURJRU5DRV9QVUJMSUMgYmFzaWMgd2ViLW9yaWdpbnMgb3BlbmlkIGVtYWlsIHJvbGVzIHByb2ZpbGUgb25kZW1hbmRfcHJvY2Vzc2luZyB1c2VyLWNvbnRleHQifQ.IgbaNlmCGFjBv-8LfepXa3MbQxJHJNOu5Jz-RN_gzVTofoPcF58SCPRZPgqnvC1lQN3B6KEv_QcgqMOoszyjCA"

print("=" * 80)
print("💧 BANGALORE WATER BODY ANALYSIS (2020-2023)")
print("=" * 80)
print("\n📍 Location: Bangalore, Karnataka, India")
print("📅 Period: 2020 - 2023 (4 years)")
print("🌊 Analyzing: Water body changes")
print("=" * 80)

# Output directory
output_dir = Path("analysis_output/bangalore_water_2020_2023")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize API client
print("\n[1/6] Initializing Copernicus API client...")
client = CopernicusClient(ACCESS_TOKEN, REFRESH_TOKEN)
print("✅ API client initialized")

# Get Bangalore coordinates
print("\n[2/6] Getting Bangalore coordinates...")
# Bangalore bounding box: [min_lon, min_lat, max_lon, max_lat]
bbox = [77.4, 12.8, 77.8, 13.1]  # Bangalore area
print(f"✓ Bounding box: {bbox}")
print(f"  Center: 12.9716°N, 77.5941°E")

# Download satellite images for each year
years = [2020, 2021, 2022, 2023]
feature_files = {}

for year in years:
    print(f"\n[3/6] Processing {year}...")
    
    # Check if already downloaded
    year_dir = output_dir / f"raw_{year}"
    year_dir.mkdir(exist_ok=True)
    
    # Count existing images
    existing_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
    
    if len(existing_images) >= 5:
        print(f"  ✓ Using {len(existing_images)} existing images from {year}")
    else:
        # Download new images
        print(f"  📥 Downloading satellite images for {year}...")
        try:
            results = client.search_images(
                bbox=bbox,
                start_date=f"{year}-06-01",
                end_date=f"{year}-06-30",
                limit=5
            )
            
            if results:
                client.download_images(results, str(year_dir))
                print(f"  ✓ Downloaded {len(results)} images for {year}")
            else:
                print(f"  ⚠️ No images found for {year}")
                continue
        except Exception as e:
            print(f"  ⚠️ Download failed for {year}: {e}")
            continue
    
    # Tile and process images
    print(f"  🔲 Creating 128×128 tiles...")
    tiles_dir = output_dir / f"tiles_{year}"
    tiles_dir.mkdir(exist_ok=True)
    
    from PIL import Image
    tile_size = 128
    stride = 128
    tile_count = 0
    
    all_images = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
    
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
    import torchvision.transforms as transforms
    from tqdm import tqdm
    
    normalized_dir = output_dir / f"normalized_{year}"
    normalized_dir.mkdir(exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tile_files = list(tiles_dir.glob("*.png"))
    for tile_path in tqdm(tile_files, desc=f"  Normalizing {year}"):
        try:
            img = Image.open(tile_path).convert('RGB')
            tensor = transform(img)
            np.save(normalized_dir / f"{tile_path.stem}.npy", tensor.numpy())
        except Exception as e:
            pass
    
    print(f"  ✓ Normalized {len(tile_files)} tiles")
    
    # Extract features
    print(f"  🧠 Extracting features with MAE model...")
    from models.encoder import EncoderLoader
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "checkpoints_30k/checkpoint_final.pth"
    
    encoder = EncoderLoader(checkpoint_path=checkpoint_path, device=device, model_type='full')
    
    tile_paths = [str(p) for p in tile_files[:min(50, len(tile_files))]]  # Max 50 tiles
    features = encoder.extract_features_batch(tile_paths)
    
    # Flatten features
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)
    
    feature_file = output_dir / f"features_{year}.npy"
    np.save(feature_file, features)
    feature_files[year] = str(feature_file)
    
    print(f"  ✓ Features extracted: {features.shape}")

print(f"\n[4/6] Analyzing water body patterns...")

if feature_files:
    # Use first year's features for analysis
    first_year = list(feature_files.keys())[0]
    features = np.load(feature_files[first_year])
    
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    # PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    print(f"✓ PCA: {features.shape[1]}D → 2D (Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%)")
    
    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    print(f"✓ K-Means: 5 clusters identified")
    
    for i in range(5):
        count = (clusters == i).sum()
        print(f"  Cluster {i}: {count} tiles ({count/len(clusters)*100:.1f}%)")

print(f"\n[5/6] Generating visualizations...")
print("  ⚠️ Run generate_all_visualizations.py separately for detailed charts")

print(f"\n[6/6] Creating interactive visual report...")

if feature_files:
    report_path = create_interactive_report(
        analysis_dir=str(output_dir),
        place_name="Bangalore, Karnataka, India",
        condition="water",
        start_year=min(feature_files.keys()),
        end_year=max(feature_files.keys()),
        feature_files=feature_files
    )
    
    print("\n" + "=" * 80)
    print("✅ BANGALORE WATER ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📍 Location: Bangalore, Karnataka, India")
    print(f"📅 Period: {min(feature_files.keys())} - {max(feature_files.keys())}")
    print(f"🌊 Condition: Water Body Analysis")
    print(f"\n📊 Years Processed: {list(feature_files.keys())}")
    print(f"📁 Results: {output_dir}")
    print(f"\n🎨 Interactive Report: {report_path}")
    print("=" * 80)
else:
    print("\n⚠️ No features extracted. Check API connectivity and try again.")
