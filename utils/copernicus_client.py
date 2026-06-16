"""
Copernicus API Client with Automatic Token Refresh
"""

import requests
import time
from pathlib import Path

class CopernicusClient:
    """Client for Copernicus Data Space API with token management"""
    
    def __init__(self, access_token: str, refresh_token: str):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.search_url = "https://catalogue.dataspace.copernicus.eu/stac/search"
        self.client_id = "cdse-public"
        self.token_expires_at = time.time() + 1800  # Token expires in 30 minutes
        
    def refresh_access_token(self):
        """Refresh the access token using refresh token"""
        print("🔄 Refreshing access token...")
        
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": self.refresh_token
        }
        
        try:
            response = requests.post(self.token_url, data=payload)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data.get('refresh_token', self.refresh_token)
            self.token_expires_at = time.time() + token_data.get('expires_in', 1800)
            
            print("✅ Access token refreshed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to refresh token: {e}")
            return False
    
    def ensure_valid_token(self):
        """Ensure we have a valid access token"""
        # Refresh if token expires in less than 5 minutes
        if time.time() > (self.token_expires_at - 300):
            return self.refresh_access_token()
        return True
    
    def search_images(self, bbox: list, start_date: str, end_date: str, limit: int = 10):
        """Search for Sentinel-2 images"""
        
        if not self.ensure_valid_token():
            return None
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Proper datetime format with T separator
        datetime_str = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
        
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": datetime_str,
            "limit": limit
        }
        
        try:
            print(f"🔍 Searching for images...")
            print(f"   BBox: {bbox}")
            print(f"   Date: {datetime_str}")
            
            response = requests.post(
                self.search_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                n_images = len(data.get('features', []))
                print(f"✅ Found {n_images} satellite images")
                return data
            else:
                print(f"❌ Search failed: {response.status_code}")
                print(f"   Error: {response.text[:200]}")
                return None
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {e}")
            print("   Possible causes:")
            print("   - Firewall blocking the API")
            print("   - Network connectivity issue")
            print("   - API endpoint temporarily unavailable")
            return None
        except Exception as e:
            print(f"❌ Error searching images: {e}")
            return None
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download a single image"""
        
        if not self.ensure_valid_token():
            return False
        
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        try:
            response = requests.get(image_url, headers=headers, timeout=60)
            response.raise_for_status()
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {image_url}: {e}")
            return False
    
    def download_images(self, search_results: dict, output_dir: str):
        """Download all images from search results"""
        
        if not search_results or 'features' not in search_results:
            print("❌ No search results to download")
            return 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        features = search_results['features']
        print(f"\n📥 Downloading {len(features)} images...")
        
        downloaded = 0
        for i, feature in enumerate(features):
            # Get image URL
            assets = feature.get('assets', {})
            img_url = None
            
            if 'preview' in assets:
                img_url = assets['preview']['href']
            elif 'thumbnail' in assets:
                img_url = assets['thumbnail']['href']
            
            if not img_url:
                print(f"  ⚠️ Image {i+1}: No preview available, skipping")
                continue
            
            # Download
            save_path = output_path / f"image_{i:03d}.jpg"
            print(f"  [{i+1}/{len(features)}] Downloading...", end=" ")
            
            if self.download_image(img_url, str(save_path)):
                print(f"✓ {save_path.name}")
                downloaded += 1
            else:
                print(f"✗ Failed")
        
        print(f"\n✅ Downloaded {downloaded}/{len(features)} images to {output_dir}")
        return downloaded
