"""Test Copernicus API with access token"""
import requests

ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJYVUh3VWZKaHVDVWo0X3k4ZF8xM0hxWXBYMFdwdDd2anhob2FPLUxzREZFIn0.eyJleHAiOjE3NzU5MzQwMjEsImlhdCI6MTc3NTkzMjIyMSwianRpIjoib25ydHJvOjlmNTFhZjQwLWRmZTgtYjlhZS1jZjNlLWFhMTQ4YjhjNmRlMyIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkuZGF0YXNwYWNlLmNvcGVybmljdXMuZXUvYXV0aC9yZWFsbXMvQ0RTRSIsImF1ZCI6WyJDTE9VREZFUlJPX1BVQkxJQyIsImFjY291bnQiXSwic3ViIjoiNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiY2RzZS1wdWJsaWMiLCJzaWQiOiI4MWUxZGIyMC1lNDVlLTJiOTgtODBiOS0yNDE5NWRiMjA5NDMiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9sb2NhbGhvc3Q6NDIwMCIsIioiLCJodHRwczovL3dvcmtzcGFjZS5zdGFnaW5nLWNkc2UtZGF0YS1leHBsb3Jlci5hcHBzLnN0YWdpbmcuaW50cmEuY2xvdWRmZXJyby5jb20iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNvcGVybmljdXMtZ2VuZXJhbC1xdW90YSIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLWNkYXMiLCJjb3Blcm5pY3VzLWdlbmVyYWwiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6IkFVRElFTkNFX1BVQkxJQyBvcGVuaWQgZW1haWwgcHJvZmlsZSBvbmRlbWFuZF9wcm9jZXNzaW5nIHVzZXItY29udGV4dCIsImdyb3VwX21lbWJlcnNoaXAiOlsiL2FjY2Vzc19ncm91cHMvdXNlcl90eXBvbG9neS9jb3Blcm5pY3VzX2dlbmVyYWwiLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIiXSwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJTbmVoYSBTaGV0dHkiLCJvcmdhbml6YXRpb25zIjpbImRlZmF1bHQtNTFiMTc2MDUtNWE1Ny00MTExLTkxOGMtM2EwYzRmYzFjMDhmIl0sInVzZXJfY29udGV4dF9pZCI6Ijc3N2NhMDhjLWY0OTUtNDM5Ni04ODkzLTNiZmI3YjU0N2JlNiIsImNvbnRleHRfcm9sZXMiOnt9LCJjb250ZXh0X2dyb3VwcyI6WyIvYWNjZXNzX2dyb3Vwcy91c2VyX3R5cG9sb2d5L2NvcGVybmljdXNfZ2VuZXJhbC8iLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4Zi9yZWd1bGFyX3VzZXIvIl0sInByZWZlcnJlZF91c2VybmFtZSI6InNuZWhhc2hldHR5LjE4MDVAZ21haWwuY29tIiwiZ2l2ZW5fbmFtZSI6IlNuZWhhIiwiZmFtaWx5X25hbWUiOiJTaGV0dHkiLCJ1c2VyX2NvbnRleHQiOiJkZWZhdWx0LTUxYjE3NjA1LTVhNTctNDExMS05MThjLTNhMGM0ZmMxYzA4ZiIsImVtYWlsIjoic25laGFzaGV0dHkuMTgwNUBnbWFpbC5jb20ifQ.mW_A2GGI0cNr6n7uskKTR_yugxj_YILmclSnzNJ6oeA1BcnO4nURIwV2EcmfWdifx3P3LRYDugrKRYYHJGwWpAKYhHezJon8zD4nPExncH3-WQ2O2q6A_MK--bwU6NQa-kHk-fDfKOvSwINer9nkOFcvMnvapsDiWgIitHHOZfxpGhQJMGVAAyLPTGlB3d_rE9wyGErirV5V7cBI_nMgIxzI8Zi6WonAm_BGTkjhdRhutsTwdhioQNBpdosr7Y3tlzupbvYNLwU0xWQIfoaV96paMj4A_9DCAeX0ul5HszPoPFBZPNDrpGbgdFol5eFSXQx4NDEFOmBItxHYYsUyug"

# Try different bbox formats
bboxes = {
    "Small area (Srinagar)": [74.7, 34.0, 75.0, 34.2],
    "Medium area": [74.5, 33.5, 75.5, 34.5],
    "Large area": [73.5, 32.3, 80.3, 36.9]
}

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

for name, bbox in bboxes.items():
    print(f"\n{'='*80}")
    print(f"Testing bbox: {name}")
    print(f"BBox: {bbox}")
    print(f"{'='*80}")
    
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": "2024-06-01/2024-06-30",  # Use 2024 data (available)
        "limit": 5
    }
    
    try:
        response = requests.post(
            "https://catalogue.dataspace.copernicus.eu/stac/search",
            json=payload,
            headers=headers
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            n_features = len(data.get('features', []))
            print(f"✅ SUCCESS! Found {n_features} images")
            
            if n_features > 0:
                # Show first image info
                first = data['features'][0]
                print(f"\nFirst image:")
                print(f"  ID: {first.get('id', 'N/A')}")
                print(f"  Date: {first.get('properties', {}).get('datetime', 'N/A')}")
                print(f"  Cloud cover: {first.get('properties', {}).get('eo:cloud_cover', 'N/A')}%")
                
                # Check available assets
                assets = first.get('assets', {})
                print(f"  Available assets: {list(assets.keys())}")
        else:
            print(f"❌ Failed: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
