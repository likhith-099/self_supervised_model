"""
Geocoding service - Converts place names to coordinates
Uses Nominatim (OpenStreetMap) API for geocoding
"""

import requests
from typing import Tuple, Optional, Dict


class Geocoder:
    """Convert place names to geographic coordinates"""
    
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {
            'User-Agent': 'SatelliteAnalysis/1.0'
        }
    
    def get_coordinates(self, place_name: str) -> Optional[Dict]:
        """
        Get coordinates for a place name
        
        Args:
            place_name: Location name (e.g., "Mangalore, India")
        
        Returns:
            Dictionary with lat, lon, bounding_box or None if not found
        """
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            results = response.json()
            
            if not results:
                print(f"Location '{place_name}' not found!")
                return None
            
            result = results[0]
            
            # Extract basic info
            lat = float(result['lat'])
            lon = float(result['lon'])
            display_name = result['display_name']
            
            # Create bounding box (default ~10km radius)
            bbox = self._create_bbox(lat, lon, size_km=10)
            
            location_info = {
                'name': place_name,
                'display_name': display_name,
                'lat': lat,
                'lon': lon,
                'bbox': bbox,
                'type': result.get('type', 'unknown')
            }
            
            print(f"✓ Found: {display_name}")
            print(f"  Coordinates: {lat:.4f}, {lon:.4f}")
            print(f"  Bounding Box: {bbox}")
            
            return location_info
            
        except Exception as e:
            print(f"Error geocoding '{place_name}': {e}")
            return None
    
    def _create_bbox(self, lat: float, lon: float, size_km: float = 10) -> list:
        """
        Create bounding box around a point
        
        Args:
            lat: Latitude
            lon: Longitude
            size_km: Size in kilometers (radius)
        
        Returns:
            Bounding box [min_lon, min_lat, max_lon, max_lat]
        """
        # Approximate degrees per km
        deg_per_km_lat = 0.009  # 1 degree ≈ 111 km
        deg_per_km_lon = 0.009 / abs(np.cos(np.radians(lat)))
        
        delta_lat = size_km * deg_per_km_lat
        delta_lon = size_km * deg_per_km_lon
        
        min_lon = lon - delta_lon
        min_lat = lat - delta_lat
        max_lon = lon + delta_lon
        max_lat = lat + delta_lat
        
        return [round(min_lon, 4), round(min_lat, 4), 
                round(max_lon, 4), round(max_lat, 4)]
    
    def get_region_coordinates(self, region_name: str) -> Optional[Dict]:
        """
        Get coordinates for predefined regions
        
        Args:
            region_name: Region name (assam, mangalore, amazon, congo)
        
        Returns:
            Location info dictionary
        """
        # Predefined regions
        regions = {
            'assam': {
                'query': 'Assam, India',
                'bbox': [89.5, 25.5, 96.0, 28.0],
                'description': 'Assam, India'
            },
            'mangalore': {
                'query': 'Mangalore, Karnataka, India',
                'bbox': [74.8, 12.8, 75.1, 13.0],
                'description': 'Mangalore, Karnataka, India'
            },
            'amazon': {
                'query': 'Amazon Rainforest, Brazil',
                'bbox': [-75.0, -10.0, -60.0, 5.0],
                'description': 'Amazon Rainforest'
            },
            'congo': {
                'query': 'Congo Basin, Africa',
                'bbox': [12.0, -5.0, 30.0, 5.0],
                'description': 'Congo Basin'
            }
        }
        
        region_lower = region_name.lower()
        
        if region_lower in regions:
            region_data = regions[region_lower]
            
            # Try to get precise coordinates
            location_info = self.get_coordinates(region_data['query'])
            
            if location_info:
                # Override with predefined bbox for larger coverage
                location_info['bbox'] = region_data['bbox']
                location_info['description'] = region_data['description']
                return location_info
        
        # Not in predefined list, try general geocoding
        return self.get_coordinates(region_name)


def test_geocoder():
    """Test geocoding functionality"""
    geocoder = Geocoder()
    
    test_locations = [
        "Mangalore, India",
        "Assam, India",
        "New York, USA",
        "London, UK"
    ]
    
    for location in test_locations:
        print(f"\nSearching: {location}")
        print("-" * 60)
        result = geocoder.get_coordinates(location)
        if result:
            print(f"Success!\n")
        else:
            print(f"Failed!\n")


if __name__ == '__main__':
    import numpy as np
    test_geocoder()
