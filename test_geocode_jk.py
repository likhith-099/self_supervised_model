"""Test geocoding for Jammu & Kashmir"""
from utils.geocoder import Geocoder

g = Geocoder()
result = g.get_region_coordinates('jammu_kashmir')

if result:
    print(f'\n✅ SUCCESS!')
    print(f'Location: {result["display_name"]}')
    print(f'Coordinates: {result["lat"]}, {result["lon"]}')
    print(f'BBox: {result["bbox"]}')
else:
    print('\n❌ Failed to geocode Jammu & Kashmir')
