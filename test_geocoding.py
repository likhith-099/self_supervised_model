"""
Test geocoding for any location
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.visual_report import get_coordinates

# Test with various locations
test_locations = [
    "Bangalore, India",
    "New York, USA",
    "London, UK",
    "Tokyo, Japan",
    "Sydney, Australia",
    "Paris, France",
    "Dubai, UAE",
    "Mumbai, India",
    "Delhi, India",
    "Chennai, India"
]

print("=" * 80)
print("🌍 TESTING GEOCODING FOR VARIOUS LOCATIONS")
print("=" * 80)

for location in test_locations:
    print(f"\n📍 Testing: {location}")
    print("-" * 60)
    coords = get_coordinates(location)
    if coords['found']:
        print(f"   ✅ SUCCESS: Lat={coords['lat']:.4f}, Lon={coords['lon']:.4f}")
    else:
        print(f"   ❌ FAILED")

print("\n" + "=" * 80)
print("✅ GEOCODING TEST COMPLETE")
print("=" * 80)
