"""
Test script for Copernicus API credentials
"""

import os
from utils.config import Config

def test_credentials():
    """Test if API credentials are configured"""
    
    print("=" * 60)
    print("COPERNICUS API CREDENTIALS TEST")
    print("=" * 60)
    
    # Check credentials
    user = os.getenv('COPERNICUS_USER', Config.COPERNICUS_USER)
    password = os.getenv('COPERNICUS_PASSWORD', Config.COPERNICUS_PASSWORD)
    
    print(f"\nUsername: {user if user else 'NOT SET'}")
    print(f"Password: {'***' if password else 'NOT SET'}")
    
    if not user or not password:
        print("\n❌ ERROR: Credentials not configured!")
        print("\nTo fix this:")
        print("1. Register at: https://dataspace.copernicus.eu/")
        print("2. Get your API credentials from My Profile")
        print("3. Set environment variables:")
        print("   Windows: $env:COPERNICUS_USER='your_username'")
        print("            $env:COPERNICUS_PASSWORD='your_password'")
        print("   Linux:   export COPERNICUS_USER='your_username'")
        print("            export COPERNICUS_PASSWORD='your_password'")
        print("\nOr edit utils/config.py and add your credentials there.")
        return False
    
    if user == 'your_username_here':
        print("\n❌ ERROR: Using default placeholder credentials!")
        print("Please edit utils/config.py with your actual credentials.")
        return False
    
    # Try to connect
    try:
        from sentinelsat import SentinelAPI
        
        print("\nTesting connection to Copernicus API...")
        api = SentinelAPI(user, password, 'https://apihub.copernicus.eu/apihub')
        
        # Test query
        products = api.query(
            'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            date=('2024-01-01', '2024-01-02'),
            platformname='Sentinel-2',
            producttype='S2MSI2A'
        )
        
        print(f"✓ Connection successful!")
        print(f"✓ Query returned {len(products)} products (test)")
        print("\n✅ Credentials are valid and working!")
        
        api.logout()
        return True
        
    except ImportError:
        print("\n⚠ WARNING: sentinelsat not installed!")
        print("Install with: pip install sentinelsat")
        print("\nCredentials appear to be set, but cannot test connection.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPossible issues:")
        print("- Incorrect username or password")
        print("- Account not activated")
        print("- Network connectivity issues")
        print("- API service temporarily unavailable")
        return False


if __name__ == '__main__':
    success = test_credentials()
    exit(0 if success else 1)
