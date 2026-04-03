# Copernicus API Setup Guide

## Where to Configure API Credentials

You have **3 options** to configure your Copernicus API credentials:

### Option 1: Environment Variables (RECOMMENDED)

Set environment variables before running the workflow:

**Windows (PowerShell):**
```powershell
$env:COPERNICUS_USER="your_username"
$env:COPERNICUS_PASSWORD="your_password"
```

**Windows (Command Prompt):**
```cmd
set COPERNICUS_USER=your_username
set COPERNICUS_PASSWORD=your_password
```

**Linux/Mac:**
```bash
export COPERNICUS_USER="your_username"
export COPERNICUS_PASSWORD="your_password"
```

### Option 2: Edit config.py Directly

Open `utils/config.py` and modify lines 45-46:

```python
COPERNICUS_USER = 'your_username_here'
COPERNICUS_PASSWORD = 'your_password_here'
```

### Option 3: Create .env File

Create a `.env` file in the project root:

```
COPERNICUS_USER=your_username
COPERNICUS_PASSWORD=your_password
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

---

## Getting Copernicus API Credentials

### Step 1: Create an Account

1. Go to: https://dataspace.copernicus.eu/
2. Click "Register" in the top right corner
3. Fill in the registration form
4. Verify your email address

### Step 2: Get Your Credentials

1. Log in to https://dataspace.copernicus.eu/
2. Click on your profile name (top right)
3. Select "My Profile"
4. Your **username** is displayed
5. Generate a new **password** for API access:
   - Go to "API Password" section
   - Click "Generate API Password"
   - Copy the generated password

### Step 3: Test Your Credentials

Run this test script:

```bash
python test_api_credentials.py
```

Or directly test the download:

```bash
python preprocessing/download_sentinel.py --bbox 89.5 25.5 96.0 28.0 --start-date 2024-01-01 --end-date 2024-01-31 --cloud-coverage 20.0 --limit 1
```

---

## Alternative Download Methods

If you prefer not to use the API, you can manually download images:

### Manual Download via Browser

1. Go to: https://browser.dataspace.copernicus.eu/
2. Define your area of interest on the map
3. Set filters:
   - Mission: Sentinel-2
   - Product Type: Level-2A
   - Cloud Cover: 0-20%
   - Date Range: Your desired period
4. Download selected products
5. Place downloaded files in: `data/raw/sentinel/`

### Using sentinelsat CLI

Install:
```bash
pip install sentinelsat
```

Query products:
```bash
sentinelsat --user your_username --password your_password \
  --geometry "POLYGON((89.5 25.5, 96.0 25.5, 96.0 28.0, 89.5 28.0, 89.5 25.5))" \
  --date 20240101 TO 20240331 \
  --producttype S2MSI2A \
  --cloud 20
```

---

## Available Regions (Pre-configured)

The following regions are pre-configured in `utils/config.py`:

| Region    | Bounding Box                    | Best Months      | Cloud Threshold |
|-----------|---------------------------------|------------------|-----------------|
| Assam     | [89.5, 25.5, 96.0, 28.0]        | Nov-Mar          | 30%             |
| Mangalore | [74.8, 12.8, 75.1, 13.0]        | Jan-Mar, Nov-Dec | 25%             |
| Amazon    | [-75.0, -10.0, -60.0, 5.0]      | Jun-Sep          | 20%             |
| Congo     | [12.0, -5.0, 30.0, 5.0]         | Dec-Mar          | 25%             |

---

## Troubleshooting

### Error: "Authentication failed"
- Check username/password are correct
- Ensure you're using API password, not website password
- Verify account is activated

### Error: "No products found"
- Try expanding date range
- Increase cloud coverage threshold
- Reduce bounding box size
- Check if area has Sentinel-2 coverage

### Error: "ImportError: No module named 'sentinelsat'"
```bash
pip install sentinelsat
```

### Slow download speeds
- Downloads come from Copernicus servers in Europe
- Consider downloading fewer products at once
- Use offline mode for already downloaded scenes

---

## Data Specifications

**Sentinel-2 Level-2A (Surface Reflectance):**
- Spatial Resolution: 10m, 20m, 60m (depending on band)
- Bands Used: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
- Image Size: ~100 km × 100 km per scene
- Format: GeoTIFF (.jp2 or .tif)
- Coordinate System: UTM/WGS84

**Recommended Settings:**
- Tile Size: 128×128 pixels
- Cloud Coverage: < 20%
- Season: Dry season for your region
- Minimum 50 scenes for good training data
