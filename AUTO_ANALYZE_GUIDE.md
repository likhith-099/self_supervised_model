# 🚀 FULLY AUTOMATED ENVIRONMENTAL ANALYSIS

## One Command Does Everything!

```bash
python auto_analyze.py --place "PLACE_NAME" --condition CONDITION --start START_DATE --end END_DATE
```

---

## Examples

### Example 1: Vegetation Analysis in Mangalore (2019-2024)

```bash
python auto_analyze.py --place "Mangalore, India" --condition vegetation --start 2019-01-01 --end 2024-01-01
```

**What it does:**
1. ✅ Converts "Mangalore, India" → coordinates automatically
2. ✅ Downloads Sentinel-2 satellite images for that location
3. ✅ Filters clouds and preprocesses images
4. ✅ Generates 128×128 pixel tiles
5. ✅ Trains MAE model (200 epochs)
6. ✅ Extracts feature embeddings
7. ✅ Analyzes vegetation change from 2019 to 2024
8. ✅ Generates report with metrics and visualizations

**Output:**
- `features_mangalore_india_2019.npy`
- `features_mangalore_india_2024.npy`
- `analysis_mangalore_india/vegetation_change/vegetation_2019_vs_2024.png`
- `analysis_mangalore_india/vegetation_change/analysis_results.txt`

---

### Example 2: Water Expansion in Assam

```bash
python auto_analyze.py --place "Assam, India" --condition water --start 2020-01-01 --end 2024-01-01
```

**Analyzes:**
- Flood expansion
- Water body changes
- Drought impact

---

### Example 3: Urban Growth Analysis

```bash
python auto_analyze.py --place "New York, USA" --condition urban --start 2018-06-01 --end 2024-06-01
```

**Analyzes:**
- Urban sprawl
- Built-up area expansion
- Infrastructure development

---

## Available Conditions

| Condition | What It Studies | Use Case |
|-----------|----------------|----------|
| `vegetation` | Forest cover, agricultural expansion | Deforestation monitoring |
| `water` | Water bodies, flooding, drought | Flood mapping, reservoir tracking |
| `urban` | Urban growth, built-up areas | Urban planning |
| `land_degradation` | Soil degradation, desertification | Land restoration |
| `environmental_stress` | Overall ecosystem health | Climate impact assessment |
| `all` | Complete analysis of all conditions | Comprehensive environmental audit |

---

## Command Syntax

```bash
python auto_analyze.py \
    --place "LOCATION_NAME" \
    --condition CONDITION \
    --start YYYY-MM-DD \
    --end YYYY-MM-DD \
    [--timeline YEAR1 YEAR2 ...] \
    [--images NUMBER_OF_IMAGES] \
    [--epochs TRAINING_EPOCHS]
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--place` | Location name (city, region, country) | **Required** |
| `--condition` | Environmental condition to study | **Required** |
| `--start` | Start date (YYYY-MM-DD) | **Required** |
| `--end` | End date (YYYY-MM-DD) | **Required** |
| `--timeline` | Specific years for time-series | Auto-detected |
| `--images` | Number of satellite images | 50 |
| `--epochs` | Training epochs for MAE | 200 |

---

## How It Works

### User Input
```
Place: "Mangalore, India"
Condition: vegetation
Timeline: 2019-01-01 to 2024-01-01
```

↓

### Step 1: Geocoding
- Converts place name → GPS coordinates
- Creates bounding box (~10-50 km radius)
- Output: `[lat, lon, bbox]`

↓

### Step 2: Satellite Image Download
- Searches Copernicus API for Sentinel-2 images
- Filters by location, date range, cloud cover
- Downloads 50+ images

↓

### Step 3: Preprocessing
- Removes cloudy scenes
- Cuts into 128×128 pixel tiles
- Normalizes pixel values (Sentinel-2 specific)

↓

### Step 4: Model Training
- Trains Masked Autoencoder (MAE)
- 200 epochs (~2-4 hours on GPU)
- Saves encoder weights

↓

### Step 5: Feature Extraction
- Passes tiles through trained encoder
- Generates 768-dimensional feature vectors
- Saves as `.npy` files

↓

### Step 6: Environmental Analysis
- Compares features across time periods
- Applies specialized algorithms for selected condition
- Computes change metrics and severity scores

↓

### Final Output
```
✅ Vegetation Change: -7.11% (LOSS, MEDIUM severity)
✅ Visualization: PCA chart showing feature space shift
✅ Report: Detailed metrics and statistics
```

---

## Sample Output Report

```
================================================================================
ENVIRONMENTAL CONDITION ANALYSIS REPORT
================================================================================
Region: mangalore_india
Condition: vegetation
Baseline Year: 2019
Current Year: 2024
Analysis Date: 2024-03-11 16:45:23

================================================================================
VEGETATION ANALYSIS
================================================================================
Similarity: 0.8234
Distance: 12.4567
Baseline Vegetation Coverage: 45.23%
Current Vegetation Coverage: 38.12%
Vegetation Change Percent: -7.11%
Status: LOSS
Severity: MEDIUM
================================================================================
```

---

## Folder Structure After Running

```
ml-engine/
├── data/
│   ├── raw/
│   │   ├── sentinel/          # Downloaded satellite images
│   │   └── filtered/          # Cloud-free images
│   └── tiles/
│       ├── train/             # Training tiles (128×128)
│       └── val/               # Validation tiles
│
├── checkpoints/
│   └── mae_encoder.pth        # Trained model
│
├── features_mangalore_india_2019.npy
├── features_mangalore_india_2024.npy
│
└── analysis_mangalore_india/
    └── vegetation_change/
        ├── vegetation_2019_vs_2024.png    # PCA visualization
        └── analysis_results.txt           # Detailed report
```

---

## Troubleshooting

### Error: "Location not found"
**Solution:** Try more specific location name
```bash
# Instead of "Mangalore"
python auto_analyze.py --place "Mangalore, Karnataka, India" ...
```

### Error: "No satellite images found"
**Solutions:**
1. Expand date range
2. Increase cloud coverage threshold (edit script)
3. Try different season

### Takes too long
**Solution:** Reduce epochs or number of images
```bash
python auto_analyze.py --place "Mangalore, India" ... --epochs 100 --images 30
```

---

## Advanced Usage

### Multi-Year Time Series Analysis

```bash
python auto_analyze.py --place "Assam, India" --condition vegetation \
    --start 2018-01-01 --end 2024-01-01 \
    --timeline 2018 2020 2022 2024
```

### Analyze All Conditions

```bash
python auto_analyze.py --place "Mangalore, India" --condition all \
    --start 2019-01-01 --end 2024-01-01
```

### Custom Output Directory

```bash
python preprocessing/download_sentinel_api.py \
    --place "Mangalore, India" \
    --condition vegetation \
    --start 2019-01-01 \
    --end 2024-01-01 \
    --limit 50 \
    --output /path/to/custom/folder
```

---

## Quick Reference

**Vegetation Monitoring:**
```bash
python auto_analyze.py --place "LOCATION" --condition vegetation --start 2019-01-01 --end 2024-01-01
```

**Water/Flood Analysis:**
```bash
python auto_analyze.py --place "LOCATION" --condition water --start 2020-01-01 --end 2024-01-01
```

**Urban Growth:**
```bash
python auto_analyze.py --place "CITY_NAME" --condition urban --start 2018-01-01 --end 2024-01-01
```

**Complete Environmental Audit:**
```bash
python auto_analyze.py --place "LOCATION" --condition all --start 2019-01-01 --end 2024-01-01
```

---

## Next Steps After Analysis

1. **Review Results:**
   ```bash
   cat analysis_LOCATION/CONDITION_change/analysis_results.txt
   ```

2. **View Visualizations:**
   ```bash
   open analysis_LOCATION/CONDITION_change/*.png
   ```

3. **Compare Multiple Locations:**
   Run analysis for different locations and compare reports

4. **Continuous Monitoring:**
   Set up quarterly or annual updates

---

## Support

For detailed documentation, see:
- `COMPLETE_WORKFLOW.md` - Full workflow guide
- `QUICK_REFERENCE.md` - Command reference
- `API_SETUP.md` - API configuration
