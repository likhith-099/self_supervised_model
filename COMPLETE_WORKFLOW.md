# Complete Environmental Analysis Workflow

## Overview

This pipeline analyzes **specific environmental conditions** (vegetation, water, urban, land degradation, environmental stress) by comparing satellite imagery from different time periods.

---

## Quick Start Examples

### Example 1: Analyze Vegetation Loss in Assam (2019-2024)

```bash
# Step 1: Generate features for 2019 (includes model training)
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31

# Step 2: Generate features for 2024 (reuses trained model)
python run_full_workflow.py --region assam --start-date 2024-01-01 --end-date 2024-03-31 --skip-training

# Step 3: Analyze vegetation change
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
```

**Result:** Detailed report showing vegetation coverage change %, severity, and PCA visualization

---

### Example 2: Monitor Water Expansion in Mangalore

```bash
# Generate baseline (2020)
python run_full_workflow.py --region mangalore --start-date 2020-01-01 --end-date 2020-03-31

# Generate current (2024) - skip download if using same scenes
python run_full_workflow.py --region mangalore --start-date 2024-01-01 --end-date 2024-03-31 --skip-training

# Analyze water changes
python analyze_condition.py --region mangalore --condition water --year1 2020 --year2 2024
```

**Result:** Water body expansion/contraction analysis with severity classification

---

### Example 3: Comprehensive Multi-Condition Analysis

```bash
# Generate features for multiple years
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31
python run_full_workflow.py --region assam --start-date 2021-01-01 --end-date 2021-03-31 --skip-training
python run_full_workflow.py --region assam --start-date 2024-01-01 --end-date 2024-03-31 --skip-training

# Analyze all conditions
python analyze_condition.py --region assam --condition all --year1 2019 --year2 2024
```

**Result:** Complete environmental report covering all 5 conditions

---

## Available Environmental Conditions

| Condition | What It Detects | Use Cases |
|-----------|----------------|-----------|
| **vegetation** | Forest cover change, agricultural expansion, vegetation health | Deforestation monitoring, reforestation tracking |
| **water** | Water body expansion/contraction, flooding, drought impact | Flood mapping, reservoir monitoring, drought assessment |
| **urban** | Urban sprawl, built-up area growth, infrastructure development | Urban planning, land use change detection |
| **land_degradation** | Soil degradation, desertification, loss of productive capacity | Land restoration planning, conservation efforts |
| **environmental_stress** | Overall ecosystem stress, climate impacts, resilience | Climate change impact assessment, ecosystem health monitoring |

---

## Detailed Workflow Steps

### Phase 1: Data Collection & Model Training

**For First Time Period (e.g., 2019):**

```bash
python run_full_workflow.py --region assam \
    --start-date 2019-01-01 \
    --end-date 2019-03-31 \
    --cloud-threshold 30.0 \
    --tile-size 128 \
    --epochs 400
```

**What happens:**
1. Downloads Sentinel-2 images for 2019
2. Filters cloudy scenes (<30% cloud cover)
3. Generates 128×128 pixel tiles
4. Normalizes pixel values (Sentinel-2 specific)
5. **Trains MAE model** (400 epochs, ~6-12 hours on GPU)
6. Extracts feature embeddings
7. Saves `features_assam_2019.npy`

**Output Files:**
- `checkpoints/mae_encoder.pth` - Trained encoder
- `features_assam_2019.npy` - Feature vectors
- `data/tiles/train/` - Training tiles
- `data/tiles/val/` - Validation tiles

---

### Phase 2: Additional Time Periods

**For Subsequent Periods (e.g., 2024):**

```bash
python run_full_workflow.py --region assam \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --skip-training  # Reuse trained model
```

**What happens:**
1. Downloads Sentinel-2 images for 2024
2. Applies same preprocessing as 2019
3. Uses **already trained encoder** to extract features
4. Saves `features_assam_2024.npy`

**Note:** Skip training (`--skip-training`) ensures consistent feature extraction across years

---

### Phase 3: Environmental Condition Analysis

**Run Specific Analysis:**

```bash
python analyze_condition.py --region assam \
    --condition vegetation \
    --year1 2019 \
    --year2 2024
```

**What it does:**
1. Loads `features_assam_2019.npy` and `features_assam_2024.npy`
2. Applies specialized algorithms for the selected condition
3. Computes change metrics and severity scores
4. Generates PCA visualizations
5. Saves detailed report

**Output:**
- `analysis_assam/vegetation_change/vegetation_2019_vs_2024.png`
- `analysis_assam/vegetation_change/analysis_results.txt`

---

## Command Reference

### Full Workflow Commands

**Complete workflow (first time period):**
```bash
python run_full_workflow.py --region <REGION> --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

**Complete workflow (subsequent periods, skip training):**
```bash
python run_full_workflow.py --region <REGION> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --skip-training
```

**Windows batch:**
```cmd
run_workflow.bat <REGION> <START_DATE> <END_DATE>
```

**Linux/Mac shell:**
```bash
./run_workflow.sh <REGION> <START_DATE> <END_DATE>
```

---

### Condition Analysis Commands

**Vegetation:**
```bash
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
```

**Water:**
```bash
python analyze_condition.py --region mangalore --condition water --year1 2020 --year2 2024
```

**Urban:**
```bash
python analyze_condition.py --region assam --condition urban --year1 2019 --year2 2024
```

**Land Degradation:**
```bash
python analyze_condition.py --region assam --condition land_degradation --year1 2019 --year2 2024
```

**Environmental Stress:**
```bash
python analyze_condition.py --region assam --condition environmental_stress --year1 2019 --year2 2024
```

**All Conditions:**
```bash
python analyze_condition.py --region assam --condition all --year1 2019 --year2 2024
```

---

## Regional Configurations

### Pre-configured Regions

**Assam, India**
```bash
--region assam
--start-date 2019-01-01 --end-date 2019-03-31  # Best: Nov-Mar
```
- Bounding Box: [89.5, 25.5, 96.0, 28.0]
- Cloud Threshold: 30%
- Best Months: Nov-Mar (dry season)

**Mangalore, India**
```bash
--region mangalore
--start-date 2019-01-01 --end-date 2019-03-31  # Best: Jan-Mar, Nov-Dec
```
- Bounding Box: [74.8, 12.8, 75.1, 13.0]
- Cloud Threshold: 25%
- Best Months: Jan-Mar, Nov-Dec

**Amazon Rainforest**
```bash
--region amazon
--start-date 2019-06-01 --end-date 2019-09-30  # Best: Jun-Sep
```
- Bounding Box: [-75.0, -10.0, -60.0, 5.0]
- Cloud Threshold: 20%
- Best Months: Jun-Sep (dry season)

**Congo Basin**
```bash
--region congo
--start-date 2019-12-01 --end-date 2020-03-31  # Best: Dec-Mar
```
- Bounding Box: [12.0, -5.0, 30.0, 5.0]
- Cloud Threshold: 25%
- Best Months: Dec-Mar

---

## Output Structure

```
ml-engine/
│
├── checkpoints/
│   └── mae_encoder.pth              # Trained model weights
│
├── features_assam_2019.npy          # Feature embeddings (N × 768)
├── features_assam_2024.npy
│
├── data/
│   ├── raw/sentinel/                # Downloaded GeoTIFFs
│   └── tiles/train/                 # 128×128 training tiles
│
└── analysis_assam/
    ├── vegetation_change/
    │   ├── vegetation_2019_vs_2024.png      # PCA visualization
    │   └── analysis_results.txt             # Detailed metrics
    │
    ├── water_change/
    │   ├── water_2019_vs_2024.png
    │   └── analysis_results.txt
    │
    ├── urban_change/
    │   ├── urban_2019_vs_2024.png
    │   └── analysis_results.txt
    │
    ├── land_degradation_change/
    │   ├── land_degradation_2019_vs_2024.png
    │   └── analysis_results.txt
    │
    └── environmental_stress_change/
        ├── environmental_stress_2019_vs_2024.png
        └── analysis_results.txt
```

---

## Understanding Analysis Results

### Vegetation Analysis Output

```
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

**Interpretation:**
- **Vegetation Change %**: -7.11% = 7.11% loss in vegetation coverage
- **Status**: LOSS indicates deforestation or vegetation decline
- **Severity**: MEDIUM = significant but not critical (10-20% change)

---

### Water Analysis Output

```
================================================================================
WATER ANALYSIS
================================================================================
Baseline Water Coverage: 12.45%
Current Water Coverage: 15.67%
Water Change Percent: 3.22%
Expansion Ratio: 25.86%
Status: EXPANSION
Severity: MEDIUM
================================================================================
```

**Interpretation:**
- **Water Change %**: +3.22% = absolute increase in water area
- **Expansion Ratio**: 25.86% = relative increase (3.22/12.45)
- **Status**: EXPANSION indicates flooding or new water bodies

---

### Urban Growth Output

```
================================================================================
URBAN ANALYSIS
================================================================================
Baseline Urban Coverage: 23.45%
Current Urban Coverage: 31.23%
Urban Change Percent: 7.78%
Growth Rate: 33.18%
Status: EXPANDING
Severity: HIGH
================================================================================
```

**Interpretation:**
- **Urban Change %**: +7.78% = significant urban sprawl
- **Growth Rate**: 33.18% = rapid expansion relative to baseline
- **Severity**: HIGH = major urbanization (>20% change)

---

### Land Degradation Output

```
================================================================================
LAND DEGRADATION ANALYSIS
================================================================================
Vegetation Loss Percent: 12.34%
Diversity Change Percent: -8.56%
Feature Shift Magnitude: 15.6789
Homogenization Index: -2.00
Degradation Score: 8.45
Status: DEGRADING
Severity: MODERATE
================================================================================
```

**Interpretation:**
- **Degradation Score**: 8.45 = moderate degradation (composite index)
- **Status**: DEGRADING = declining land quality
- **Severity**: MODERATE = requires attention but not critical

---

### Environmental Stress Output

```
================================================================================
ENVIRONMENTAL STRESS ANALYSIS
================================================================================
Ecosystem Shift: 0.1523
Variability Change Percent: 18.45%
Outlier Frequency Change: 0.0523
Resilience Change: -0.1234
Stress Index: 12.67
Status: MODERATE_STRESS
Severity: MODERATE
================================================================================
```

**Interpretation:**
- **Stress Index**: 12.67 = moderate ecosystem stress
- **Status**: MODERATE_STRESS = detectable stress signals
- **Resilience Change**: Negative = reduced ecosystem recovery capacity

---

## Troubleshooting

### Issue: "Features file not found"

**Solution:** Generate features first
```bash
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31
```

---

### Issue: "Model checkpoint not found"

**Solution:** Remove `--skip-training` flag for first run
```bash
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31
```

---

### Issue: "No products found" in download

**Solutions:**
1. Expand date range
2. Increase cloud threshold
3. Check bounding box coordinates
4. Try different season

---

### Issue: Analysis takes too long

**Solution:** Reduce sample size
```bash
# Edit script to use max_samples parameter
python inference/extract_features.py --max-samples 1000
```

---

## Advanced Usage

### Custom Date Ranges

```bash
python analyze_condition.py --region assam \
    --condition vegetation \
    --year1 2019 --year2 2024 \
    --date-range1 2019-11-01 2019-12-31 \
    --date-range2 2024-11-01 2024-12-31
```

---

### Multi-Year Time Series

```bash
# Generate features for consecutive years
for year in 2019 2020 2021 2022 2023 2024; do
    python run_full_workflow.py --region assam \
        --start-date ${year}-01-01 --end-date ${year}-03-31 \
        --skip-training
done

# Analyze trend over time
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
```

---

### Batch Processing Multiple Regions

Create `batch_analysis.sh`:
```bash
#!/bin/bash

regions=("assam" "mangalore")
conditions=("vegetation" "water" "urban")

for region in "${regions[@]}"; do
    for condition in "${conditions[@]}"; do
        echo "Analyzing $condition in $region..."
        python analyze_condition.py --region $region \
            --condition $condition \
            --year1 2019 --year2 2024
    done
done
```

---

## Performance Benchmarks

| Task | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Download 100 scenes | 30-60 min | 30-60 min |
| Cloud filtering | 5-10 min | 5-10 min |
| Tiling | 10-15 min | 10-15 min |
| Normalization | 2-3 min | 2-3 min |
| **Training (400 epochs)** | **6-12 hours** | **2-3 days** |
| Feature extraction | 5-10 min | 10-20 min |
| Condition analysis | 1-2 min | 1-2 min |

---

## Next Steps After Analysis

1. **Validate findings** with ground truth data
2. **Cross-reference** with known events (floods, fires, policy changes)
3. **Quantify affected areas** using GIS tools
4. **Generate reports** for stakeholders
5. **Set up continuous monitoring** (quarterly/annual updates)
6. **Integrate** with other data sources (climate, socioeconomic)
