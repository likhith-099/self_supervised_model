# Quick Reference: Environmental Condition Analysis

## Workflow Overview

```
1. Download Data → 2. Train Model → 3. Extract Features → 4. Analyze Conditions
```

---

## Step 1: Initial Setup (One-time)

### Configure API Credentials
```bash
# Windows PowerShell
$env:COPERNICUS_USER="your_username"
$env:COPERNICUS_PASSWORD="your_password"

# Test credentials
python test_api_credentials.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Step 2: Generate Feature Files (First Time for Each Year)

### For Year 2019
```bash
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31
```
**Output:** `features_assam_2019.npy`

### For Year 2024
```bash
python run_full_workflow.py --region assam --start-date 2024-01-01 --end-date 2024-03-31 --skip-training
```
**Output:** `features_assam_2024.npy`

> **Note:** Use `--skip-training` after first run (model already trained)

---

## Step 3: Analyze Specific Environmental Conditions

### 🌿 Vegetation Analysis
```bash
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
```

**Detects:**
- Forest cover loss/gain
- Agricultural expansion
- Vegetation health changes

**Outputs:**
- `analysis_assam/vegetation_change/vegetation_2019_vs_2024.png`
- `analysis_assam/vegetation_change/analysis_results.txt`

---

### 💧 Water Expansion Analysis
```bash
python analyze_condition.py --region mangalore --condition water --year1 2020 --year2 2024
```

**Detects:**
- Water body expansion/contraction
- Flooding events
- Drought impact on water resources

---

### 🏙️ Urban Growth Analysis
```bash
python analyze_condition.py --region assam --condition urban --year1 2019 --year2 2024
```

**Detects:**
- Urban area expansion
- Built-up land increase
- Infrastructure development

---

### 🏜️ Land Degradation Analysis
```bash
python analyze_condition.py --region assam --condition land_degradation --year1 2019 --year2 2024
```

**Detects:**
- Soil degradation
- Desertification
- Loss of productive land capacity

---

### 🌡️ Environmental Stress Analysis
```bash
python analyze_condition.py --region assam --condition environmental_stress --year1 2019 --year2 2024
```

**Detects:**
- Overall ecosystem stress
- Climate change impacts
- Ecosystem resilience changes

---

### 🔍 Comprehensive Analysis (All Conditions)
```bash
python analyze_condition.py --region assam --condition all --year1 2019 --year2 2024
```

**Generates:** Complete report for all 5 conditions

---

## Available Regions

| Region | Command | Best Analysis Period |
|--------|---------|---------------------|
| **Assam** | `--region assam` | Jan-Mar (dry season) |
| **Mangalore** | `--region mangalore` | Jan-Mar, Nov-Dec |
| **Amazon** | `--region amazon` | Jun-Sep (dry season) |
| **Congo** | `--region congo` | Dec-Mar |

---

## Output Structure

After analysis, results organized as:

```
ml-engine/
├── features_assam_2019.npy          # Feature embeddings
├── features_assam_2024.npy
│
└── analysis_assam/
    ├── vegetation_change/
    │   ├── vegetation_2019_vs_2024.png    # PCA visualization
    │   └── analysis_results.txt           # Detailed metrics
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

## Understanding Results

### Vegetation Analysis Metrics
- **Vegetation Change %**: Positive = gain, Negative = loss
- **Severity**: LOW (<10%), MEDIUM (10-20%), HIGH (>20%)
- **Status**: GAIN / STABLE / LOSS

### Water Analysis Metrics
- **Water Change %**: Positive = expansion, Negative = contraction
- **Expansion Ratio**: Percentage change relative to baseline
- **Status**: EXPANSION / STABLE / CONTRACTION

### Urban Analysis Metrics
- **Urban Change %**: Positive = growth, Negative = decline
- **Growth Rate**: Annual growth percentage
- **Status**: EXPANDING / STABLE / DECLINING

### Land Degradation Metrics
- **Degradation Score**: Composite index (higher = worse)
- **Severity**: LOW / MODERATE / HIGH / CRITICAL
- **Status**: DEGRADING / STABLE / IMPROVING

### Environmental Stress Metrics
- **Stress Index**: Composite stress indicator
- **Severity**: LOW / MODERATE / HIGH / CRITICAL
- **Status**: LOW_STRESS / MODERATE_STRESS / HIGH_STRESS

---

## Example Analysis Session

### Scenario: Assess deforestation in Assam (2019-2024)

```bash
# Step 1: Generate 2019 features (full workflow with training)
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31

# Step 2: Generate 2024 features (skip training, reuse model)
python run_full_workflow.py --region assam --start-date 2024-01-01 --end-date 2024-03-31 --skip-training

# Step 3: Analyze vegetation change
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
```

**Expected Output:**
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

---

## Batch Processing Multiple Regions

Create a batch script (`batch_analysis.bat`):

```batch
@echo off
REM Batch analysis for multiple regions

echo Analyzing Assam...
python analyze_condition.py --region assam --condition all --year1 2019 --year2 2024

echo Analyzing Mangalore...
python analyze_condition.py --region mangalore --condition all --year1 2019 --year2 2024

echo All analyses complete!
pause
```

---

## Troubleshooting

### Error: "Features file not found"
**Solution:** Run workflow to generate features first
```bash
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31
```

### Error: "Model checkpoint not found"
**Solution:** Train model first (remove `--skip-training` flag)
```bash
python run_full_workflow.py --region assam --start-date 2019-01-01 --end-date 2019-03-31
```

### Analysis takes too long
**Solution:** Reduce number of samples or use fewer clusters
```bash
# Edit analyze_condition.py, add: --max-samples 1000
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

### Compare Multiple Years
```bash
# 2019 vs 2020
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2020

# 2020 vs 2021
python analyze_condition.py --region assam --condition vegetation --year1 2020 --year2 2021

# 2021 vs 2024
python analyze_condition.py --region assam --condition vegetation --year1 2021 --year2 2024
```

---

## Export Results for Reporting

Results can be exported to CSV or Excel:

```python
import pandas as pd
import numpy as np

# Load results
results = {
    '2019': np.load('features_assam_2019.npy'),
    '2024': np.load('features_assam_2024.npy')
}

# Compute statistics
stats = []
for year, features in results.items():
    stats.append({
        'Year': year,
        'Mean': features.mean(),
        'Std': features.std(),
        'Samples': len(features)
    })

# Save to CSV
df = pd.DataFrame(stats)
df.to_csv('analysis_assam/summary_statistics.csv', index=False)
```
