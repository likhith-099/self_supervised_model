# ML Engineering Pipeline for Satellite Imagery

Self-supervised learning pipeline using Masked Autoencoders (MAE) for satellite imagery analysis.

## Folder Structure

```
ml-engine/
│
├── data/
│   ├── raw/                     # Downloaded satellite scenes
│   │   ├── sentinel/            # Sentinel-2 imagery
│   │   └── landsat/             # Landsat imagery
│   │
│   ├── tiles/                   # Tiles for training
│   │   ├── train/               # Training tiles
│   │   └── val/                 # Validation tiles
│   │
│   └── region_data/             # Processed regional data
│
├── preprocessing/
│   ├── download_sentinel.py     # Download satellite images
│   ├── cloud_filter.py          # Remove cloudy images
│   ├── tile_generator.py        # Generate training tiles
│   └── normalize.py             # Pixel normalization
│
├── models/
│   ├── mae_model.py             # MAE architecture
│   └── encoder.py               # Encoder loader
│
├── training/
│   ├── dataset_loader.py        # Data loading utilities
│   └── train_mae.py             # MAE training script
│
├── inference/
│   ├── extract_features.py      # Extract feature vectors
│   └── analyze_region.py        # Regional analysis
│
├── utils/
│   ├── config.py                # Configuration management
│   └── visualization.py         # Visualization utilities
│
├── checkpoints/                 # Saved model weights
│
└── requirements.txt             # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

### Configure Copernicus API (Required)

**See detailed instructions in [API_SETUP.md](API_SETUP.md)**

Quick setup (choose one method):

**Method 1: Environment Variables**
```bash
# Windows PowerShell
$env:COPERNICUS_USER="your_username"
$env:COPERNICUS_PASSWORD="your_password"

# Linux/Mac
export COPERNICUS_USER="your_username"
export COPERNICUS_PASSWORD="your_password"
```

**Method 2: Edit config.py**
Open `utils/config.py` and set:
```python
COPERNICUS_USER = 'your_username'
COPERNICUS_PASSWORD = 'your_password'
```

**Get credentials:** Register at https://dataspace.copernicus.eu/

## Quick Start

### Automated Workflow (Recommended)

**Windows:**
```cmd
run_workflow.bat assam 2024-01-01 2024-03-31
```

**Linux/Mac:**
```bash
./run_workflow.sh assam 2024-01-01 2024-03-31
```

This runs all 6 steps automatically!

---

### Analyze Environmental Conditions

After generating feature files, analyze specific conditions:

**Vegetation Change:**
```bash
python analyze_condition.py --region assam --condition vegetation --year1 2019 --year2 2024
```

**Water Expansion:**
```bash
python analyze_condition.py --region mangalore --condition water --year1 2020 --year2 2024
```

**Urban Growth:**
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

### Manual Step-by-Step

### 1. Download Satellite Data

```bash
python preprocessing/download_sentinel.py \
    --output data/raw/sentinel \
    --bbox 89.5 25.5 96.0 28.0 \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --cloud-coverage 20.0
```

### 2. Filter Cloudy Images

```bash
python preprocessing/cloud_filter.py \
    --input data/raw/sentinel \
    --output data/raw/filtered \
    --max-cloud 20.0
```

### 3. Generate Training Tiles

```bash
python preprocessing/tile_generator.py \
    --input data/raw/filtered \
    --output data/tiles \
    --tile-size 256 \
    --train-ratio 0.8
```

### 4. Normalize Tiles

```bash
python preprocessing/normalize.py \
    --input data/tiles/train \
    --output data/tiles_normalized \
    --method zscore \
    --stats-from data/tiles/train
```

### 5. Train MAE Model

```bash
python training/train_mae.py \
    --train-dir data/tiles/train \
    --val-dir data/tiles/val \
    --batch-size 64 \
    --epochs 400 \
    --lr 1e-4 \
    --device cuda
```

### 6. Extract Features

```bash
python inference/extract_features.py \
    --checkpoint checkpoints/mae_encoder.pth \
    --input data/tiles/val \
    --output features.npy \
    --device cuda
```

### 7. Analyze Region

```bash
python inference/analyze_region.py \
    --features features.npy \
    --n-clusters 5 \
    --tsne \
    --output-dir analysis_output
```

## Configuration

Edit `utils/config.py` to customize:

- Model hyperparameters
- Training parameters
- Data paths
- Region-specific settings

## Pre-trained Models

Download pre-trained MAE encoders and place them in `checkpoints/`:

- [MAE Encoder for Sentinel-2](#) (coming soon)
- [MAE Encoder for Landsat](#) (coming soon)

## Example Notebooks

See example notebooks in the `examples/` directory (not included in this structure).

## License

MIT License
