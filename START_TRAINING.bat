@echo off
REM ================================================
REM SIMPLE ONE-CLICK MODEL TRAINING
REM ================================================
REM Just run this file and it will do everything!

echo ================================================================================
echo STARTING MODEL TRAINING
echo ================================================================================
echo.
echo This will:
echo   1. Filter cloudy images
echo   2. Generate 128x128 pixel tiles
echo   3. Normalize pixel values
echo   4. Train MAE model (2-4 hours)
echo   5. Extract features
echo   6. Generate analysis visualizations
echo.
echo Estimated time: 3-5 hours total
echo You can leave it running and come back later!
echo.
echo Press any key to start...
pause >nul
echo.

REM Step 1: Filter clouds
echo [STEP 1/6] Filtering cloudy images...
python preprocessing/cloud_filter.py --input data/raw/sentinel --output data/raw/filtered --max-cloud 30.0
if errorlevel 1 (
    echo ERROR: Cloud filtering failed!
    pause
    exit /b 1
)
echo.

REM Step 2: Generate tiles
echo [STEP 2/6] Generating 128x128 pixel tiles...
python preprocessing/tile_generator.py --input data/raw/filtered --output data/tiles --tile-size 128 --train-ratio 0.8
if errorlevel 1 (
    echo ERROR: Tile generation failed!
    pause
    exit /b 1
)
echo.

REM Step 3: Normalize
echo [STEP 3/6] Normalizing pixel values...
python preprocessing/normalize.py --input data/tiles/train --output data/tiles_normalized --method sentinel2
if errorlevel 1 (
    echo ERROR: Normalization failed!
    pause
    exit /b 1
)
echo.

REM Step 4: Train model
echo [STEP 4/6] Training MAE model...
echo This will take 2-4 hours. You can minimize this window and do other work.
python training/train_mae.py --train-dir data/tiles/train --val-dir data/tiles/val --batch-size 64 --epochs 200 --lr 1e-4 --img-size 128 --device cuda
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)
echo.

REM Step 5: Extract features
echo [STEP 5/6] Extracting environmental features...
python inference/extract_features.py --checkpoint checkpoints/mae_encoder.pth --input data/tiles/val --output features_analysis.npy --device cuda
if errorlevel 1 (
    echo ERROR: Feature extraction failed!
    pause
    exit /b 1
)
echo.

REM Step 6: Analyze
echo [STEP 6/6] Generating analysis and visualizations...
python inference/map_visualization.py --baseline features_analysis.npy --current features_analysis.npy --condition vegetation --output analysis_report.png
if errorlevel 1 (
    echo ERROR: Analysis failed!
    pause
    exit /b 1
)
echo.

echo ================================================================================
echo TRAINING COMPLETE!
echo ================================================================================
echo.
echo Your model is trained and ready!
echo.
echo Results saved to:
echo   - Model: checkpoints/mae_encoder.pth
echo   - Features: features_analysis.npy
echo   - Visualizations: analysis_report.png
echo.
echo Next steps:
echo   1. View analysis_report.png to see visualizations
echo   2. Run: python analyze_condition.py --region mangalore --condition vegetation --year1 2024 --year2 2025
echo.
echo To train again with different data, just run this file again!
echo.
pause
