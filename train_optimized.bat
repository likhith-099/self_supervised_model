@echo off
echo ========================================
echo Starting Optimized MAE Training
echo ========================================
echo.
echo GPU: NVIDIA GeForce 940MX
echo Batch Size: 16 (optimized for your GPU)
echo Image Size: 128x128
echo Epochs: 50
echo Mixed Precision: ENABLED (faster training)
echo Checkpoints: Every 500 batches
echo.
echo Starting training...
echo.

cd /d "e:\major_ml\ml-engine"

python training/train_mae.py ^
    --train-dir data/tiles/train ^
    --val-dir data/tiles/val ^
    --batch-size 16 ^
    --epochs 50 ^
    --img-size 128 ^
    --device cuda ^
    --workers 2 ^
    --lr 1e-4 ^
    --checkpoint-dir checkpoints

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
