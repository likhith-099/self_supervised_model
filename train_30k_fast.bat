@echo off
echo ========================================
echo Fast MAE Training - 30k Dataset
echo ========================================
echo.
echo GPU: NVIDIA GeForce 940MX
echo Dataset: 30,000 images (subset)
echo Batch Size: 16
echo Image Size: 128x128
echo Epochs: 50
echo Mixed Precision: ENABLED
echo Checkpoints: Every 500 batches
echo.
echo Estimated time: ~1.7 days (vs 19 days for full dataset)
echo.
echo Starting training...
echo.

cd /d "e:\major_ml\ml-engine"

python training/train_mae.py ^
    --train-dir data/tiles/train_30k ^
    --val-dir data/tiles/val ^
    --batch-size 16 ^
    --epochs 50 ^
    --img-size 128 ^
    --device cuda ^
    --workers 2 ^
    --lr 1e-4 ^
    --checkpoint-dir checkpoints_30k

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
