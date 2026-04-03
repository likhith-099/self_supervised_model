# MAE Training on Google Colab - Complete Guide

## Step-by-Step Instructions

### 1. Go to Google Colab
- Visit: https://colab.research.google.com/
- Click "New Notebook"

### 2. Enable GPU Runtime
- Click: **Runtime** → **Change runtime type**
- Select: **GPU** under Hardware accelerator
- Click: **Save**

### 3. Upload Your Code
Copy and paste this entire code into the notebook cell, then click the play button ▶️

```python
# ============================================
# COMPLETE MAE TRAINING ON COLAB
# ============================================

# Step 1: Mount Google Drive (optional, for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Clone your repository or upload files
!git clone https://github.com/your-username/your-repo.git
# OR upload files manually using the file upload button

# Step 3: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install albumentations pillow tensorboard

# Step 4: Verify GPU is working
!nvidia-smi

# Step 5: Check CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Step 6: Run training
!python training/train_mae.py --train-dir data/tiles/train --val-dir data/tiles/val --batch-size 64 --epochs 100 --img-size 128 --device cuda
```

### 4. Alternative: Use This Pre-made Notebook
I can create a complete .ipynb file for you to download and upload to Colab.

---

## Expected Results on Colab:

| Metric | Colab (Free T4 GPU) | Your CPU |
|--------|---------------------|----------|
| **Time per epoch** | ~15-20 minutes | ~36 hours |
| **Total for 100 epochs** | **~25-35 hours** | ~150 days |
| **GPU** | NVIDIA T4 (16GB) | None |
| **Cost** | FREE | Electricity 😅 |

---

## Quick Start Command (after enabling GPU):

If GPU gets enabled locally, just run:
```powershell
cd e:\major_ml\ml-engine
python training/fast_train_mae.py
```

This will automatically use GPU if available!
