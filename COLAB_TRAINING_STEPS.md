# 🚀 Complete Guide: Training MAE Model on Google Colab

## Overview
This guide walks you through training your Masked Autoencoder (MAE) model for satellite imagery analysis using **FREE GPU** on Google Colab.

---

## 📋 Prerequisites

1. ✅ GitHub account
2. ✅ Google account (for Colab and Drive)
3. ✅ Your satellite image dataset (tiles in PNG/JPG format)
4. ✅ ~25-35 hours of time (can run in background)

---

## 🔧 Step-by-Step Instructions

### **STEP 1: Open Google Colab**

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **"GitHub"** tab
3. Enter your repository URL: `https://github.com/SnehaShetty-18/ml_project`
4. Navigate to: `ml-engine/TRAINING_COLAB_GUIDE.ipynb`
5. Click to open the notebook

**OR** upload the notebook manually:
- Click **File → Upload notebook**
- Select `TRAINING_COLAB_GUIDE.ipynb` from your computer

---

### **STEP 2: Enable GPU (CRITICAL!)**

⚠️ **DO THIS BEFORE RUNNING ANY CELLS:**

1. Click **Runtime** menu at the top
2. Select **Change runtime type**
3. Under "Hardware accelerator", select **GPU**
4. Click **Save**
5. Wait for the runtime to restart (you'll see a green checkmark)

✅ You now have FREE access to NVIDIA T4 GPU!

---

### **STEP 3: Run Cell by Cell**

Follow the notebook cells in order. Here's what each step does:

#### **Cell 1: Install Dependencies**
```python
# Just click the play button (▶️) on the left
# This installs PyTorch, torchvision, and other libraries
# Takes ~2-3 minutes
```

#### **Cell 2: Verify GPU**
```python
# Confirms GPU is working
# You should see:
# - CUDA Available: True
# - GPU Device: Tesla T4
```

#### **Cell 3: Mount Google Drive**
```python
# Click the link that appears
# Sign in with your Google account
# Copy the authorization code
# Paste it in the box and press Enter
# This lets you save results to Google Drive
```

#### **Cell 4: Clone Repository**
```python
# Automatically downloads your code from GitHub
# Creates folder structure
# Ready in ~30 seconds
```

#### **Cell 5: Upload Training Data**

**Option A: From Google Drive (Recommended)**
If your data is already in Google Drive:
```python
# Replace 'your_data_folder' with your actual folder name
!cp -r /content/drive/MyDrive/your_data_folder/tiles/train/* data/tiles/train/
!cp -r /content/drive/MyDrive/your_data_folder/tiles/val/* data/tiles/val/
```

**Option B: Direct Upload**
```python
# Run the cell
# Click "Choose Files" button
# Select your training images (can select multiple)
# Wait for upload to complete
```

📌 **Data Requirements:**
- Format: PNG or JPG images
- Size: Any size (will be resized to 128x128 automatically)
- Location: 
  - Training images → `data/tiles/train/`
  - Validation images → `data/tiles/val/`
- Recommended: At least 1000+ training images for good results

#### **Cell 6: Verify Data**
```python
# Shows how many images were uploaded
# Should display:
#   Training images: X,XXX
#   Validation images: X,XXX
```

#### **Cell 7: Quick Test (Optional)**
```python
# Tests if model can run on GPU
# Confirms everything is set up correctly
# Takes ~10 seconds
```

#### **Cell 8: START TRAINING! 🚀**
```python
# This is the main training cell
# Configuration:
#   - Batch size: 64
#   - Epochs: 100
#   - Image size: 128x128
#   - Learning rate: 1e-4
#   - Device: CUDA GPU

# Just click play and wait!
# Training will show progress like:
#   Epoch 1/100: Loss: 0.XXX | Val Loss: 0.XXX
#   Epoch 2/100: Loss: 0.XXX | Val Loss: 0.XXX
#   ...

# ⏱️ Estimated time: 25-35 hours for 100 epochs
# 💡 You can close the browser - training continues!
# 💡 Checkpoints auto-save every epoch
```

#### **Cell 9: Save Results**
```python
# After training completes, run this cell
# Saves model and logs to Google Drive
# Organized by timestamp
# Location: My Drive/MAE_Training_YYYYMMDD_HHMMSS/
```

#### **Cell 10: Monitor with TensorBoard (Optional)**
```python
# Shows real-time training graphs
# Displays loss curves, learning rate, metrics
# Great for monitoring progress
```

---

## 📊 Understanding Training Output

During training, you'll see output like:

```
Epoch 1/100:
  Train Loss: 0.8234
  Val Loss: 0.7891
  Learning Rate: 1.00e-04
  Time: 15m 23s
  
Checkpoint saved: checkpoints/mae_encoder_epoch_1.pth
```

**What these mean:**
- **Train Loss:** How well model learns training data (should decrease)
- **Val Loss:** How well model generalizes (should also decrease)
- **Learning Rate:** Step size for updates (starts low, increases, then decreases)
- **Time:** How long each epoch takes (~15-20 minutes per epoch)

---

## 💾 Where Are My Files?

After training, you'll have:

1. **Trained Model:** `checkpoints/mae_encoder.pth`
2. **Training Logs:** `runs/` folder (TensorBoard logs)
3. **Google Drive Backup:** `My Drive/MAE_Training_TIMESTAMP/`

---

## ⚙️ Customizing Training

You can modify training parameters in Cell 8:

```python
# Reduce batch size if you get out-of-memory errors
--batch-size 32  # or 16

# Train for more epochs for better results
--epochs 200  # or 300

# Use larger images (slower but potentially better)
--img-size 224  # instead of 128

# Adjust learning rate
--lr 5e-5  # smaller learning rate

# More workers for faster data loading
--workers 8  # if you have enough RAM
```

---

## ❓ Troubleshooting

### **Problem: "CUDA out of memory"**
**Solution:**
```python
# Reduce batch size in Cell 8
--batch-size 32  # or even 16
```

### **Problem: "No module named 'torch'"**
**Solution:**
- Re-run Cell 1 (Install Dependencies)
- Make sure you selected GPU runtime (Step 2)

### **Problem: Training is very slow**
**Solution:**
- Verify GPU is enabled: Runtime → Change runtime type → GPU
- Check GPU is working: Run Cell 2

### **Problem: Colab disconnects**
**Solution:**
- Don't worry! Training auto-saves every epoch
- Just reconnect and resume from last checkpoint
- To prevent disconnection:
  - Click the lock icon in bottom-right
  - Keep tab open (but you can minimize it)

### **Problem: Can't find uploaded files**
**Solution:**
```python
# List files to verify
!ls -lh data/tiles/train/ | head -20
!ls -lh data/tiles/val/ | head -20
```

### **Problem: Not enough storage in Colab**
**Solution:**
- Colab gives ~80GB free storage
- If running out, delete unnecessary files:
```python
!rm -rf /content/sample_data
```

---

## 🎯 After Training: What Next?

Once training is complete, you can:

### **1. Extract Features from New Images**
```bash
python inference/extract_features.py \
  --checkpoint checkpoints/mae_encoder.pth \
  --input path/to/new/images \
  --output features.npy \
  --device cuda
```

### **2. Analyze Geographic Regions**
```bash
python inference/analyze_region.py \
  --features features.npy \
  --n-clusters 5 \
  --tsne \
  --output-dir analysis_output
```

### **3. Environmental Condition Analysis**
```bash
python analyze_condition.py \
  --region assam \
  --condition vegetation \
  --year1 2019 \
  --year2 2024
```

### **4. Fine-tune for Specific Tasks**
- Classification (land cover types)
- Segmentation (identifying specific features)
- Change detection (comparing different time periods)

---

## 📈 Expected Results

**After 100 epochs:**
- Train Loss: ~0.3-0.5
- Val Loss: ~0.4-0.6
- Good feature representations for downstream tasks

**After 200+ epochs:**
- Train Loss: ~0.2-0.4
- Val Loss: ~0.3-0.5
- Excellent feature quality

**Note:** Lower loss = better reconstruction = better learned features

---

## 💡 Pro Tips

1. **Start Small:** Test with 10 epochs first to ensure everything works
2. **Monitor Progress:** Use TensorBoard (Cell 10) to watch training
3. **Save Frequently:** Results auto-save to Google Drive
4. **Use Checkpoints:** If training stops, you can resume from last checkpoint
5. **Batch Size Matters:** Larger batch = faster training but more memory
6. **More Data = Better Results:** Aim for 10,000+ training images if possible
7. **Patience:** Quality self-supervised learning takes time!

---

## 🔗 Useful Links

- **Your Repository:** https://github.com/SnehaShetty-18/ml_project
- **Google Colab:** https://colab.research.google.com
- **PyTorch Docs:** https://pytorch.org/docs
- **MAE Paper:** https://arxiv.org/abs/2111.06377

---

## 🆘 Need Help?

1. Check the troubleshooting section above
2. Review error messages carefully
3. Check GitHub Issues: https://github.com/SnehaShetty-18/ml_project/issues
4. Create a new issue with:
   - Error message
   - What step you're on
   - Screenshot if possible

---

## ✅ Checklist Before Starting

- [ ] GPU runtime enabled (Runtime → Change runtime type → GPU)
- [ ] Dependencies installed (Cell 1 completed successfully)
- [ ] Google Drive mounted (Cell 3 completed)
- [ ] Repository cloned (Cell 4 completed)
- [ ] Training data uploaded (Cell 5 completed)
- [ ] Data verified (Cell 6 shows correct counts)
- [ ] Quick test passed (Cell 7 successful)
- [ ] Ready to train! 🚀

---

**Good luck with your training! 🛰️✨**

*Expected completion: 25-35 hours*
*Result: Powerful MAE model for satellite imagery analysis*
