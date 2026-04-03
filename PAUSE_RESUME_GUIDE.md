# Pause & Resume Training Guide

## ✅ How It Works

The training script **automatically saves checkpoints** every epoch. You can pause and resume anytime!

---

## ⏸️ **To Pause Training**

Just **stop the training process** - your progress is automatically saved!

**Methods:**
1. Press `Ctrl+C` in terminal (graceful stop)
2. Close terminal (checkpoint already saved from last epoch)
3. Power off laptop (resume later)

**What gets saved:**
- Model weights ✓
- Optimizer state ✓
- Learning rate ✓
- Current epoch ✓
- Training history ✓

---

## ▶️ **To Resume Training**

### **Option 1: Resume from Auto-Saved Checkpoint**

```powershell
python training/train_mae.py --train-dir data/tiles_normalized --val-dir data/tiles/val --epochs 200 --batch-size 16 --resume checkpoints/mae_encoder.pth
```

This will:
- Load all previous training state
- Continue from where you left off
- Preserve learning rate schedule
- Keep training history

---

### **Option 2: Using setup_training.py**

If `setup_training.py` created a checkpoint, resume with:

```powershell
python training/train_mae.py --resume checkpoints/mae_encoder.pth --epochs 200
```

---

## 📊 **Example Workflow**

### **Day 1: Start Training**
```powershell
cd e:\major_ml\ml-engine
python setup_training.py --epochs 200 --batch-size 16
```
Training runs for 50 epochs, then you need to leave...

---

### **Day 2: Resume Training**
```powershell
python training/train_mae.py --resume checkpoints/mae_encoder.pth --epochs 200
```
Training continues from epoch 51 to 200!

---

## 🎯 **Key Points**

1. **Checkpoints are saved every epoch** - You never lose more than 1 epoch of progress

2. **Resume as many times as needed** - No limit on pause/resume cycles

3. **All state is preserved**:
   - Model weights
   - Optimizer momentum
   - Learning rate
   - Best validation loss
   - Full training history

4. **No quality loss** - Resumed training works exactly the same as continuous training

---

## 💡 **Pro Tips**

### **Train in Sessions**

**Session 1 (1 hour):**
```powershell
python setup_training.py --epochs 50 --batch-size 16
```

**Session 2 (resume next day):**
```powershell
python training/train_mae.py --resume checkpoints/mae_encoder.pth --epochs 100
```

**Session 3 (continue later):**
```powershell
python training/train_mae.py --resume checkpoints/mae_encoder.pth --epochs 200
```

---

### **Check Available Checkpoints**

```powershell
ls checkpoints/
```

You'll see:
- `mae_encoder.pth` ← Best model (updated during training)
- `checkpoint_final.pth` ← Last epoch saved

---

## ⚠️ **Important Notes**

1. **Don't delete `checkpoints/` folder** - Contains your saved progress

2. **Same parameters when resuming** - Keep `--train-dir`, `--val-dir`, `--img-size` the same

3. **Can change epochs** - You can increase `--epochs` when resuming to train longer

4. **Automatic detection** - If you specify more epochs than completed, it continues automatically

---

## 🚀 **Quick Reference**

| Action | Command |
|--------|---------|
| **Start fresh** | `python setup_training.py --epochs 200` |
| **Pause** | Press `Ctrl+C` or close terminal |
| **Resume** | `python training/train_mae.py --resume checkpoints/mae_encoder.pth --epochs 200` |
| **Check progress** | Look at epoch numbers in terminal output |

---

## ✅ **Summary**

**You can now:**
- ✅ Pause training anytime (Ctrl+C)
- ✅ Close laptop / shutdown
- ✅ Resume days/weeks later
- ✅ Continue from exact stopping point
- ✅ No progress lost!

**Training is flexible and interruption-safe!** 🎉
