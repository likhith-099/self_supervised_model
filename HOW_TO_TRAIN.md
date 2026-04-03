# 🚀 HOW TO TRAIN YOUR MODEL - SIMPLE GUIDE

## YOU ALREADY HAVE DATA! ✓

You have **25 Sentinel-2 satellite images** downloaded in `data/raw/sentinel/`

---

## OPTION 1: EASIEST (Double-Click)

### For Windows Users:

1. **Double-click this file:** `START_TRAINING.bat`
2. Press any key when it asks
3. Leave it running (takes 3-5 hours)
4. Come back to see your results!

**That's it!** The script will:
- ✅ Filter cloudy images
- ✅ Cut images into 128×128 tiles
- ✅ Normalize pixel values
- ✅ Train the AI model (2-4 hours)
- ✅ Extract environmental features
- ✅ Generate visualizations with maps and predictions

---

## OPTION 2: PYTHON SCRIPT (Step-by-Step)

Run this command:
```bash
python simple_train.py
```

This will guide you through each step interactively.

---

## OPTION 3: FULLY AUTOMATED (For New Analysis)

If you want to analyze a NEW location from scratch:

```bash
python auto_analyze.py --place "Mangalore, India" --condition vegetation --start 2024-01-01 --end 2024-03-31
```

This does EVERYTHING automatically:
- Downloads new satellite images
- Converts place name to coordinates
- Preprocesses everything
- Trains model
- Generates analysis

---

## WHAT'S HAPPENING DURING TRAINING

### Step 1-3: Preprocessing (5-10 minutes)
```
Filtering clouds → Cutting tiles → Normalizing pixels
```

### Step 4: Model Training (2-4 hours) ⏰
```
Epoch 1/200:   Learning basic patterns
Epoch 50/200:  Recognizing features
Epoch 100/200: Understanding context
Epoch 200/200: Master reconstruction
```

The AI is learning to:
- Identify vegetation types
- Detect water bodies
- Recognize urban areas
- Understand land cover patterns

### Step 5-6: Analysis (10-15 minutes)
```
Extracting features → Generating maps → Creating predictions
```

---

## AFTER TRAINING COMPLETE

### You'll Get:

**Files Created:**
```
checkpoints/mae_encoder.pth       ← Your trained AI model
features_analysis.npy              ← Environmental features
analysis_report.png                ← Visualizations
```

**Visualizations:**
- Before/After satellite map comparison
- Change detection heatmap (red/yellow/green)
- Trend predictions for next 10 years
- Severity gauge showing environmental health

---

## TO VIEW RESULTS

Open the generated files:
```bash
# On Windows
start analysis_report.png

# Or manually open:
# ml-engine/analysis_report.png
```

---

## TO ANALYZE SPECIFIC CONDITIONS

After training is complete, you can analyze different things:

### Vegetation Analysis:
```bash
python analyze_condition.py --region mangalore --condition vegetation --year1 2024 --year2 2025
```

### Water/Flood Analysis:
```bash
python analyze_condition.py --region mangalore --condition water --year1 2024 --year2 2025
```

### Urban Growth:
```bash
python analyze_condition.py --region mangalore --condition urban --year1 2024 --year2 2025
```

---

## TROUBLESHOOTING

### "CUDA not available" error:
The model will train on CPU instead (slower, but works).
Expected time on CPU: 8-12 hours

### "Out of memory" error:
Edit the training command to use smaller batches:
```bash
python training/train_mae.py --batch-size 32 ...
```

### Want faster training?
Reduce epochs from 200 to 100:
```bash
python training/train_mae.py --epochs 100 ...
```

---

## CURRENT STATUS

**Your Data:**
- ✅ 25 satellite images ready
- ✅ Location: Mangalore area (Tile 43PDQ)
- ✅ Dates: Various from 2025

**Next Step:** Run the training!

---

## QUICK COMMAND REFERENCE

| What You Want | Command |
|---------------|---------|
| **Start training** | `START_TRAINING.bat` (double-click) |
| **Interactive training** | `python simple_train.py` |
| **Check status** | `python check_status.py` |
| **Analyze vegetation** | `python analyze_condition.py --region mangalore --condition vegetation --year1 2024 --year2 2025` |
| **Full automation** | `python auto_analyze.py --place "Mangalore, India" --condition vegetation --start 2024-01-01 --end 2024-03-31` |

---

## ESTIMATED TIMES

| Computer Type | Training Time |
|---------------|---------------|
| GPU (RTX 3060+) | 2-3 hours |
| GPU (older) | 3-4 hours |
| CPU (modern) | 8-12 hours |
| CPU (older) | 12-18 hours |

---

## NEED HELP?

Run the status checker:
```bash
python check_status.py
```

This will tell you:
- If model is trained
- If data is ready
- What to do next

---

## EXAMPLE OUTPUT

After training and analysis, you'll see something like:

```
================================================================================
VEGETATION ANALYSIS
================================================================================
Baseline Coverage: 45.23%
Current Coverage: 38.12%
Change: -7.11% (LOSS)
Severity: MEDIUM

PREDICTION:
If trend continues → 32.45% by 2029
```

Plus colorful maps showing exactly WHERE the changes happened!

---

## READY TO START?

### Just run:
```bash
START_TRAINING.bat
```

Or double-click the file in Windows Explorer!

Good luck! 🎯
