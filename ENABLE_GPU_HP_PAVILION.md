# HP Pavilion Laptop - Enable NVIDIA GPU Guide

## Your Laptop Specs:
- **Model**: HP Pavilion (with NVIDIA GeForce 940MX)
- **OS**: Windows 11
- **Graphics**: Intel UHD 620 + NVIDIA 940MX (Switchable Graphics)

---

## 🔧 METHOD 1: Windows 11 Graphics Settings (EASIEST - TRY FIRST)

### Step-by-Step:

1. **Open Windows Settings**
   - Press `Win + I` keys together
   - Or click Start → Settings (gear icon)

2. **Go to System → Display**
   - Click on "System" in left sidebar
   - Click on "Display" on the right

3. **Open Graphics Settings**
   - Scroll down and click "Graphics" (or "Graphics settings")
   
4. **Enable Hardware Acceleration**
   - Click "Change default graphics settings"
   - Turn ON: **"Hardware-accelerated GPU scheduling"**
   - Restart when prompted

5. **Add Python to High Performance**
   - Go back to Graphics settings
   - Under "Custom options for apps", click "Browse"
   - Navigate to: `C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe`
   - Click "Add"
   - Now you'll see python.exe in the list
   - Click on it → "Options"
   - Select **"High Performance"** (should show NVIDIA GeForce 940MX)
   - Click Save

6. **Restart your laptop** (IMPORTANT!)

---

## 🔧 METHOD 2: HP Graphics Switcher (If Available)

Some HP Pavilion models have a dedicated graphics switcher:

1. **Search for "HP Command Center" or "OMEN Gaming Hub"**
   - Press `Win + S`
   - Type: "HP Command Center" or "OMEN"
   - Open it

2. **Look for Graphics Settings**
   - Find "Graphics Switcher" or "GPU Switch"
   - Set to **"Discrete Graphics"** or "NVIDIA GPU only"
   - NOT "Hybrid" or "Integrated"

3. **Apply and Restart**

---

## 🔧 METHOD 3: BIOS Settings (Advanced)

⚠️ **Only if comfortable with BIOS**

1. **Enter BIOS:**
   - Shut down completely
   - Press power button, then immediately tap **F10** repeatedly (or Esc then F10)

2. **Look for Graphics Settings:**
   - Navigate using arrow keys
   - Look for:
     - "Device Configuration"
     - "Built-in Device Options"
     - "Video" or "Display"
     - "Switchable Graphics"
   
3. **Enable Discrete GPU:**
   - Change from "Hybrid" to "Discrete" or "Enabled"
   - Or look for "Dynamic Switchable Graphics" and disable it

4. **Save and Exit:**
   - Press F10 to save
   - Select "Yes"
   - Laptop will restart

---

## 🔧 METHOD 4: NVIDIA Control Panel

1. **Open NVIDIA Control Panel**
   - Right-click on desktop
   - Select "NVIDIA Control Panel"
   - (If not there, search in Start menu)

2. **Manage 3D Settings:**
   - Click "Manage 3D settings" on left
   - Go to "Program Settings" tab
   - Click "Add"
   - Find and add: `python.exe`
   - Set preferred graphics processor to: **"High-performance NVIDIA processor"**
   - Click Apply

3. **Set PhysX Configuration:**
   - In NVIDIA Control Panel, click "Set PhysX Configuration"
   - Under "Processor", select: **GeForce 940MX**
   - Click Apply

---

## ✅ VERIFICATION STEPS

After trying any method above:

### Step 1: Check if GPU is Visible
```powershell
# Open PowerShell and run:
Get-WmiObject Win32_VideoController | Select-Object Name, Status
```

You should see both:
- Intel(R) UHD Graphics 620
- NVIDIA GeForce 940MX

### Step 2: Test PyTorch CUDA
```powershell
cd e:\major_ml\ml-engine
python check_cuda.py
```

**Expected output if working:**
```
CUDA available: True
Device count: 1
Current device: NVIDIA GeForce 940MX
```

### Step 3: Run Training
```powershell
python training/fast_train_mae.py
```

Should show:
```
✓ CUDA available: True
✓ Device: cuda
```

---

## 🎯 QUICK FIX FOR HP PAVILION (Most Common Solution)

For HP Pavilion laptops with 940MX, this usually works:

1. **Right-click Desktop → NVIDIA Control Panel**
2. **Manage 3D Settings → Program Settings**
3. **Add python.exe** (browse to installation folder)
4. **Select "High-performance NVIDIA processor"**
5. **Click Apply**
6. **Restart laptop**
7. **Test:** `python check_cuda.py`

---

## ❌ IF NOTHING WORKS

Your laptop might have:
- **Broken/dislodged GPU** (hardware issue)
- **Corrupted drivers**
- **BIOS blocking discrete GPU**

### Try:
1. **Reinstall NVIDIA Drivers:**
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Search for: GeForce 940MX
   - Download and install latest driver
   - Restart

2. **Use DDU (Display Driver Uninstaller):**
   - Download DDU from Guru3D
   - Boot into Safe Mode
   - Run DDU to completely remove old drivers
   - Reboot and install fresh NVIDIA drivers

3. **Use Google Colab Instead:**
   - Upload the notebook I created: `MAE_Training_Colab.ipynb`
   - Free GPU, no hardware issues!

---

## 📞 HP Support Quick Reference

If hardware issue suspected:
- **HP Support Assistant**: Pre-installed on your laptop
- **HP Customer Support**: 1-800-HP-INVENT (US)
- **Check warranty**: Some Pavilion models still under warranty

---

## 🚀 NEXT STEPS AFTER ENABLING GPU

Once `python check_cuda.py` shows CUDA available:

```powershell
cd e:\major_ml\ml-engine
python training/fast_train_mae.py
```

This will automatically use your NVIDIA 940MX and train in ~25-35 hours instead of 150 days!
