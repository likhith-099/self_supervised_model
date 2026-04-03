"""
Test script to force PyTorch to use NVIDIA GPU
"""
import torch
import os

# Try different methods to enable CUDA
print("="*70)
print("ATTEMPTING TO ENABLE NVIDIA GPU")
print("="*70)

# Method 1: Check if GPU is in power saving mode
print("\n[Method 1] Checking Windows Graphics Settings...")
print("GO TO: Settings > System > Display > Graphics settings")
print("Make sure 'Hardware-accelerated GPU scheduling' is ON")
print("Add Python.exe and set it to 'High Performance'")

# Method 2: Try with different CUDA versions
print("\n[Method 2] Testing CUDA availability...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Method 3: Force device selection
print("\n[Method 3] Attempting to force GPU usage...")

try:
    # Try to initialize CUDA
    torch.cuda.init()
    print("✓ CUDA initialized successfully")
    
    # Check devices
    device_count = torch.cuda.device_count()
    print(f"✓ Device count: {device_count}")
    
    if device_count > 0:
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ No devices found even after initialization")
        
except Exception as e:
    print(f"✗ CUDA initialization failed: {e}")

# Method 4: Test tensor on GPU
print("\n[Method 4] Testing tensor operations...")
try:
    device = torch.device('cuda:0')
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(f"✓ Successfully created tensor on GPU!")
    print(f"  Device: {x.device}")
    print(f"  Values: {x}")
except Exception as e:
    print(f"✗ Failed to use GPU: {e}")
    print("\n  This confirms GPU is not accessible to PyTorch")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

if not torch.cuda.is_available():
    print("\n⚠️  GPU IS NOT ACCESSIBLE")
    print("\nNext steps:")
    print("1. Restart your computer (sometimes GPU gets re-enabled)")
    print("2. Check BIOS settings - look for 'Discrete GPU' or 'Switchable Graphics'")
    print("3. In Windows: Settings > System > Display > Graphics")
    print("   - Add python.exe")
    print("   - Set to 'High Performance' (NVIDIA GPU)")
    print("4. If still failing, use Google Colab (free GPU cloud)")
else:
    print("\n✅ GPU IS WORKING!")
    print(f"Using: {torch.cuda.get_device_name(0)}")
