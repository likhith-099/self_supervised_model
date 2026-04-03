"""
Force CUDA usage in PyTorch
"""
import torch
import os

# Set environment variable to force CUDA visibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("="*60)
print("CUDA DETECTION TEST")
print("="*60)

# Check if CUDA is available
print(f"\n1. torch.cuda.is_available(): {torch.cuda.is_available()}")

# Check device count
print(f"2. torch.cuda.device_count(): {torch.cuda.device_count()}")

# Try to manually set device
try:
    device = torch.device('cuda')
    print(f"3. Created CUDA device: {device}")
    
    # Try to create a tensor on GPU
    test_tensor = torch.randn(3, 3).to(device)
    print(f"4. Successfully created tensor on GPU!")
    print(f"   Tensor device: {test_tensor.device}")
    print(f"   Tensor shape: {test_tensor.shape}")
except Exception as e:
    print(f"4. ERROR creating GPU tensor: {e}")
    print("\nTrying CUDA device index 0...")
    try:
        device = torch.device('cuda:0')
        test_tensor = torch.randn(3, 3).to(device)
        print(f"✓ Success with cuda:0!")
        print(f"  Device: {test_tensor.device}")
    except Exception as e2:
        print(f"✗ Also failed: {e2}")

# Check available devices
if torch.cuda.is_available():
    print(f"\n5. Available GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"      Compute Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("\n5. No CUDA devices found by PyTorch")
    print("\nPossible solutions:")
    print("  1. Restart computer after driver installation")
    print("  2. Check if GPU is disabled in BIOS")
    print("  3. Try reinstalling PyTorch with specific CUDA version")
    print("  4. Use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*60)
