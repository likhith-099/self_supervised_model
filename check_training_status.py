"""
Check Training Progress and Checkpoints
"""
import torch
from pathlib import Path

def check_training_status():
    """Check if training is running and checkpoints exist"""
    
    print("=" * 60)
    print("TRAINING STATUS CHECK")
    print("=" * 60)
    
    # Check if checkpoints directory exists
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("❌ Checkpoints directory does not exist!")
        print("   Training may not have started saving yet.")
        return
    
    # List all checkpoints
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print("⚠️  No checkpoint files found yet.")
        print("   Training needs to complete at least one epoch with improved validation loss.")
        return
    
    print(f"\n✓ Found {len(checkpoints)} checkpoint(s):\n")
    
    for ckpt_path in sorted(checkpoints, key=lambda p: p.stat().st_mtime):
        file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            epoch = checkpoint.get('epoch', 'unknown')
            train_loss = checkpoint.get('train_loss', [])
            val_loss = checkpoint.get('val_loss', [])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            n_epochs_completed = len(train_loss)
            
            print(f"📄 File: {ckpt_path.name}")
            print(f"   Size: {file_size_mb:.1f} MB")
            print(f"   Epoch saved: {epoch}")
            print(f"   Total epochs completed: {n_epochs_completed}")
            
            if train_loss:
                print(f"   Last train loss: {train_loss[-1]:.4f}")
            
            if val_loss:
                print(f"   Last val loss: {val_loss[-1]:.4f}")
                print(f"   Best val loss: {best_val_loss:.4f}")
            
            print()
            
        except Exception as e:
            print(f"⚠️  Error reading {ckpt_path.name}: {e}")
            print()
    
    print("=" * 60)
    print("✅ Training checkpoints are being saved!")
    print("=" * 60)
    
    # Check if currently training
    import subprocess
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Where-Object {(Get-WmiObject Win32_Process -Filter 'ProcessId = $_.Id').CommandLine -like '*train*'}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout.strip():
            print("\n🔄 Training process is currently RUNNING")
        else:
            print("\n⏸️  No active training process found")
            
    except Exception as e:
        print(f"\n⚠️  Could not check running processes: {e}")

if __name__ == "__main__":
    check_training_status()
