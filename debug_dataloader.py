"""
Debug data loader to check batch shapes
"""
import torch
from training.dataset_loader import create_dataloaders

def debug_dataloader(train_dir: str, val_dir: str, batch_size: int = 64, img_size: int = 128):
    """Check what the dataloader is actually producing"""
    print(f"Creating dataloaders with img_size={img_size}, batch_size={batch_size}")
    
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for easier debugging
        img_size=img_size,
        pin_memory=False
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Check first few batches
    print("\nChecking train batches:")
    for i, (images, targets) in enumerate(train_loader):
        print(f"Batch {i}: images.shape = {images.shape}, targets.shape = {targets.shape}")
        
        if i == 0:
            print(f"  - First batch details:")
            print(f"    - Images dtype: {images.dtype}")
            print(f"    - Images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"    - Has NaN: {torch.isnan(images).any()}")
            print(f"    - Has Inf: {torch.isinf(images).any()}")
        
        if i >= 2:  # Only check first 3 batches
            break
    
    print("\nChecking val batches:")
    for i, (images, targets) in enumerate(val_loader):
        print(f"Batch {i}: images.shape = {images.shape}, targets.shape = {targets.shape}")
        
        if i >= 2:
            break

if __name__ == "__main__":
    debug_dataloader(
        train_dir='data/tiles_normalized',
        val_dir='data/tiles/val',
        batch_size=64,
        img_size=128
    )
