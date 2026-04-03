"""
Fast MAE Training - Optimized for speed
Uses data prefetching and larger batches
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mae_model import MaskedAutoencoder
from training.dataset_loader import SatelliteTileDataset


def train_fast():
    """Optimized training with maximum speed"""
    
    # Configuration - OPTIMIZED FOR SPEED
    config = {
        'train_dir': 'data/tiles/train',
        'val_dir': 'data/tiles/val',
        'batch_size': 32,  # Larger batch for efficiency
        'epochs': 100,
        'img_size': 128,
        'lr': 1e-4,
        'num_workers': 2,  # Parallel data loading
        'pin_memory': True,  # Faster GPU transfer
        'prefetch_factor': 2  # Preload batches
    }
    
    print("="*70)
    print("FAST MAE TRAINING - OPTIMIZED FOR SPEED")
    print("="*70)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA NOT AVAILABLE! Training will be VERY slow on CPU")
        print("   Install PyTorch with CUDA support for 10-50x speedup")
        device = 'cpu'
    else:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    
    config['device'] = device
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Image size: {config['img_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Workers: {config['num_workers']}")
    print("="*70)
    
    # Create datasets
    print("\n[1/5] Loading datasets...")
    start_load = time.time()
    
    train_dataset = SatelliteTileDataset(
        config['train_dir'],
        img_size=config['img_size'],
        augment=True
    )
    
    val_dataset = SatelliteTileDataset(
        config['val_dir'],
        img_size=config['img_size'],
        augment=False
    )
    
    print(f"✓ Loaded in {time.time()-start_load:.1f}s")
    print(f"  Train: {len(train_dataset):,} images")
    print(f"  Val: {len(val_dataset):,} images")
    
    # Create dataloaders with optimization
    print("\n[2/5] Creating optimized dataloaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,
        prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Test batch loading speed
    print("\n[3/5] Testing data loading speed...")
    test_start = time.time()
    images, _ = next(iter(train_loader))
    load_time = time.time() - test_start
    print(f"✓ First batch loaded in {load_time:.2f}s")
    print(f"  Shape: {images.shape}")
    print(f"  Device: {images.device}")
    
    # Create model
    print("\n[4/5] Creating MAE model...")
    model = MaskedAutoencoder(
        img_size=config['img_size'],
        patch_size=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.75
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {n_params:,}")
    print(f"✓ Device: {next(model.parameters()).device}")
    
    # Setup training
    print("\n[5/5] Starting training...\n")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    total_start = time.time()
    
    print("-" * 70)
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            reconstructions, mask = model(images)
            
            # Compute loss
            B, C, H, W = images.shape
            patch_size = 16
            target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
            target = target.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, patch_size * patch_size * C)
            target = target.to(device, non_blocking=True)
            
            loss = criterion(reconstructions, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress logging
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                avg_loss = train_loss / (batch_idx + 1)
                elapsed = time.time() - epoch_start
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                eta_min = (len(train_loader) - batch_idx) / batches_per_sec / 60 if batches_per_sec > 0 else 0
                
                print(f"  Batch {batch_idx+1:5d}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {batches_per_sec:.1f} batch/s | "
                      f"ETA: {eta_min:.0f}m")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                reconstructions, mask = model(images)
                
                B, C, H, W = images.shape
                patch_size = 16
                target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
                target = target.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, patch_size * patch_size * C)
                target = target.to(device, non_blocking=True)
                
                loss = criterion(reconstructions, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / 'mae_encoder.pth')
            print(f"  ✓ Best model! Val: {avg_val_loss:.4f}")
        
        # Summary
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        
        print(f"\n  Epoch {epoch+1:3d} Summary:")
        print(f"    Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
        print(f"    Time: {epoch_time/60:.1f}m | Total: {total_elapsed/3600:.1f}h")
        print(f"    Best Val: {best_val_loss:.4f}")
        
        # Estimate completion
        epochs_done = epoch + 1
        avg_epoch_time = total_elapsed / epochs_done
        remaining_epochs = config['epochs'] - epochs_done
        eta_hours = remaining_epochs * avg_epoch_time / 3600
        print(f"    ETA to complete: {eta_hours:.1f}h ({remaining_epochs} epochs left)")
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved: checkpoint_epoch_{epoch+1}.pth")
    
    writer.close()
    
    total_time = time.time() - total_start
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/mae_encoder.pth")
    print("="*70)


if __name__ == '__main__':
    train_fast()
