"""
Simple MAE Training Script - More robust with better logging
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mae_model import MaskedAutoencoder
from training.dataset_loader import create_dataloaders


def train_simple():
    """Simplified training with immediate feedback"""
    
    # Configuration
    config = {
        'train_dir': 'data/tiles/train',
        'val_dir': 'data/tiles/val',
        'batch_size': 16,
        'epochs': 100,
        'img_size': 128,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*70)
    print("SIMPLE MAE TRAINING")
    print("="*70)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['img_size']}")
    print(f"Epochs: {config['epochs']}")
    print("="*70)
    
    # Create dataloaders
    print("\n[1/5] Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=0,  # Use main process for stability
        img_size=config['img_size'],
        pin_memory=False
    )
    
    print(f"✓ Train samples: {len(train_loader.dataset):,}")
    print(f"✓ Val samples: {len(val_loader.dataset):,}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Test first batch loading
    print("\n[2/5] Testing data loading...")
    try:
        images, targets = next(iter(train_loader))
        print(f"✓ First batch shape: {images.shape}")
        print(f"✓ Value range: [{images.min():.3f}, {images.max():.3f}]")
    except Exception as e:
        print(f"✗ ERROR: Data loading failed - {e}")
        return
    
    # Create model
    print("\n[3/5] Creating MAE model...")
    model = MaskedAutoencoder(
        img_size=config['img_size'],
        patch_size=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.75
    ).to(config['device'])
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {n_params:,}")
    
    # Setup training
    print("\n[4/5] Setting up training...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    # TensorBoard
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    print(f"✓ Logs: runs/{run_name}")
    
    # Checkpoint dir
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n[5/5] Starting training...\n")
    print("-" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (images, _) in enumerate(train_loader):
            try:
                images = images.to(config['device'])
                
                # Forward pass
                reconstructions, mask = model(images)
                
                # Compute loss (patch-based)
                B, C, H, W = images.shape
                patch_size = 16
                target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
                target = target.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, patch_size * patch_size * C)
                target = target.to(config['device'])
                
                loss = criterion(reconstructions, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
                
                # Print progress EVERY batch
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    avg_loss = train_loss / (batch_idx + 1)
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error at batch {batch_idx}: {e}")
                continue
        
        if n_batches == 0:
            print("  ✗ No batches processed!")
            continue
            
        avg_train_loss = train_loss / n_batches
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for images, _ in val_loader:
                try:
                    images = images.to(config['device'])
                    reconstructions, mask = model(images)
                    
                    B, C, H, W = images.shape
                    patch_size = 16
                    target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
                    target = target.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, patch_size * patch_size * C)
                    target = target.to(config['device'])
                    
                    loss = criterion(reconstructions, target)
                    val_loss += loss.item()
                    n_val_batches += 1
                except Exception as e:
                    print(f"  ✗ Val error: {e}")
                    continue
        
        if n_val_batches > 0:
            avg_val_loss = val_loss / n_val_batches
        else:
            avg_val_loss = 0.0
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / 'mae_encoder.pth')
            print(f"  ✓ New best! Val loss: {avg_val_loss:.4f}")
        
        # Print summary
        epoch_time = time.time() - start_time
        print(f"\n  Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"    Best Val Loss: {best_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
    
    writer.close()
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/mae_encoder.pth")
    print("="*70)


if __name__ == '__main__':
    train_simple()
