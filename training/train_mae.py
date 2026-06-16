"""
MAE Training Script
Trains Masked Autoencoder on satellite imagery
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mae_model import MaskedAutoencoder
from training.dataset_loader import create_dataloaders


class MAETrainer:
    """Trainer class for MAE model"""
    
    def __init__(
        self,
        model: MaskedAutoencoder,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 400,
        checkpoint_dir: str = 'checkpoints',
        resume_from: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model: MAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum training epochs
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Loss function - reconstruction loss (MSE)
        self.criterion = nn.MSELoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Optimizer with AdamW
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=lr * 0.01
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        log_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if provided
        self.start_epoch = 0
        if resume_from and Path(resume_from).exists():
            self.load_checkpoint(resume_from)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        print('='*60)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from epoch {checkpoint.get('epoch', 0)}")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded optimizer state")
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Loaded scheduler state")
        
        # Load training history
        if 'train_loss' in checkpoint:
            self.history['train_loss'] = checkpoint['train_loss']
            self.history['val_loss'] = checkpoint['val_loss']
            self.history['learning_rate'] = checkpoint.get('learning_rate', [])
            print(f"✓ Loaded training history")
        
        # Load best validation loss
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"✓ Loaded best validation loss: {self.best_val_loss:.4f}")
        
        # Load batch index if available (for mid-epoch resume)
        if 'batch_idx' in checkpoint and checkpoint['batch_idx'] is not None:
            print(f"✓ Mid-epoch checkpoint: Epoch {checkpoint['epoch']}, Batch {checkpoint['batch_idx']}")
        
        # Set start epoch
        self.start_epoch = len(self.history['train_loss'])
        print(f"\n📊 Training will resume from epoch {self.start_epoch + 1}")
        print('='*60 + '\n')
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, (images, _) in enumerate(self.train_loader):
            # Debug: Check tensor dimensions
            if len(images.shape) != 4 or images.shape[1] != 3:
                print(f"ERROR: Invalid image shape {images.shape} at batch {batch_idx}")
                print(f"Expected shape: [batch_size, 3, height, width]")
                continue
            
            images = images.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with autocast():
                reconstructions, mask = self.model(images)
                
                # Compute loss
                # Convert target images to patch format to match reconstruction
                B, C, H, W = images.shape
                patch_size = self.model.encoder.patch_embed.patch_size
                
                # Reshape target to patches: (B, C, H, W) -> (B, N_patches, patch_size*patch_size*C)
                target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
                target = target.permute(0, 2, 4, 3, 5, 1)  # (B, n_h, n_w, ps, ps, C)
                target = target.reshape(B, -1, patch_size * patch_size * C)
                target = target.to(self.device)
                
                # Only compute loss on masked patches
                loss = self.criterion(reconstructions, target)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Progress logging - every batch for first epoch, then every 50 batches
            log_interval = 1 if epoch == 0 else 50
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / n_batches
                batch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Time: {batch_time:.1f}s")
            
            # Save mid-epoch checkpoint every 500 batches for safety
            if batch_idx > 0 and batch_idx % 500 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch{epoch}_batch{batch_idx}.pth', batch_idx)
        
        avg_train_loss = total_loss / n_batches
        epoch_total_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_total_time:.1f}s")
        return avg_train_loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for images, _ in self.val_loader:
            images = images.to(self.device)
            
            reconstructions, mask = self.model(images)
            
            # Match reconstruction output shape - use same logic as training
            B, C, H, W = images.shape
            patch_size = self.model.encoder.patch_embed.patch_size
            
            # Reshape target to patches: (B, C, H, W) -> (B, N_patches, patch_size*patch_size*C)
            target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
            target = target.permute(0, 2, 4, 3, 5, 1)  # (B, n_h, n_w, ps, ps, C)
            target = target.reshape(B, -1, patch_size * patch_size * C)
            target = target.to(self.device)
            
            loss = self.criterion(reconstructions, target)
            
            total_loss += loss.item()
            n_batches += 1
        
        # Handle empty validation set
        if n_batches == 0:
            print("WARNING: Empty validation set! Returning 0.0 for validation loss")
            return 0.0
        
        avg_val_loss = total_loss / n_batches
        return avg_val_loss
    
    def adjust_lr_warmup(self, epoch: int):
        """Adjust learning rate during warmup phase"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_factor * self.scheduler.get_last_lr()[0]
    
    def save_checkpoint(self, epoch: int, filename: str = 'checkpoint.pth', batch_idx: int = None):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(state, checkpoint_path)
        if batch_idx is not None:
            print(f"\n💾 Saved mid-epoch checkpoint: {checkpoint_path} (Epoch {epoch}, Batch {batch_idx})")
        else:
            print(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self) -> dict:
        """Full training loop"""
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Mixed precision: {'Enabled' if self.device == 'cuda' else 'Disabled'}")
        print("-" * 60)
        
        training_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()
            
            # Warmup LR adjustment
            self.adjust_lr_warmup(epoch)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'mae_encoder.pth')
                print(f"✓ New best model! Val loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs for safety
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            total_elapsed = time.time() - training_start_time
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Epoch Time: {epoch_time:.1f}s | "
                  f"Total Time: {total_elapsed:.1f}s")
            print("-" * 60)
            
            # Early stopping check (optional)
            if epoch > 50 and val_loss > self.best_val_loss * 1.5:
                print("Early stopping triggered")
                break
        
        # Save final checkpoint
        self.save_checkpoint(self.max_epochs - 1, 'checkpoint_final.pth')
        
        total_training_time = time.time() - training_start_time
        hours = total_training_time / 3600
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {total_training_time:.1f}s ({hours:.2f} hours)")
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Train MAE on satellite imagery')
    parser.add_argument('--train-dir', type=str, default='data/tiles/train',
                        help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='data/tiles/val',
                        help='Validation data directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Encoder depth')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Masking ratio')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=args.img_size
    )
    
    # Create model
    print("Creating MAE model...")
    model = MaskedAutoencoder(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mask_ratio=args.mask_ratio
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create trainer
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        max_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume
    )
    
    # Start training
    history = trainer.train()
    
    print("\nTraining finished!")
    print(f"Final checkpoint saved to: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
