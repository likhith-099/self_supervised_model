"""
Test MAE Model Accuracy and Performance
Evaluates the trained model on validation data
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.mae_model import MaskedAutoencoder
from training.dataset_loader import SatelliteTileDataset
from torch.utils.data import DataLoader


def find_best_checkpoint(checkpoint_dir='checkpoints'):
    """Find the best/latest checkpoint"""
    checkpoint_path = Path(checkpoint_dir)
    
    # First, check if mae_encoder.pth exists (best model)
    best_model = checkpoint_path / 'mae_encoder.pth'
    if best_model.exists():
        return str(best_model)
    
    # Otherwise, find the latest batch checkpoint
    checkpoints = list(checkpoint_path.glob('checkpoint_epoch*.pth'))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by batch number and get the latest
    def extract_batch(path):
        try:
            return int(path.stem.split('batch')[-1])
        except:
            return 0
    
    checkpoints.sort(key=extract_batch)
    return str(checkpoints[-1])


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate model on validation set
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    all_losses = []
    
    criterion = nn.MSELoss()
    patch_size = model.encoder.patch_embed.patch_size
    
    print(f"\nEvaluating on {len(dataloader.dataset)} samples...")
    print("-" * 60)
    
    start_time = time.time()
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        B, C, H, W = images.shape
        
        # Forward pass
        reconstructions, mask = model(images)
        
        # Reshape target to patches (same as training)
        target = images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        target = target.permute(0, 2, 4, 3, 5, 1)
        target = target.reshape(B, -1, patch_size * patch_size * C)
        target = target.to(device)
        
        # Compute loss
        loss = criterion(reconstructions, target)
        total_loss += loss.item()
        all_losses.append(loss.item())
        n_batches += 1
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / n_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / n_batches
    std_loss = np.std(all_losses)
    min_loss = np.min(all_losses)
    max_loss = np.max(all_losses)
    
    elapsed = time.time() - start_time
    
    metrics = {
        'avg_loss': avg_loss,
        'std_loss': std_loss,
        'min_loss': min_loss,
        'max_loss': max_loss,
        'n_samples': len(dataloader.dataset),
        'n_batches': n_batches,
        'evaluation_time': elapsed,
        'all_losses': all_losses
    }
    
    return metrics


@torch.no_grad()
def test_reconstruction_quality(model, dataset, device, num_samples=5):
    """
    Test reconstruction quality on sample images
    
    Returns:
        List of (original, masked, reconstructed) image tuples
    """
    model.eval()
    results = []
    
    print(f"\nGenerating {num_samples} reconstruction examples...")
    print("-" * 60)
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        
        # Forward pass
        recon, mask = model(image)
        
        # Convert back to image format
        B, C, H, W = image.shape
        patch_size = model.encoder.patch_embed.patch_size
        
        # Reshape reconstruction to image
        # recon shape: (B, N_patches, patch_size*patch_size*C)
        N_patches = (H // patch_size) * (W // patch_size)
        recon_reshaped = recon.reshape(B, H // patch_size, W // patch_size, 
                                       patch_size, patch_size, C)
        recon_reshaped = recon_reshaped.permute(0, 5, 1, 3, 2, 4)
        recon_image = recon_reshaped.reshape(B, C, H, W)
        
        # Denormalize images (reverse the normalization from dataset)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        original = (image * std + mean).clamp(0, 1)
        reconstructed = (recon_image * std + mean).clamp(0, 1)
        
        # Create masked image
        mask = mask.cpu()
        masked = original.clone()
        mask_expanded = mask.reshape(1, 1, H // patch_size, W // patch_size)
        mask_expanded = mask_expanded.repeat_interleave(patch_size, dim=2)
        mask_expanded = mask_expanded.repeat_interleave(patch_size, dim=3)
        mask_expanded = mask_expanded.unsqueeze(1).expand(-1, 3, -1, -1)
        masked = masked * (1 - mask_expanded) + mask_expanded * 0.5  # Gray out masked
        
        results.append({
            'original': original.squeeze().cpu(),
            'masked': masked.squeeze().cpu(),
            'reconstructed': reconstructed.squeeze().cpu()
        })
    
    return results


def visualize_reconstructions(results, save_path='reconstruction_quality.png'):
    """Visualize reconstruction quality"""
    n_samples = len(results)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original
        axes[i, 0].imshow(result['original'].permute(1, 2, 0))
        axes[i, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Masked
        axes[i, 1].imshow(result['masked'].permute(1, 2, 0))
        axes[i, 1].set_title('Masked Input (75% masked)', fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Reconstructed
        axes[i, 2].imshow(result['reconstructed'].permute(1, 2, 0))
        axes[i, 2].set_title('Reconstructed', fontsize=14, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved reconstruction visualization to: {save_path}")
    plt.close()


def print_evaluation_summary(metrics):
    """Print comprehensive evaluation summary"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nDataset Information:")
    print(f"  Total samples evaluated: {metrics['n_samples']}")
    print(f"  Total batches: {metrics['n_batches']}")
    print(f"  Evaluation time: {metrics['evaluation_time']:.2f} seconds")
    
    print(f"\nReconstruction Loss (MSE):")
    print(f"  Average Loss: {metrics['avg_loss']:.4f}")
    print(f"  Std Deviation: {metrics['std_loss']:.4f}")
    print(f"  Min Loss: {metrics['min_loss']:.4f}")
    print(f"  Max Loss: {metrics['max_loss']:.4f}")
    
    # Convert MSE to PSNR (Peak Signal-to-Noise Ratio)
    # PSNR = 10 * log10(MAX^2 / MSE), where MAX=1 for normalized images
    mse = metrics['avg_loss']
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
        print(f"\nImage Quality Metrics:")
        print(f"  PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
        
        # Quality assessment
        if psnr > 30:
            quality = "Excellent"
        elif psnr > 25:
            quality = "Good"
        elif psnr > 20:
            quality = "Fair"
        else:
            quality = "Poor"
        print(f"  Reconstruction Quality: {quality}")
    
    print("\n" + "=" * 80)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test MAE model accuracy')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (auto-detects if not provided)')
    parser.add_argument('--val-dir', type=str, default='data/tiles/val',
                        help='Validation data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
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
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of reconstruction examples to generate')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Find checkpoint
    if args.checkpoint is None:
        try:
            args.checkpoint = find_best_checkpoint('checkpoints')
            print(f"Auto-detected checkpoint: {args.checkpoint}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print("Creating MAE model...")
    model = MaskedAutoencoder(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mask_ratio=args.mask_ratio
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    batch_idx = checkpoint.get('batch_idx', 'unknown')
    best_val_loss = checkpoint.get('best_val_loss', 'unknown')
    
    print(f"✓ Model loaded successfully")
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Checkpoint batch: {batch_idx}")
    if best_val_loss != 'unknown':
        print(f"  Best validation loss at training: {best_val_loss:.4f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Create validation dataloader
    print(f"\nLoading validation dataset from: {args.val_dir}")
    val_dataset = SatelliteTileDataset(
        args.val_dir,
        img_size=args.img_size,
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation images")
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("STARTING MODEL EVALUATION")
    print("=" * 80)
    
    metrics = evaluate_model(model, val_loader, device)
    
    # Print summary
    print_evaluation_summary(metrics)
    
    # Test reconstruction quality
    results = test_reconstruction_quality(model, val_dataset, device, args.num_samples)
    
    # Visualize results
    viz_path = output_dir / 'reconstruction_quality.png'
    visualize_reconstructions(results, save_path=str(viz_path))
    
    # Save metrics
    metrics_path = output_dir / 'evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("MAE Model Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Batch: {batch_idx}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Validation samples: {metrics['n_samples']}\n\n")
        f.write(f"Reconstruction Loss (MSE):\n")
        f.write(f"  Average: {metrics['avg_loss']:.4f}\n")
        f.write(f"  Std: {metrics['std_loss']:.4f}\n")
        f.write(f"  Min: {metrics['min_loss']:.4f}\n")
        f.write(f"  Max: {metrics['max_loss']:.4f}\n\n")
        
        mse = metrics['avg_loss']
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
            f.write(f"Image Quality:\n")
            f.write(f"  PSNR: {psnr:.2f} dB\n")
            if psnr > 30:
                quality = "Excellent"
            elif psnr > 25:
                quality = "Good"
            elif psnr > 20:
                quality = "Fair"
            else:
                quality = "Poor"
            f.write(f"  Quality: {quality}\n")
        
        f.write(f"\nEvaluation time: {metrics['evaluation_time']:.2f} seconds\n")
    
    print(f"✓ Saved metrics to: {metrics_path}")
    
    # Save sample images
    for i, result in enumerate(results):
        # Save original
        orig_img = (result['original'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(orig_img).save(output_dir / f'sample_{i}_original.png')
        
        # Save reconstructed
        recon_img = (result['reconstructed'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(recon_img).save(output_dir / f'sample_{i}_reconstructed.png')
    
    print(f"✓ Saved {len(results)} sample images to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - evaluation_metrics.txt: Detailed metrics")
    print("  - reconstruction_quality.png: Visual comparison")
    print("  - sample_*_original.png: Original images")
    print("  - sample_*_reconstructed.png: Reconstructed images")
    print("=" * 80)


if __name__ == '__main__':
    main()
