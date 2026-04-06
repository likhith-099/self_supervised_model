"""
Dataset loader for MAE training
Loads and preprocesses satellite image tiles for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import albumentations as A


class SatelliteTileDataset(Dataset):
    """PyTorch Dataset for satellite image tiles"""
    
    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        augment: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing tile images
            img_size: Target image size
            augment: Whether to apply data augmentation
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Collect all image paths
        self.image_paths = list(self.data_dir.rglob('*.png'))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
        
        # Define transformations
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Tuple of (image tensor, dummy target - for compatibility)
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image
            img = np.array(Image.open(img_path).convert('RGB'))
            
            # Apply transformations
            img = self.transform(image=img)['image']
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            
            # Verify dimensions
            if img_tensor.shape[0] != 3:
                print(f"Warning: Image {img_path} has wrong channels: {img_tensor.shape[0]}")
                # Return a valid placeholder
                img_tensor = torch.zeros(3, self.img_size, self.img_size)
            
            # For MAE, target is same as input (reconstruction task)
            return img_tensor, img_tensor
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a valid placeholder
            img_tensor = torch.zeros(3, self.img_size, self.img_size)
            return img_tensor, img_tensor


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 256,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SatelliteTileDataset(
        train_dir,
        img_size=img_size,
        augment=True
    )
    
    val_dataset = SatelliteTileDataset(
        val_dir,
        img_size=img_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_dataloader_stats(dataloader: DataLoader) -> dict:
    """
    Compute dataset statistics
    
    Args:
        dataloader: PyTorch dataloader
    
    Returns:
        Dictionary with mean and std statistics
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    
    for images, _ in dataloader:
        B, C, H, W = images.shape
        total_pixels += B * H * W
        
        total_sum += images.sum(dim=(0, 2, 3)).sum()
        total_sq_sum += (images ** 2).sum(dim=(0, 2, 3)).sum()
    
    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_samples': len(dataloader.dataset)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset loader')
    parser.add_argument('--train-dir', type=str, default='data/tiles/train',
                        help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='data/tiles/val',
                        help='Validation data directory')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    train_loader, val_loader = create_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test data loading
    images, targets = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Compute statistics
    stats = get_dataloader_stats(train_loader)
    print(f"\nDataset statistics:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std: {stats['std']}")
    print(f"  Samples: {stats['n_samples']}")


if __name__ == '__main__':
    main()
