"""
Check decoder output shapes
"""
import torch
from models.mae_model import MaskedAutoencoder

def check_shapes():
    model = MaskedAutoencoder(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.75
    )
    
    batch_size = 64
    imgs = torch.randn(batch_size, 3, 128, 128)
    
    print(f"Input shape: {imgs.shape}")
    
    # Forward pass
    recon, mask = model(imgs)
    
    print(f"\nReconstruction shape: {recon.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # What target should be
    target = imgs.clone()
    print(f"\nTarget initial shape: {target.shape}")
    
    # Reshape target to patches
    patch_size = 16
    B, C, H, W = target.shape
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    # Reshape to (B, C, N_patches_h, patch_size, N_patches_w, patch_size)
    target_unfold = target.reshape(B, C, n_patches_h, patch_size, n_patches_w, patch_size)
    # Permute to (B, n_patches_h, n_patches_w, patch_size, patch_size, C)
    target_unfold = target_unfold.permute(0, 2, 4, 3, 5, 1)
    # Flatten to (B, N_patches, patch_size*patch_size*C)
    target_unfold = target_unfold.reshape(B, -1, patch_size * patch_size * C)
    
    print(f"Target unfolded shape: {target_unfold.shape}")
    print(f"Expected reconstruction shape: (B={B}, N_patches={n_patches_h*n_patches_w}, dim={patch_size*patch_size*C})")
    
    # Check if they match
    print(f"\nShape match: {recon.shape == target_unfold.shape}")

if __name__ == "__main__":
    check_shapes()
