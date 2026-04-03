"""
Masked Autoencoder (MAE) implementation for satellite imagery
Self-supervised learning with masked image modeling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings"""
    
    def __init__(self, n_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indices is not None:
            # Select positional embeddings for specific patch indices
            pos_embed_selected = torch.gather(
                self.pos_embed.expand(x.shape[0], -1, -1),
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1])
            )
            x = x + pos_embed_selected
        else:
            x = x + self.pos_embed
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class MAEEncoder(nn.Module):
    """MAE Encoder - processes visible patches"""
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_encoding = PositionalEncoding(
            n_patches=(img_size // patch_size) ** 2,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                ids_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x is already patch-embedded: (B, N_patches, embed_dim)
        # Add positional encoding
        x = self.pos_encoding(x, ids_keep)
        
        # Apply masking if provided
        if mask is not None:
            # Keep only visible patches
            visible_mask = ~mask.bool()
            x = x[visible_mask].unsqueeze(0)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class MAEDecoder(nn.Module):
    """MAE Decoder - reconstructs masked patches"""
    
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Project back to pixel space
        self.patch_size = patch_size
        self.proj_out = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]  # Batch size
        N_total = mask.shape[1]  # Total number of patches
        
        x = self.embed_to_decoder(x)
        
        # Add mask tokens for masked patches
        # mask shape: (B, N_total), we need to add mask tokens where mask=1
        n_masked = int(mask.sum(dim=1).max())  # Max number of masked patches in batch
        
        # Create mask tokens for all masked positions
        mask_tokens = self.mask_token.expand(B, -1, -1)  # (1, 1, dim) -> (B, 1, dim)
        mask_tokens = mask_tokens.repeat(1, n_masked, 1)  # (B, n_masked, dim)
        
        # Concatenate visible and masked tokens
        x_with_masks = torch.cat([x, mask_tokens], dim=1)
        
        # Reorder based on mask (simplified)
        # In full implementation, need to restore original positions
        
        for block in self.decoder_blocks:
            x_with_masks = block(x_with_masks)
        
        x_with_masks = self.decoder_norm(x_with_masks)
        
        # Project to pixels
        recon = self.proj_out(x_with_masks)
        
        # Reshape to (B, N_patches, patch_size*patch_size*3)
        # recon shape: (B, N_total, patch_size*patch_size*3)
        return recon


class MaskedAutoencoder(nn.Module):
    """Complete MAE model"""
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        mask_ratio: float = 0.75
    ):
        super().__init__()
        
        self.encoder = MAEEncoder(
            img_size, patch_size, in_chans, embed_dim,
            depth, num_heads, mlp_ratio=4.0
        )
        
        self.decoder = MAEDecoder(
            patch_size, embed_dim, decoder_embed_dim,
            decoder_depth, num_heads=16
        )
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random mask and apply it"""
        B, N, _ = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        
        # Generate random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first n_keep tokens
        ids_keep = ids_shuffle[:, :n_keep]
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        
        # Unshuffle to get mask in original order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return ids_keep, mask, ids_restore
    
    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Tuple of (reconstructed images, mask)
        """
        # Encode
        x = self.encoder.patch_embed(imgs)
        ids_keep, mask, ids_restore = self.random_masking(x)
        
        # Get visible patches
        x_visible = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        
        # Process through encoder with ids_keep for positional encoding
        x_encoded = self.encoder(x_visible, mask=None, ids_keep=ids_keep)
        
        # Decode
        recon = self.decoder(x_encoded, mask)
        
        return recon, mask


def main():
    # Example usage
    model = MaskedAutoencoder(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.75
    )
    
    # Create dummy input
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    recon, mask = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.mean().item():.2%}")


if __name__ == '__main__':
    main()
