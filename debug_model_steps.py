"""
Debug model forward pass step by step
"""
import torch
from models.mae_model import MaskedAutoencoder, PatchEmbed

def debug_forward():
    """Trace through forward pass step by step"""
    print("Creating model...")
    model = MaskedAutoencoder(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.75
    )
    
    # Create batch
    batch_size = 64
    imgs = torch.randn(batch_size, 3, 128, 128)
    print(f"\nInput images shape: {imgs.shape}")
    
    # Step 1: Patch embedding
    print("\n--- Step 1: Patch Embedding ---")
    x = model.encoder.patch_embed(imgs)
    print(f"After patch_embed: {x.shape}")
    
    # Step 2: Random masking
    print("\n--- Step 2: Random Masking ---")
    B, N, _ = x.shape
    print(f"B={B}, N={N}, embed_dim={_}")
    
    n_keep = int(N * (1 - model.mask_ratio))
    print(f"Patches to keep: {n_keep} out of {N}")
    
    noise = torch.rand(B, N, device=imgs.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    
    mask = torch.ones(B, N, device=imgs.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.mean().item():.2%}")
    
    # Step 3: Get visible patches
    print("\n--- Step 3: Gather Visible Patches ---")
    x_visible = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    print(f"x_visible shape: {x_visible.shape}")
    
    # Step 4: Pass through encoder
    print("\n--- Step 4: Encoder Forward ---")
    print(f"Calling encoder with x_visible shape: {x_visible.shape}, mask=None, ids_keep shape: {ids_keep.shape}")
    try:
        x_encoded = model.encoder(x_visible, mask=None, ids_keep=ids_keep)
        print(f"x_encoded shape: {x_encoded.shape}")
    except Exception as e:
        print(f"❌ Encoder failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Pass through decoder
    print("\n--- Step 5: Decoder Forward ---")
    print(f"Calling decoder with x_encoded shape: {x_encoded.shape}, mask shape: {mask.shape}")
    try:
        recon = model.decoder(x_encoded, mask)
        print(f"Reconstruction shape: {recon.shape}")
        print(f"✅ Forward pass successful!")
    except Exception as e:
        print(f"❌ Decoder failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_forward()
