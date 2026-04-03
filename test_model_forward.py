"""
Test model forward pass
"""
import torch
from models.mae_model import MaskedAutoencoder

def test_model():
    """Test if model can process batches correctly"""
    print("Creating MAE model with img_size=128...")
    
    model = MaskedAutoencoder(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        depth=12,
        mask_ratio=0.75
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create dummy batch matching your actual data
    batch_size = 64
    x = torch.randn(batch_size, 3, 128, 128)
    
    print(f"\nInput tensor shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input device: {x.device}")
    
    # Try forward pass on CPU first
    print("\nTesting forward pass on CPU...")
    try:
        model_cpu = model.cpu()
        recon, mask = model_cpu(x)
        print(f"✅ CPU Forward pass successful!")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Mask shape: {mask.shape}")
    except Exception as e:
        print(f"❌ CPU Forward pass failed: {str(e)}")
        return
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        print("\nTesting forward pass on CUDA...")
        try:
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            recon, mask = model_cuda(x_cuda)
            print(f"✅ CUDA Forward pass successful!")
            print(f"  Reconstruction shape: {recon.shape}")
            print(f"  Mask shape: {mask.shape}")
        except Exception as e:
            print(f"❌ CUDA Forward pass failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_model()
