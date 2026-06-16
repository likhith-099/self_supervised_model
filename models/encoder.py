"""
Encoder loader for pre-trained MAE models
Handles loading and inference with saved encoders
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from models.mae_model import MAEEncoder, MaskedAutoencoder


class EncoderLoader:
    """Load and use pre-trained MAE encoders"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        model_type: str = 'full'  # 'full' or 'encoder_only'
    ):
        """
        Initialize encoder loader
        
        Args:
            checkpoint_path: Path to saved checkpoint
            device: Device to run inference on
            model_type: Type of model ('full' or 'encoder_only')
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.encoder = None
        self.full_model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if self.model_type == 'encoder_only':
            # Load encoder directly
            self.encoder = MAEEncoder(
                img_size=128,
                patch_size=16,
                embed_dim=768,
                depth=12
            )
            
            # Extract encoder weights from checkpoint
            encoder_state = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                    encoder_state[new_key] = value
            
            self.encoder.load_state_dict(encoder_state)
        
        else:
            # Load full MAE model
            self.full_model = MaskedAutoencoder(
                img_size=128,
                patch_size=16,
                embed_dim=768,
                depth=12
            )
            self.full_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.encoder.eval() if self.encoder else self.full_model.eval()
        
        # Move model to device
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        else:
            self.full_model = self.full_model.to(self.device)
        
        print(f"Loaded model from: {self.checkpoint_path}")
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract feature vectors from an image
        
        Args:
            image_path: Path to input image
        
        Returns:
            Feature array
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))  # Changed from 256 to 128 to match training
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            if self.encoder:
                features = self.encoder(img_tensor)
            else:
                # Use encoder part of full model
                x = self.full_model.encoder.patch_embed(img_tensor)
                features = self.full_model.encoder(x)
        
        # Convert to numpy
        features_np = features.cpu().numpy().squeeze()
        return features_np
    
    def extract_features_batch(self, image_paths: list) -> np.ndarray:
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            Stacked feature arrays
        """
        all_features = []
        
        for img_path in image_paths:
            features = self.extract_features(img_path)
            all_features.append(features)
        
        return np.stack(all_features)
    
    def visualize_attention(self, image_path: str, save_path: str = None):
        """
        Visualize attention maps (optional advanced feature)
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
        """
        # TODO: Implement attention visualization
        print("Attention visualization not yet implemented")


def load_encoder(
    checkpoint_path: str,
    device: str = 'cuda'
) -> EncoderLoader:
    """
    Convenience function to load an encoder
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
    
    Returns:
        EncoderLoader instance
    """
    return EncoderLoader(checkpoint_path, device)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and test MAE encoder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Test image path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    loader = EncoderLoader(args.checkpoint, args.device)
    
    if args.image:
        features = loader.extract_features(args.image)
        print(f"Extracted features shape: {features.shape}")
        print(f"Feature statistics:")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")
    else:
        print("No test image provided. Use --image to test feature extraction")


if __name__ == '__main__':
    main()
