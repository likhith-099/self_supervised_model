"""
Feature extraction from satellite tiles using pre-trained MAE encoder
Converts image tiles to feature vectors for downstream analysis
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import EncoderLoader


class FeatureExtractor:
    """Extract features from satellite imagery using MAE encoder"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        img_size: int = 256
    ):
        """
        Initialize feature extractor
        
        Args:
            checkpoint_path: Path to MAE checkpoint
            device: Device to run inference on
            img_size: Image size for processing
        """
        self.device = device
        self.img_size = img_size
        self.loader = EncoderLoader(checkpoint_path, device)
        self.img_size = img_size
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats (common practice)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract features from single image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Feature vector
        """
        img_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            if self.loader.encoder:
                features = self.loader.encoder(img_tensor)
            else:
                x = self.loader.full_model.encoder.patch_embed(img_tensor)
                features = self.loader.full_model.encoder(x)
        
        # Global average pooling to get fixed-size feature vector
        features_np = features.cpu().numpy().squeeze()
        
        if len(features_np.shape) == 2:
            # Average over patches
            feature_vector = features_np.mean(axis=0)
        else:
            feature_vector = features_np
        
        return feature_vector
    
    def extract_from_directory(
        self,
        input_dir: str,
        output_file: str,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract features from all images in directory
        
        Args:
            input_dir: Input directory with images
            output_file: File to save features (.npy)
            max_samples: Maximum number of samples to process
        
        Returns:
            Stacked feature arrays
        """
        input_path = Path(input_dir)
        image_paths = list(input_path.rglob('*.png'))
        
        if max_samples:
            image_paths = image_paths[:max_samples]
        
        print(f"Processing {len(image_paths)} images...")
        
        all_features = []
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
            
            try:
                features = self.extract_features(str(img_path))
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        features_array = np.stack(all_features)
        
        # Save features
        np.save(output_file, features_array)
        print(f"Saved features to: {output_file}")
        print(f"Feature shape: {features_array.shape}")
        
        return features_array


def main():
    parser = argparse.ArgumentParser(description='Extract features from satellite tiles')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MAE checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with tile images')
    parser.add_argument('--output', type=str, default='features.npy',
                        help='Output file for features')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create feature extractor
    extractor = FeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Extract features
    features = extractor.extract_from_directory(
        input_dir=args.input,
        output_file=args.output,
        max_samples=args.max_samples
    )
    
    # Print statistics
    print("\nFeature Statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")


if __name__ == '__main__':
    main()
