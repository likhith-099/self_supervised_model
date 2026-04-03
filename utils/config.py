"""
Configuration management for ML pipeline
Centralized configuration for hyperparameters and paths
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Config:
    """Configuration manager for the ML pipeline"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
    OUTPUT_DIR = PROJECT_ROOT / 'output'
    
    # Data paths
    RAW_SENTINEL_DIR = DATA_DIR / 'raw' / 'sentinel'
    RAW_LANDSAT_DIR = DATA_DIR / 'raw' / 'landsat'
    TRAIN_TILES_DIR = DATA_DIR / 'tiles' / 'train'
    VAL_TILES_DIR = DATA_DIR / 'tiles' / 'val'
    REGION_DATA_DIR = DATA_DIR / 'region_data'
    
    # Model hyperparameters
    IMG_SIZE = 256
    PATCH_SIZE = 16
    EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 12
    MLP_RATIO = 4.0
    MASK_RATIO = 0.75
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 10
    MAX_EPOCHS = 400
    
    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Normalization parameters (ImageNet)
    NORMALIZATION = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    # Device configuration
    DEVICE = 'cuda' if os.name != 'nt' else 'cuda'  # Default to CUDA if available
    
    # Copernicus API Configuration
    # SET YOUR CREDENTIALS HERE or use environment variables
    COPERNICUS_USER = os.getenv('COPERNICUS_USER', '')
    COPERNICUS_PASSWORD = os.getenv('COPERNICUS_PASSWORD', '')
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = cls.__dict__
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            elif hasattr(cls, k):
                value = getattr(cls, k)
            else:
                return default
        
        return value
    
    @classmethod
    def set(cls, key: str, value: Any):
        """Set configuration value"""
        setattr(cls, key, value)
    
    @classmethod
    def all(cls) -> Dict[str, Any]:
        """Get all configuration values"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def save_to_yaml(cls, filepath: str):
        """Save configuration to YAML file"""
        config_dict = cls.all()
        
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_from_yaml(cls, filepath: str):
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("Current Configuration")
        print("=" * 60)
        
        for key, value in cls.all().items():
            if isinstance(value, (str, int, float, list, dict, Path)):
                print(f"{key}: {value}")
        
        print("=" * 60)


# Region-specific configurations
REGION_CONFIGS = {
    'assam': {
        'bbox': [89.5, 25.5, 96.0, 28.0],  # min_lon, min_lat, max_lon, max_lat
        'name': 'Assam, India',
        'cloud_threshold': 30.0,
        'best_months': list(range(11, 4)),  # Nov-Mar (dry season)
        'tile_size': 128,
        'bands': ['B2', 'B3', 'B4', 'B8']
    },
    'mangalore': {
        'bbox': [74.8, 12.8, 75.1, 13.0],
        'name': 'Mangalore, India',
        'cloud_threshold': 25.0,
        'best_months': list(range(1, 4)) + list(range(11, 13)),  # Jan-Mar, Nov-Dec
        'tile_size': 128,
        'bands': ['B2', 'B3', 'B4', 'B8']
    },
    'amazon': {
        'bbox': [-75.0, -10.0, -60.0, 5.0],
        'name': 'Amazon Rainforest',
        'cloud_threshold': 20.0,
        'best_months': list(range(6, 10)),  # Jun-Sep (dry season)
        'tile_size': 128,
        'bands': ['B2', 'B3', 'B4', 'B8']
    },
    'congo': {
        'bbox': [12.0, -5.0, 30.0, 5.0],
        'name': 'Congo Basin',
        'cloud_threshold': 25.0,
        'best_months': list(range(12, 4)),  # Dec-Mar
        'tile_size': 128,
        'bands': ['B2', 'B3', 'B4', 'B8']
    }
}


def get_region_config(region_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for specific region"""
    return REGION_CONFIGS.get(region_name.lower())


def main():
    """Test configuration"""
    Config.print_config()
    
    # Test saving/loading
    Config.save_to_yaml('config.yaml')
    print("\nSaved config to config.yaml")
    
    # Load region config
    assam_config = get_region_config('assam')
    if assam_config:
        print(f"\nAssam region config:")
        print(f"  Bounding box: {assam_config['bbox']}")
        print(f"  Cloud threshold: {assam_config['cloud_threshold']}%")


if __name__ == '__main__':
    main()
