"""
Check model training status and system readiness
"""

import os
from pathlib import Path


def check_training_status():
    """Check if MAE model training is complete"""
    
    print("=" * 80)
    print("MODEL TRAINING STATUS CHECK")
    print("=" * 80)
    
    # Check for checkpoint files
    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    
    checkpoints_found = []
    if checkpoint_dir.exists():
        for file in checkpoint_dir.glob('*.pth'):
            size_mb = file.stat().st_size / (1024 * 1024)
            checkpoints_found.append((file.name, size_mb))
    
    if checkpoints_found:
        print("\n✓ Model checkpoints found:")
        for name, size in checkpoints_found:
            print(f"  ✓ {name} ({size:.1f} MB)")
        
        # Check for main encoder
        encoder_path = checkpoint_dir / 'mae_encoder.pth'
        if encoder_path.exists():
            print("\n✅ MAIN ENCODER READY!")
            print("   Location: checkpoints/mae_encoder.pth")
            print("   Status: Training complete - ready for inference")
            return True
        else:
            print("\n⚠ WARNING: Main encoder not found")
            print("   Expected: checkpoints/mae_encoder.pth")
            print("   Recommendation: Run training first")
            return False
    
    else:
        print("\n❌ NO MODEL CHECKPOINTS FOUND")
        print("\nTraining required before analysis can proceed.")
        print("\nTo train the model, run:")
        print("  python auto_analyze.py --place \"Your Location\" --condition vegetation --start 2019-01-01 --end 2024-01-01")
        print("\nOr run training directly:")
        print("  python training/train_mae.py --train-dir data/tiles/train --val-dir data/tiles/val --epochs 200")
        return False


def check_data_readiness():
    """Check if data is ready for analysis"""
    
    print("\n" + "=" * 80)
    print("DATA READINESS CHECK")
    print("=" * 80)
    
    checks = {
        'Raw satellite images': 'data/raw/sentinel',
        'Filtered images': 'data/raw/filtered',
        'Training tiles': 'data/tiles/train',
        'Validation tiles': 'data/tiles/val',
        'Normalized tiles': 'data/tiles_normalized'
    }
    
    all_ready = True
    for name, path in checks.items():
        full_path = Path(__file__).parent / path
        if full_path.exists() and any(full_path.iterdir()):
            n_files = len(list(full_path.glob('*')))
            print(f"  ✓ {name}: {n_files} files")
        else:
            print(f"  ✗ {name}: NOT FOUND or EMPTY")
            all_ready = False
    
    return all_ready


def check_feature_files():
    """Check for extracted feature files"""
    
    print("\n" + "=" * 80)
    print("FEATURE FILES CHECK")
    print("=" * 80)
    
    feature_files = list(Path(__file__).parent.glob('features_*.npy'))
    
    if feature_files:
        print(f"\n✓ Found {len(feature_files)} feature files:")
        for f in feature_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  ✓ {f.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("\n⚠ No feature files found")
        print("  Features will be extracted during analysis")
        return False


def get_next_steps(model_ready: bool, data_ready: bool):
    """Provide next steps based on current status"""
    
    print("\n" + "=" * 80)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 80)
    
    if not model_ready:
        print("\n📍 STEP 1: Train MAE Model")
        print("   Command: python auto_analyze.py --place \"Mangalore, India\" \\")
        print("                         --condition vegetation \\")
        print("                         --start 2019-01-01 --end 2024-01-01")
        print("   \n   This will:")
        print("   1. Download satellite images")
        print("   2. Preprocess and tile images")
        print("   3. Train MAE model (~2-4 hours)")
        print("   4. Extract features automatically")
    
    elif not data_ready:
        print("\n📍 STEP 1: Download/Prepare Data")
        print("   Command: python preprocessing/download_sentinel_api.py \\")
        print("                         --place \"Mangalore, India\" \\")
        print("                         --condition vegetation \\")
        print("                         --start 2019-01-01 --end 2024-01-01")
    
    else:
        print("\n✅ SYSTEM READY FOR ANALYSIS!")
        print("\n📍 Run Environmental Analysis:")
        print("   Command: python analyze_condition.py \\")
        print("                         --region mangalore \\")
        print("                         --condition vegetation \\")
        print("                         --year1 2019 --year2 2024")
        print("\n   Or use fully automated workflow:")
        print("   Command: python auto_analyze.py \\")
        print("                         --place \"Mangalore, India\" \\")
        print("                         --condition vegetation \\")
        print("                         --start 2019-01-01 --end 2024-01-01")
    
    print("\n" + "=" * 80)


def main():
    """Main status check function"""
    
    model_ready = check_training_status()
    data_ready = check_data_readiness()
    features_ready = check_feature_files()
    
    get_next_steps(model_ready, data_ready)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model Training: {'✅ COMPLETE' if model_ready else '❌ PENDING'}")
    print(f"Data Prepared:  {'✅ READY' if data_ready else '⚠ NEEDS ATTENTION'}")
    print(f"Features Ready: {'✅ YES' if features_ready else '⏳ WILL GENERATE'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
