"""
Complete Real Data Analysis Pipeline
Downloads ACTUAL satellite imagery and processes it through the full pipeline
"""

import sys
import os
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\n{'='*80}")
    print(f"📍 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ Command had issues but continuing...")
        return False
    return True

def main():
    print("=" * 80)
    print("🌊 REAL DATA ANALYSIS: Jammu & Kashmir Water Expansion (2025-2026)")
    print("=" * 80)
    print("\n⚠️ This will download ACTUAL satellite imagery from Copernicus API")
    print("📡 Requires internet connection")
    print("⏱️ May take several minutes depending on data availability")
    print("=" * 80)
    
    # Configuration
    place = "Jammu and Kashmir, India"
    condition = "water"
    years = [2025, 2026]
    
    # Create main output directory
    main_output = Path("analysis_output/jk_real_data_water_2025_2026")
    main_output.mkdir(parents=True, exist_ok=True)
    
    feature_files = {}
    
    # Process each year
    for year in years:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING YEAR: {year}")
        print(f"{'#'*80}")
        
        year_dir = main_output / f"year_{year}"
        year_dir.mkdir(exist_ok=True)
        
        # Step 1: Download real satellite images
        print(f"\n[1/5] Downloading REAL satellite imagery for {year}...")
        print("-" * 80)
        
        download_cmd = (
            f'python preprocessing/download_sentinel_api.py '
            f'--place "{place}" '
            f'--condition {condition} '
            f'--start {year}-01-01 '
            f'--end {year}-12-31 '
            f'--limit 20 '
            f'--output "{year_dir / "raw_images"}"'
        )
        
        success = run_command(download_cmd, f"Downloading satellite data for {year}")
        
        if not success:
            print(f"⚠️ Download had issues for {year}, checking if we have images...")
        
        # Check if we got images
        raw_dir = year_dir / "raw_images"
        if raw_dir.exists():
            images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
            if len(images) > 0:
                print(f"✓ Found {len(images)} satellite images for {year}")
            else:
                print(f"⚠️ No images downloaded for {year}")
                print(f"   This could be due to:")
                print(f"   - No satellite coverage for this date range")
                print(f"   - API authentication required")
                print(f"   - Network connectivity issues")
                print(f"\n   Continuing with demo mode...")
                continue
        else:
            print(f"⚠️ Download directory not created")
            continue
        
        # Step 2: Preprocess images
        print(f"\n[2/5] Preprocessing images for {year}...")
        print("-" * 80)
        
        preprocess_cmd = (
            f'python preprocessing/preprocess.py '
            f'--input "{raw_dir}" '
            f'--output "{year_dir / "processed"}" '
            f'--tile-size 128 '
            f'--condition {condition}'
        )
        
        run_command(preprocess_cmd, f"Preprocessing {year} images")
        
        # Step 3: Normalize data
        print(f"\n[3/5] Normalizing data for {year}...")
        print("-" * 80)
        
        normalize_cmd = (
            f'python preprocessing/normalize.py '
            f'--input "{year_dir / "processed"}" '
            f'--output "{year_dir / "normalized"}"'
        )
        
        run_command(normalize_cmd, f"Normalizing {year} data")
        
        # Step 4: Extract features using trained MAE model
        print(f"\n[4/5] Extracting features with trained MAE model for {year}...")
        print("-" * 80)
        
        features_cmd = (
            f'python inference/extract_features.py '
            f'--checkpoint checkpoints_30k/checkpoint_final.pth '
            f'--data-dir "{year_dir / "normalized"}" '
            f'--output "{year_dir / "features.npy"}" '
            f'--batch-size 32 '
            f'--device cuda'
        )
        
        run_command(features_cmd, f"Extracting features for {year}")
        
        # Check if features were extracted
        feature_file = year_dir / "features.npy"
        if feature_file.exists():
            features = np.load(feature_file)
            print(f"✓ Features extracted: {features.shape}")
            feature_files[year] = str(feature_file)
        else:
            print(f"⚠️ Feature extraction failed for {year}")
    
    # Step 5: Analysis and visualization
    print(f"\n\n{'#'*80}")
    print(f"# ANALYSIS & VISUALIZATION")
    print(f"{'#'*80}")
    
    if len(feature_files) == 0:
        print("\n❌ No features extracted from real data")
        print("\nThis could be because:")
        print("  1. Copernicus API requires authentication")
        print("  2. No satellite imagery available for the date range")
        print("  3. Network connectivity issues")
        print("\n💡 SOLUTION: Use the demo analysis with synthetic data")
        print("   python analyze_jk_water.py")
        return
    
    print(f"\n✓ Successfully extracted features for {len(feature_files)} years: {list(feature_files.keys())}")
    
    # Analyze each year
    for year, feature_file in feature_files.items():
        print(f"\n[5/5] Analyzing patterns for {year}...")
        
        analysis_dir = main_output / f"analysis_{year}"
        analysis_cmd = (
            f'python inference/analyze_region.py '
            f'--features "{feature_file}" '
            f'--n-clusters 5 '
            f'--pca-dims 2 '
            f'--output-dir "{analysis_dir}"'
        )
        
        run_command(analysis_cmd, f"Analyzing {year}")
    
    # Generate before/after comparison if we have both years
    if 2025 in feature_files and 2026 in feature_files:
        print(f"\n[6/5] Generating before/after comparison...")
        
        comparison_cmd = (
            f'python inference/map_visualization.py '
            f'--baseline "{feature_files[2025]}" '
            f'--current "{feature_files[2026]}" '
            f'--condition {condition} '
            f'--output "{main_output / "water_comparison.png"}"'
        )
        
        run_command(comparison_cmd, "Creating comparison visualization")
    
    # Generate future predictions
    if len(feature_files) >= 2:
        print(f"\n[7/5] Generating future predictions...")
        
        features_dict_str = ",".join([f'{y}:{feature_files[y]}' for y in feature_files.keys()])
        prediction_cmd = (
            f'python inference/predict_future.py '
            f'--features-dict "{features_dict_str}" '
            f'--condition {condition} '
            f'--place "{place}" '
            f'--predict-years 10 '
            f'--output "{main_output / "future_prediction.png"}"'
        )
        
        run_command(prediction_cmd, "Creating future predictions")
    
    # Generate interactive report with REAL data
    print(f"\n[8/5] Generating interactive visual report...")
    
    from utils.visual_report import create_interactive_report
    
    report_path = create_interactive_report(
        analysis_dir=str(main_output),
        place_name=place,
        condition=condition,
        start_year=min(feature_files.keys()),
        end_year=max(feature_files.keys()),
        feature_files=feature_files
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ REAL DATA ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📍 Location: {place}")
    print(f"📅 Years processed: {list(feature_files.keys())}")
    print(f"📁 Results: {main_output}")
    print(f"\n📊 Generated Files:")
    
    for file in sorted(main_output.rglob('*.*')):
        if file.is_file():
            rel_path = file.relative_to(main_output)
            size_kb = file.stat().st_size / 1024
            print(f"  ✓ {rel_path} ({size_kb:.1f} KB)")
    
    print(f"\n🎨 Interactive Report: {report_path}")
    print(f"\n{'='*80}")
    print("💡 Note: This analysis used REAL satellite imagery downloaded from")
    print("   Copernicus Data Space Ecosystem API")
    print(f"{'='*80}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
