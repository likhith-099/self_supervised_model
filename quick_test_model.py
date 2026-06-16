"""
Quick Test: Run trained model on validation data (unseen during training)
Generates visualizations and analysis automatically
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("QUICK TEST: Running Trained Model on Unseen Validation Data")
print("=" * 80)

# Configuration
CHECKPOINT = "checkpoints_30k/checkpoint_final.pth"
VAL_DATA = "data/tiles/val"
OUTPUT_DIR = "test_output/quick_test"

# Step 1: Extract features
print("\n[1/3] Extracting features from validation data...")
print("-" * 80)

feature_file = f"{OUTPUT_DIR}/features.npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

extract_cmd = (
    f"python inference/extract_features.py "
    f"--checkpoint {CHECKPOINT} "
    f"--input {VAL_DATA} "
    f"--output {feature_file} "
    f"--device cuda"
)

print(f"Running: {extract_cmd}")
result = os.system(extract_cmd)

if result != 0 or not Path(feature_file).exists():
    print("❌ Feature extraction failed!")
    sys.exit(1)

print(f"\n✓ Features extracted: {feature_file}")

# Step 2: Analyze region
print("\n[2/3] Analyzing patterns and clustering...")
print("-" * 80)

analysis_dir = f"{OUTPUT_DIR}/analysis"

analyze_cmd = (
    f"python inference/analyze_region.py "
    f"--features {feature_file} "
    f"--n-clusters 5 "
    f"--pca-dims 2 "
    f"--tsne "
    f"--output-dir {analysis_dir}"
)

print(f"Running: {analyze_cmd}")
os.system(analyze_cmd)

# Step 3: Generate visualization report
print("\n[3/3] Creating visualization summary...")
print("-" * 80)

# List all generated visualizations
analysis_path = Path(analysis_dir)
if analysis_path.exists():
    print("\nGenerated Visualizations:")
    print("-" * 80)
    for img_file in sorted(analysis_path.glob('*.png')):
        print(f"  ✓ {img_file.name}")
        print(f"    Path: {img_file}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print(f"\n📊 What was tested:")
    print(f"  • Model processed {Path(VAL_DATA).glob('*.png')} unseen images")
    print(f"  • Features extracted using trained MAE encoder")
    print(f"  • Clustering and pattern analysis performed")
    print(f"  • Visualizations generated")
    print("\n💡 Open the PNG files to see the results!")
    print("=" * 80)
else:
    print("❌ Analysis directory not found!")
    sys.exit(1)
