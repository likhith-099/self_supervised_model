"""
Quick Demo: Generate Interactive Visual Report
Shows real satellite maps + analysis + predictions in browser
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.visual_report import create_interactive_report

print("=" * 80)
print("🎨 DEMO: Interactive Visual Report with Real Maps")
print("=" * 80)

# Create demo data
demo_dir = Path("demo_visual_report")
demo_dir.mkdir(exist_ok=True)

# Generate synthetic feature data for 5 years
print("\n📊 Generating sample analysis data...")
feature_files = {}
for year in range(2020, 2025):
    # Simulate environmental change
    np.random.seed(year)
    trend_offset = (year - 2020) * 0.05
    features = np.random.randn(500, 768).astype(np.float32) + trend_offset
    feature_file = str(demo_dir / f"features_{year}.npy")
    np.save(feature_file, features)
    feature_files[year] = feature_file
    print(f"  ✓ Generated features for {year}")

# Copy existing visualizations if available
print("\n📁 Looking for existing visualizations...")
viz_files = [
    'test_output/quick_test/analysis/pca_kmeans_clusters.png',
    'analysis_output/mangalore_india_20260411_224738/before_after_comparison_before_after.png',
    'analysis_output/mangalore_india_20260411_224738/before_after_comparison_change_map.png',
    'analysis_output/mangalore_india_20260411_224738/before_after_comparison_predictions.png'
]

for viz_file in viz_files:
    if Path(viz_file).exists():
        # Copy to demo directory
        dest = demo_dir / Path(viz_file).name
        import shutil
        shutil.copy(viz_file, dest)
        print(f"  ✓ Copied {Path(viz_file).name}")

print("\n🎨 Generating interactive report...")
print("-" * 80)

# Create interactive report
report_path = create_interactive_report(
    analysis_dir=str(demo_dir),
    place_name="Mangalore, India",
    condition="vegetation",
    start_year=2020,
    end_year=2024,
    feature_files=feature_files,
    output_html=str(demo_dir / "VISUAL_REPORT.html")
)

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
print(f"\n📂 Report opened in your browser")
print(f"📁 Files saved to: {demo_dir}")
print("\n🎯 The report shows:")
print("  ✓ Real satellite map from Google/Esri")
print("  ✓ Before/after comparison images")
print("  ✓ Interactive trend charts")
print("  ✓ Future predictions (10 years)")
print("  ✓ Year-by-year analysis")
print("  ✓ Key findings & recommendations")
print("\n💡 If the browser didn't open automatically, open:")
print(f"   {report_path}")
print("=" * 80)
