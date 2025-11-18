#!/usr/bin/env python3
"""
Run HSOKV benchmarks and generate visualizations

This script:
1. Runs catastrophic forgetting benchmark (HSOKV vs Fine-tuning)
2. Generates comparison visualizations
3. Displays summary results

Usage: python run_benchmarks.py
"""

import sys
from pathlib import Path

# Add benchmarks directory to path
sys.path.insert(0, str(Path(__file__).parent / 'benchmarks'))

print("\n" + "="*70)
print("HSOKV BENCHMARK SUITE")
print("Comparing HSOKV vs Traditional Fine-tuning on Sequential Learning")
print("="*70 + "\n")

try:
    # Step 1: Run benchmark
    print("Step 1/2: Running benchmark...")
    print("(This will take 2-5 minutes on GPU, 5-15 minutes on CPU)\n")

    from benchmark_catastrophic_forgetting import main as run_benchmark
    run_benchmark()

    # Step 2: Generate visualizations
    print("\n\nStep 2/2: Generating visualizations...\n")

    from visualize_results import generate_all_plots
    generate_all_plots()

    # Success message
    print("\n" + "="*70)
    print("✓ BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files in benchmarks/:")
    print("  📊 forgetting_comparison.png - Main result (use in papers!)")
    print("  📊 all_tasks_comparison.png - Task-by-task comparison")
    print("  📊 accuracy_heatmap.png - Accuracy evolution")
    print("  📊 metrics_comparison.png - Key metrics")
    print("  📄 benchmark_results.json - Raw data")
    print("\nNext steps:")
    print("  1. Review the graphs to see catastrophic forgetting prevention")
    print("  2. Include results in your paper/documentation")
    print("  3. Share on social media to demonstrate innovation")
    print("="*70 + "\n")

except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure you have installed visualization dependencies:")
    print("  pip install matplotlib seaborn")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
