#!/usr/bin/env python3
"""
Run real-world benchmark: HSOKV vs actual fine-tuned SLMs

Tests HSOKV against REAL small language models:
- SmolLM2-135M/360M
- Llama 3.2 1B
- Qwen2.5-0.5B/1.5B

On REAL continual learning benchmarks:
- AGNews sequential classification
- Multi-domain text classification

This provides rigorous scientific validation of HSOKV's claims.

Usage: python run_realworld_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'benchmarks'))

print("\n" + "="*70)
print("REAL-WORLD BENCHMARK SUITE")
print("HSOKV vs Fine-tuned Small Language Models")
print("="*70)
print("\nThis benchmark:")
print("  ✓ Uses REAL language models (SmolLM2, Llama, Qwen)")
print("  ✓ Actually fine-tunes them on sequential tasks")
print("  ✓ Uses REAL datasets (AGNews, 20Newsgroups)")
print("  ✓ Measures ACTUAL catastrophic forgetting")
print("\nEstimated time: 10-30 minutes on GPU")
print("="*70 + "\n")

try:
    from benchmark_realworld_slms import main as run_benchmark
    results = run_benchmark()

    print("\n" + "="*70)
    print("✓ REAL-WORLD BENCHMARK COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  📄 benchmarks/realworld_benchmark_results.json")
    print("\nNext steps:")
    print("  1. Review results to see catastrophic forgetting in REAL models")
    print("  2. Use these results in your paper/publication")
    print("  3. This provides rigorous scientific validation!")
    print("="*70 + "\n")

except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure you have installed real-world benchmark dependencies:")
    print("  pip install datasets accelerate")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  - Ensure you have enough GPU memory (at least 8GB)")
    print("  - Try reducing batch sizes or using smaller models")
    print("  - Check internet connection for downloading models")
    sys.exit(1)
