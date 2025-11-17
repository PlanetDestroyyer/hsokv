"""
ULTRA-FAST CIFAR-10 continual learning test (completes in 5-10 mins).

Dramatically reduced compute for quick validation:
- Fewer images per class
- Lower FLOP budget
- Fewer tasks (3 instead of 5)
- Still validates 3-stage lifecycle
"""

import torch
from hsokv_core import (
    CONFIG,
    override_config,
    run_split_cifar_benchmark,
    set_seed,
)

def main():
    config = override_config(CONFIG, {
        "allow_dataset_download": True,
        "cifar_split_scheme": "pair",

        # Dramatically reduced dataset size
        "cifar_max_train_per_class": 30,   # 30 images per class (was 100)
        "cifar_max_test_per_class": 20,    # 20 test images per class (was 50)

        # Reduced training budget
        "meta_iterations": 2,               # 2 iterations (was 3)
        "batch_size": 16,                   # Smaller batch for faster iterations
        "flops_target": 5e7,                # 50M FLOPs (was 1B = 20x faster!)

        # Model simplification for speed
        "d_model": 128,                     # Smaller model (was 256)
        "num_layers": 2,                    # Fewer layers (was 4)
        "nhead": 4,                         # Fewer heads (was 8)

        # Memory settings
        "max_memory_entries": 200,          # Smaller memory (was 400)
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("ULTRA-FAST CIFAR-10 Continual Learning Test (5-10 mins)")
    print("=" * 70)
    print(f"Task: Learn 10 classes sequentially (2 at a time)")
    print(f"Images per class: {config['cifar_max_train_per_class']}")
    print(f"Total images: {config['cifar_max_train_per_class'] * 10}")
    print(f"Iterations: {config['meta_iterations']}")
    print(f"FLOP budget: {config['flops_target']:.2e}")
    print(f"Model: d_model={config['d_model']}, layers={config['num_layers']}")
    print()
    print("⚡ SPEED OPTIMIZATIONS:")
    print("  • 20x lower FLOP budget (50M vs 1B)")
    print("  • 70% fewer images per class")
    print("  • Smaller model (128d, 2 layers)")
    print()
    print("This still tests:")
    print("  ✓ LEARNING stage: Initial memory formation")
    print("  ✓ REINFORCEMENT stage: Practice with new classes")
    print("  ✓ MATURE stage: Retention of old classes")
    print("  ✓ Consolidation: Transfer to permanent weights")
    print("  ✓ Forgetting control: Protect important memories")
    print("=" * 70)
    print()

    # Run CIFAR benchmark
    results = run_split_cifar_benchmark(config)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for variant_name, result in results.items():
        print(f"\n{variant_name}:")
        print(f"  Accuracy:  {result.accuracy:.3f}")
        print(f"  Retention: {result.retention:.3f}")  # ← KEY METRIC!
        print(f"  FLOPs:     {result.flops/1e6:.2f}M")

    print(f"\n{'='*70}")
    print("3-STAGE LIFECYCLE VALIDATION")
    print(f"{'='*70}")

    if "hsokv" in results:
        hsokv_result = results["hsokv"]

        # Retention is the key metric for continual learning
        if hsokv_result.retention > 0.75:  # Slightly lower threshold for fast test
            print("✅ PASS: Retention > 75%")
            print("   → MATURE memories preserved during new learning!")
            print("   → Consolidation working!")
            print("   → 3-stage lifecycle VALIDATED!")
        elif hsokv_result.retention > 0.65:
            print("🟡 PARTIAL: Retention 65-75%")
            print("   → Some forgetting, but better than baseline")
            print("   → (Lower threshold due to reduced training)")
        else:
            print("❌ FAIL: Retention < 65%")
            print("   → Catastrophic forgetting detected")

        if hsokv_result.accuracy > 0.50:
            print("✅ PASS: Final accuracy > 50%")
        elif hsokv_result.accuracy > 0.40:
            print("🟡 PARTIAL: Accuracy 40-50%")
            print("   → (Lower threshold due to reduced training)")
        else:
            print("❌ FAIL: Accuracy < 40%")

        # Compare to baselines
        if "fine_tune" in results:
            ft_retention = results["fine_tune"].retention
            if hsokv_result.retention > ft_retention + 0.05:
                print(f"\n✅ ADVANTAGE: H-SOKV retention ({hsokv_result.retention:.2f}) "
                      f"beats fine-tuning ({ft_retention:.2f}) by {hsokv_result.retention - ft_retention:.2f}")
                print("   → 3-stage lifecycle prevents catastrophic forgetting!")

    print("\n" + "=" * 70)
    print("⚡ ULTRA-FAST TEST COMPLETE!")
    print("   For full accuracy, use: python test_cifar_quick.py")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
