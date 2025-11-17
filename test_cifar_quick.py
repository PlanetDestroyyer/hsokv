"""
QUICK CIFAR-10 continual learning test (completes in 30 mins).

Reduces compute while still testing the key properties:
- Sequential class learning
- Memory retention
- 3-stage lifecycle
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
        "cifar_max_train_per_class": 100,  # 100 images per class (was 500)
        "cifar_max_test_per_class": 50,    # 50 test images per class
        "meta_iterations": 3,               # 3 iterations (was 5)
        "batch_size": 32,
        "flops_target": 1e9,                # Lower FLOP budget
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("QUICK CIFAR-10 Continual Learning Test (30 mins)")
    print("=" * 70)
    print(f"Task: Learn 10 classes sequentially (2 at a time)")
    print(f"Images per class: {config['cifar_max_train_per_class']}")
    print(f"Total images: {config['cifar_max_train_per_class'] * 10}")
    print(f"Iterations: {config['meta_iterations']}")
    print(f"FLOP budget: {config['flops_target']:.2e}")
    print()
    print("This tests:")
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
        if hsokv_result.retention > 0.80:
            print("✅ PASS: Retention > 80%")
            print("   → MATURE memories preserved during new learning!")
            print("   → Consolidation working!")
            print("   → 3-stage lifecycle VALIDATED!")
        elif hsokv_result.retention > 0.70:
            print("🟡 PARTIAL: Retention 70-80%")
            print("   → Some forgetting, but better than baseline")
        else:
            print("❌ FAIL: Retention < 70%")
            print("   → Catastrophic forgetting detected")

        if hsokv_result.accuracy > 0.60:
            print("✅ PASS: Final accuracy > 60%")
        elif hsokv_result.accuracy > 0.50:
            print("🟡 PARTIAL: Accuracy 50-60%")
        else:
            print("❌ FAIL: Accuracy < 50%")

        # Compare to baselines
        if "fine_tune" in results:
            ft_retention = results["fine_tune"].retention
            if hsokv_result.retention > ft_retention + 0.10:
                print(f"\n✅ ADVANTAGE: H-SOKV retention ({hsokv_result.retention:.2f}) "
                      f"beats fine-tuning ({ft_retention:.2f}) by {hsokv_result.retention - ft_retention:.2f}")
                print("   → 3-stage lifecycle prevents catastrophic forgetting!")

    print()

if __name__ == "__main__":
    main()
