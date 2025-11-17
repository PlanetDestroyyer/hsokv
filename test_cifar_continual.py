"""
Test on CIFAR-10 continual learning task.

This is the BEST test for the 3-stage lifecycle because:
1. Model learns classes sequentially (airplane → car → bird → ...)
2. Must retain old class memories while learning new ones
3. Tests if MATURE memories persist (consolidation)
4. Tests if forgetting is controlled
5. Cannot memorize everything in weights
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
        "cifar_split_scheme": "pair",  # Learn 2 classes at a time
        "cifar_max_train_per_class": 500,  # Use 500 images per class
        "meta_iterations": 5,
        "batch_size": 32,
        "flops_target": 5e9,
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("Testing H-SOKV on CIFAR-10 Continual Learning")
    print("=" * 70)
    print(f"Task: Learn 10 classes sequentially (2 at a time)")
    print(f"Split scheme: {config['cifar_split_scheme']}")
    print(f"Images per class: {config['cifar_max_train_per_class']}")
    print(f"Iterations: {config['meta_iterations']}")
    print()
    print("This tests:")
    print("  - LEARNING stage: Initial memory formation")
    print("  - REINFORCEMENT stage: Practice with new classes")
    print("  - MATURE stage: Retention of old classes")
    print("  - Consolidation: Transfer to permanent weights")
    print("  - Forgetting control: Don't delete important memories")
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
        if hsokv_result.retention > 0.85:
            print("✅ PASS: Retention > 85%")
            print("   → MATURE memories preserved during new learning!")
            print("   → Consolidation working!")
            print("   → 3-stage lifecycle VALIDATED!")
        else:
            print("❌ FAIL: Retention < 85%")
            print("   → Catastrophic forgetting detected")
            print("   → Consolidation may not be working")

        if hsokv_result.accuracy > 0.70:
            print("✅ PASS: Final accuracy > 70%")
        else:
            print("❌ FAIL: Final accuracy < 70%")

    print()

if __name__ == "__main__":
    main()
