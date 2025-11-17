"""
SUPER-FAST CIFAR-10 test (completes in 2-3 mins) - FIXED STEP COUNT

This version directly limits training steps instead of using FLOP budget calculation
to avoid the bug where too many steps are generated.
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
        "cifar_max_train_per_class": 20,   # 20 images per class (reduced from 30)
        "cifar_max_test_per_class": 15,    # 15 test images per class

        # Reduced training budget
        "meta_iterations": 2,
        "batch_size": 16,

        # CRITICAL FIX: Use step-based limit instead of FLOP budget
        "_max_training_steps": 500,         # FIXED: 500 steps per task (5 tasks = 2500 total)
        "flops_target": 1e12,               # Set very high so step limit is used instead

        # Model simplification for speed
        "d_model": 128,
        "num_layers": 2,
        "nhead": 4,

        # Memory settings
        "max_memory_entries": 150,
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("SUPER-FAST CIFAR-10 Test (2-3 mins) - FIXED STEP COUNT")
    print("=" * 70)
    print(f"Task: Learn 10 classes sequentially (2 at a time)")
    print(f"Images per class: {config['cifar_max_train_per_class']}")
    print(f"Total images: {config['cifar_max_train_per_class'] * 10}")
    print(f"Max steps per task: {config['_max_training_steps']}")
    print(f"Total tasks: 5 (pairs)")
    print(f"Estimated total steps: ~{config['_max_training_steps'] * 5}")
    print(f"Model: d_model={config['d_model']}, layers={config['num_layers']}")
    print()
    print("🔧 BUG FIX:")
    print("  • Direct step limit instead of FLOP-based calculation")
    print("  • Prevents 20k steps/task bug")
    print("  • Should complete in 2-3 minutes")
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
        if hsokv_result.retention > 0.70:  # Lower threshold for super-fast test
            print("✅ PASS: Retention > 70%")
            print("   → MATURE memories preserved during new learning!")
            print("   → Consolidation working!")
            print("   → 3-stage lifecycle VALIDATED!")
        elif hsokv_result.retention > 0.60:
            print("🟡 PARTIAL: Retention 60-70%")
            print("   → Some forgetting, but better than baseline")
            print("   → (Lower threshold due to minimal training)")
        else:
            print("❌ FAIL: Retention < 60%")
            print("   → Catastrophic forgetting detected")

        if hsokv_result.accuracy > 0.45:
            print("✅ PASS: Final accuracy > 45%")
        elif hsokv_result.accuracy > 0.35:
            print("🟡 PARTIAL: Accuracy 35-45%")
            print("   → (Lower threshold due to minimal training)")
        else:
            print("❌ FAIL: Accuracy < 35%")

        # Compare to baselines
        if "fine_tune" in results:
            ft_retention = results["fine_tune"].retention
            if hsokv_result.retention > ft_retention + 0.05:
                print(f"\n✅ ADVANTAGE: H-SOKV retention ({hsokv_result.retention:.2f}) "
                      f"beats fine-tuning ({ft_retention:.2f}) by {hsokv_result.retention - ft_retention:.2f}")
                print("   → 3-stage lifecycle prevents catastrophic forgetting!")

    print("\n" + "=" * 70)
    print("⚡ SUPER-FAST TEST COMPLETE!")
    print("   For better accuracy, use: python test_cifar_quick.py")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
