"""
Direct GLUE testing script - bypasses synthetic training.

This script trains and evaluates ONLY on GLUE data.
"""

import torch
from hsokv_core import (
    CONFIG,
    override_config,
    run_glue_benchmark,
    set_seed,
)

def main():
    # Configure for GLUE testing
    config = override_config(CONFIG, {
        "allow_dataset_download": True,
        "glue_task": "sst2",
        "glue_shots_per_class": 16,  # Few-shot learning
        "meta_iterations": 5,  # Demo preset
        "batch_size": 16,
        "flops_target": 1e9,
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("Testing H-SOKV on GLUE SST-2 (Sentiment Classification)")
    print("=" * 70)
    print(f"Task: SST-2 (binary sentiment classification)")
    print(f"Few-shot examples: {config['glue_shots_per_class']} per class")
    print(f"Training iterations: {config['meta_iterations']}")
    print(f"Device: {config['device']}")
    print("=" * 70)
    print()

    # Run GLUE benchmark (trains fresh model on GLUE data)
    results = run_glue_benchmark("sst2", config)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for variant_name, result in results.items():
        print(f"\n{variant_name}:")
        print(f"  Accuracy:  {result.accuracy:.3f}")
        print(f"  Retention: {result.retention:.3f}")
        print(f"  FLOPs:     {result.flops/1e6:.2f}M")

    # Check if KV memory was used effectively
    if "hsokv" in results:
        hsokv_result = results["hsokv"]
        print(f"\n{'='*70}")
        print("3-STAGE LIFECYCLE VALIDATION")
        print(f"{'='*70}")

        if hsokv_result.accuracy > 0.60:
            print("✅ PASS: Accuracy > 60% (3-stage lifecycle helping!)")
        else:
            print("❌ FAIL: Accuracy < 60% (3-stage lifecycle not working)")

        if hsokv_result.retention > 0.90:
            print("✅ PASS: Retention > 90% (MATURE memories preserved)")
        else:
            print("❌ FAIL: Retention < 90% (Memory consolidation issue)")

    print()

if __name__ == "__main__":
    main()
