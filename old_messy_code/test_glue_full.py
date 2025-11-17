"""
Test on FULL GLUE training set (not just 16-shot).

This uses thousands of training examples, making memorization impossible.
"""

import torch
from hsokv_core import (
    CONFIG,
    override_config,
    run_glue_benchmark,
    set_seed,
)

def main():
    # Use FULL training set, not few-shot
    config = override_config(CONFIG, {
        "allow_dataset_download": True,
        "glue_task": "sst2",
        "glue_shots_per_class": 2000,  # Use 2000 examples per class!
        "glue_max_train_examples": 4000,  # Total 4000 examples
        "meta_iterations": 5,
        "batch_size": 16,
        "flops_target": 5e9,  # More compute budget
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("Testing H-SOKV on FULL GLUE SST-2 Dataset")
    print("=" * 70)
    print(f"Training examples: {config['glue_max_train_examples']}")
    print(f"Examples per class: {config['glue_shots_per_class']}")
    print(f"Iterations: {config['meta_iterations']}")
    print(f"FLOP budget: {config['flops_target']:.2e}")
    print("=" * 70)
    print()

    # Run GLUE benchmark
    results = run_glue_benchmark("sst2", config)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for variant_name, result in results.items():
        print(f"\n{variant_name}:")
        print(f"  Accuracy:  {result.accuracy:.3f}")
        print(f"  Retention: {result.retention:.3f}")
        print(f"  FLOPs:     {result.flops/1e6:.2f}M")

    print()

if __name__ == "__main__":
    main()
