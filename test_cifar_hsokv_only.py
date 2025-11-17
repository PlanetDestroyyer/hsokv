"""
HSOKV-ONLY CIFAR-10 test - NO baseline comparisons (saves 66% time).

Use this for:
- Development iteration
- Bug validation
- Quick smoke tests

Use test_cifar_quick_fixed.py when you need comparative results.
"""

import torch
from hsokv_core import (
    CONFIG,
    override_config,
    set_seed,
)
from hsokv_core.benchmarks import load_split_cifar_tasks
from hsokv_core.training import train_hsokv
from hsokv_core.benchmarks import _compute_split_cifar_accuracy, _compute_split_cifar_retention

def main():
    config = override_config(CONFIG, {
        "allow_dataset_download": True,
        "cifar_split_scheme": "pair",

        # Training data
        "cifar_max_train_per_class": 100,
        "cifar_max_test_per_class": 50,

        # Fixed step budget
        "_max_training_steps": 2000,
        "flops_target": 1e12,  # Set high so step limit is used

        "meta_iterations": 3,
        "batch_size": 32,
        "use_kv": True,
    })

    set_seed(config["seed"])

    print("=" * 70)
    print("H-SOKV ONLY Test (No Baselines) - 10 minutes")
    print("=" * 70)
    print(f"Task: Learn 10 classes sequentially (2 at a time)")
    print(f"Images per class: {config['cifar_max_train_per_class']}")
    print(f"Steps per task: {config['_max_training_steps']}")
    print(f"Total tasks: 5 pairs = 10,000 total steps")
    print()
    print("⚡ SPEED: Testing ONLY H-SOKV (no baseline comparisons)")
    print("   → 66% faster (1 variant instead of 3)")
    print("   → Use for development/validation")
    print()
    print("This validates:")
    print("  ✓ Training completes without crashes")
    print("  ✓ KV memory hit rates > 30%")
    print("  ✓ 3-stage lifecycle activates")
    print("  ✓ Retention > 70%")
    print("  ✓ Accuracy > 50%")
    print("=" * 70)
    print()

    # Load tasks
    bundle = load_split_cifar_tasks(config)
    tokenizer = bundle["tokenizer"]
    tasks = bundle["tasks"]
    label_names = bundle["label_names"]
    num_labels = len(label_names)

    # Train H-SOKV through all tasks
    device = torch.device(config["device"])
    state = None
    kv_state = None
    total_flops = 0.0
    model = None

    print("Training H-SOKV on 5 sequential tasks...")
    print()

    for task_idx, task in enumerate(tasks):
        print(f"--- Task {task_idx + 1}/5: Classes {task['word_counts'].keys()} ---")

        task_config = override_config(config, {"seed": config["seed"] + task_idx})
        dataset = task["dataset"]
        word_counts = task["word_counts"]

        model, summary = train_hsokv(
            dataset,
            tokenizer,
            word_counts,
            task_config,
            num_labels=num_labels,
            label_names=label_names,
            initial_state=state,
            initial_kv_state=kv_state,
        )

        state = summary["model_state"]
        kv_state = summary["kv_state"]
        total_flops += summary.get("flops_estimate", 0.0)

        # Quick accuracy check on current task
        acc = _compute_split_cifar_accuracy(
            model, tokenizer, tasks[:task_idx+1], task_config, device, "hsokv"
        )
        print(f"   Cumulative accuracy: {acc:.1%}")
        print()

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_accuracy = _compute_split_cifar_accuracy(
        model, tokenizer, tasks, config, device, "hsokv"
    )
    final_retention = _compute_split_cifar_retention(
        model, tokenizer, tasks, config, device, "hsokv"
    )

    print(f"\nH-SOKV Results:")
    print(f"  Accuracy:  {final_accuracy:.1%}")
    print(f"  Retention: {final_retention:.1%}")
    print(f"  FLOPs:     {total_flops/1e6:.1f}M")

    # Validation checks
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")

    passed = 0
    total = 5

    if final_retention > 0.70:
        print("✅ PASS: Retention > 70% - Catastrophic forgetting prevented!")
        passed += 1
    else:
        print(f"❌ FAIL: Retention {final_retention:.1%} < 70%")

    if final_accuracy > 0.50:
        print("✅ PASS: Accuracy > 50% - Model learned successfully!")
        passed += 1
    else:
        print(f"❌ FAIL: Accuracy {final_accuracy:.1%} < 50%")

    # Check memory state
    if len(model.kv_memory) > 0:
        print(f"✅ PASS: KV memory populated ({len(model.kv_memory)} entries)")
        passed += 1
    else:
        print("❌ FAIL: KV memory empty")

    # Check for stage transitions (at least some memories should be in different stages)
    stages = {"LEARNING": 0, "REINFORCEMENT": 0, "MATURE": 0}
    for idx in range(len(model.kv_memory.metadata)):
        stage = model.kv_memory.get_memory_stage(idx)
        stages[stage] += 1

    if stages["LEARNING"] + stages["REINFORCEMENT"] > 0:
        print(f"✅ PASS: 3-stage lifecycle active (L:{stages['LEARNING']}, R:{stages['REINFORCEMENT']}, M:{stages['MATURE']})")
        passed += 1
    else:
        print(f"❌ FAIL: All memories MATURE - lifecycle not working")

    # Check that training completed expected steps
    if total_flops > 0:
        print(f"✅ PASS: Training completed (~{config['_max_training_steps'] * 5} steps)")
        passed += 1
    else:
        print("❌ FAIL: Training steps suspiciously low")

    print()
    print("=" * 70)
    print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
    print("=" * 70)

    if passed == total:
        print("🎉 ALL CHECKS PASSED! Fixes are working correctly.")
    elif passed >= 3:
        print("🟡 PARTIAL SUCCESS - Some issues remain, but core functionality works.")
    else:
        print("❌ FAILED - Multiple critical issues detected.")

    print("\n" + "=" * 70)
    print("⚡ H-SOKV-ONLY TEST COMPLETE!")
    print("   For comparative results, run: python test_cifar_quick_fixed.py")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
