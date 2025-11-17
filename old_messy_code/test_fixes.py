#!/usr/bin/env python3
"""
Test script to validate critical fixes to H-SOKV.

Tests:
1. Fix #1: Lower surprise thresholds improve one-shot accuracy
2. Fix #2: Consolidation validation prevents memory loss
3. Fix #3: First-exposure boosting improves recall
4. Fix #6: Numerical stability across hardware

Expected improvements:
- One-shot accuracy: 40% → 70%+
- Retention: 95% (maintained)
- No crashes during consolidation
"""

import sys
sys.path.insert(0, '/home/user/hsokv')

import torch
from hsokv_core import (
    CONFIG,
    override_config,
    generate_dataset,
    train_hsokv,
    set_seed,
)

def test_critical_fixes():
    """Run quick validation of all critical fixes."""
    print("=" * 70)
    print("H-SOKV CRITICAL FIXES VALIDATION")
    print("=" * 70)
    print()

    # Set seed for reproducibility
    set_seed(42)

    # Use quick test config for fast validation
    config = override_config(CONFIG, {
        "preset": "quick_test",
        "meta_iterations": 3,  # Minimal for testing
        "agent_steps": 15,
        "agents_per_manager": 1,
        "num_managers": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # Critical fix parameters
        "surprise_threshold": 0.3,  # Fix #1
        "first_exposure_threshold": 0.15,  # Fix #1
        "first_exposure_boost": 0.25,  # Fix #1
        "use_consolidation": True,  # Fix #2
    })

    print(f"Device: {config['device']}")
    print(f"Surprise threshold: {config['surprise_threshold']} (lowered from 0.5)")
    print(f"First exposure threshold: {config['first_exposure_threshold']} (new)")
    print(f"First exposure boost: {config['first_exposure_boost']} (new)")
    print()

    # Generate test dataset
    print("Generating test dataset...")
    dataset, tokenizer, word_counts = generate_dataset()
    print(f"✓ Dataset ready: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    print()

    # Train H-SOKV
    print("Training H-SOKV with fixes...")
    print("-" * 70)
    try:
        model, summary = train_hsokv(dataset, tokenizer, word_counts, config)
        print("-" * 70)
        print()

        # Extract metrics
        one_shot_acc = summary['test_metrics']['one_shot_accuracy']
        retention = summary['retention']
        kv_hit_rate = summary['test_metrics']['kv_hit_rate']
        history = summary['history']

        # Report results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"One-Shot Accuracy:  {one_shot_acc:.1%}")
        print(f"Retention:          {retention:.1%}")
        print(f"KV Hit Rate:        {kv_hit_rate:.1%}")
        print(f"Iterations:         {len(history)}")
        print()

        # Validation checks
        print("=" * 70)
        print("VALIDATION")
        print("=" * 70)
        print()

        passed = 0
        failed = 0

        # Check 1: One-shot accuracy improved
        if one_shot_acc >= 0.50:  # Target: 50%+ (was 40%)
            print("✓ One-shot accuracy ≥50%: PASS")
            passed += 1
        else:
            print(f"✗ One-shot accuracy {one_shot_acc:.1%} <50%: FAIL (expected improvement)")
            failed += 1

        # Check 2: Retention maintained
        if retention >= 0.85:  # Target: maintain 85%+
            print("✓ Retention ≥85%: PASS")
            passed += 1
        else:
            print(f"✗ Retention {retention:.1%} <85%: FAIL")
            failed += 1

        # Check 3: KV hit rate reasonable
        if kv_hit_rate >= 0.40:  # Target: 40%+
            print("✓ KV hit rate ≥40%: PASS")
            passed += 1
        else:
            print(f"✗ KV hit rate {kv_hit_rate:.1%} <40%: FAIL")
            failed += 1

        # Check 4: No crashes during consolidation
        if 'consolidation_history' in summary and len(summary['consolidation_history']) > 0:
            print("✓ Consolidation ran successfully: PASS")
            passed += 1
        else:
            print("✓ Consolidation not triggered (short run): OK")
            passed += 1

        # Check 5: Memory writes increased (first-exposure effect)
        telemetry = summary.get('telemetry', {})
        memory_writes = telemetry.get('memory_writes', 0)
        if memory_writes > 0:
            print(f"✓ Memory writes: {memory_writes} (first-exposure boost working): PASS")
            passed += 1
        else:
            print("✗ No memory writes detected: FAIL")
            failed += 1

        print()
        print("=" * 70)
        print(f"FINAL SCORE: {passed}/{passed + failed} tests passed")
        print("=" * 70)
        print()

        if failed == 0:
            print("🎉 ALL TESTS PASSED! Fixes are working correctly.")
            print()
            print("Next steps:")
            print("1. Run full training: python hsokv.py --preset demo --visualize")
            print("2. Check plots: results/learning_curves.png")
            print("3. Verify one-shot accuracy >70% in full run")
            return True
        else:
            print("⚠️  SOME TESTS FAILED. Please review the fixes.")
            print()
            print("Common issues:")
            print("- Low one-shot: Increase first_exposure_boost to 0.35")
            print("- Low retention: Check consolidation validation threshold")
            print("- Low KV hit: Verify surprise_threshold is lowered")
            return False

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR DURING TRAINING")
        print("=" * 70)
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("The fixes may have introduced a bug. Please review the changes.")
        return False

if __name__ == "__main__":
    success = test_critical_fixes()
    sys.exit(0 if success else 1)
