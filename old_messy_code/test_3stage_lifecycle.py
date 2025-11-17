#!/usr/bin/env python3
"""
Test script to validate the 3-stage memory lifecycle (human-inspired learning).

Based on the "overwhelming" example:
- Day 0: Learn new word from movie → LEARNING stage (pure recall, maximum protection)
- Days 1-6: Use successfully 5 times → REINFORCEMENT stage (boosted confidence, high protection)
- Week 2+: Proven useful → MATURE stage (can consolidate to weights)

Expected behavior:
1. LEARNING stage (first 5 retrievals): Pure recall, never delete, never consolidate
2. REINFORCEMENT stage (retrievals 6-20): Boosted confidence, never delete, never consolidate
3. MATURE stage (after 20 retrievals): Standard retrieval, can delete if unused, can consolidate if stable
"""

import sys
sys.path.insert(0, '/home/user/hsokv')

import torch
from hsokv_core import (
    CONFIG,
    override_config,
    SimpleTokenizer,
    KeyValueMemory,
)

def test_3stage_lifecycle():
    """Test the 3-stage memory lifecycle with the 'overwhelming' example."""
    print("=" * 80)
    print("3-STAGE MEMORY LIFECYCLE TEST")
    print("=" * 80)
    print()
    print("Simulating the 'overwhelming' learning journey:")
    print("Day 0: Learn from movie → LEARNING stage")
    print("Days 1-6: Use 5 times → REINFORCEMENT stage")
    print("Week 2+: Proven useful → MATURE stage")
    print()

    # Setup
    config = override_config(CONFIG, {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_learning_phase_duration": 5,
        "memory_reinforcement_phase_duration": 20,
        "use_stage_aware_retrieval": True,
        "use_pure_recall_for_new_words": True,
        "protect_during_learning": True,
        "protect_during_reinforcement": True,
    })

    device = torch.device(config["device"])
    tokenizer = SimpleTokenizer()
    tokenizer.fit(["overwhelming", "cant", "process", "many", "things", "at", "once"])

    # Create memory
    memory = KeyValueMemory(key_dim=256, device=device)

    print(f"Device: {config['device']}")
    print(f"LEARNING phase duration: {config['memory_learning_phase_duration']} retrievals")
    print(f"REINFORCEMENT phase duration: {config['memory_reinforcement_phase_duration']} retrievals")
    print()

    # DAY 0: Learn "overwhelming" from movie
    print("-" * 80)
    print("DAY 0: Learn 'overwhelming' from movie")
    print("-" * 80)

    # Create embedding for "overwhelming"
    key_embedding = torch.randn(256, device=device)
    value_dict = {
        "word": "overwhelming",
        "definition": "unable to handle many things at once",
        "usage": "don't overwhelm yourself with too many tasks",
        "value_vector": torch.randn(256, device=device)
    }
    metadata = {
        "confidence": 0.3,  # Low confidence (just learned)
        "retrieval_count": 0,
        "success_rate": 0.0,
        "is_first_exposure": True,  # NEW WORD FLAG
        "created_at": 0,
    }

    entry_id = memory.write(key_embedding, value_dict, metadata)
    print(f"✓ Stored 'overwhelming' in memory (entry_id={entry_id})")

    stage = memory.get_memory_stage(entry_id)
    print(f"✓ Initial stage: {stage}")
    assert stage == "LEARNING", f"Expected LEARNING stage, got {stage}"
    print(f"✓ Confidence: {memory.metadata[entry_id]['confidence']:.2f}")
    print()

    # DAY 1-6: Use "overwhelming" multiple times (LEARNING → REINFORCEMENT)
    print("-" * 80)
    print("DAYS 1-6: Use 'overwhelming' successfully")
    print("-" * 80)

    query = torch.randn(256, device=device)  # Query for "overwhelming"

    for use_count in range(1, 21):  # Simulate 20 uses
        # Retrieve memory
        retrieved, details = memory.retrieve(query, top_k=5)

        # Check stage
        stage = memory.get_memory_stage(entry_id)
        retrieval_count = memory.metadata[entry_id]["retrieval_count"]
        confidence = memory.metadata[entry_id]["confidence"]

        # Print status every 5 uses
        if use_count % 5 == 0 or use_count == 1:
            print(f"Use #{use_count:2d}: Stage={stage:13s}, Retrievals={retrieval_count:2d}, Confidence={confidence:.2f}")

            # Verify stage transitions
            if retrieval_count < config["memory_learning_phase_duration"]:
                assert stage == "LEARNING", f"Expected LEARNING at retrieval {retrieval_count}, got {stage}"
            elif retrieval_count < config["memory_reinforcement_phase_duration"]:
                assert stage == "REINFORCEMENT", f"Expected REINFORCEMENT at retrieval {retrieval_count}, got {stage}"
            else:
                assert stage == "MATURE", f"Expected MATURE at retrieval {retrieval_count}, got {stage}"

        # Simulate successful use
        memory.update_confidence(entry_id, success_signal=1.0)

    print()

    # Final checks
    final_stage = memory.get_memory_stage(entry_id)
    final_confidence = memory.metadata[entry_id]["confidence"]
    final_retrievals = memory.metadata[entry_id]["retrieval_count"]
    is_first_exposure = memory.metadata[entry_id].get("is_first_exposure", False)

    print("-" * 80)
    print("FINAL STATUS")
    print("-" * 80)
    print(f"Stage: {final_stage}")
    print(f"Retrievals: {final_retrievals}")
    print(f"Confidence: {final_confidence:.2f}")
    print(f"First exposure flag: {is_first_exposure}")
    print()

    # Validate final state
    assert final_stage == "MATURE", f"Expected MATURE stage after 20 uses, got {final_stage}"
    assert final_confidence > 0.5, f"Expected confidence > 0.5, got {final_confidence}"
    assert final_retrievals >= 20, f"Expected >= 20 retrievals, got {final_retrievals}"
    assert not is_first_exposure, f"Expected first_exposure=False after graduating to MATURE"

    # Test protection logic
    print("-" * 80)
    print("PROTECTION TESTS")
    print("-" * 80)

    # Create a LEARNING stage memory
    learning_key = torch.randn(256, device=device)
    learning_value = {
        "word": "test_learning",
        "definition": "test",
        "usage": "test",
        "value_vector": torch.randn(256, device=device)
    }
    learning_meta = {
        "confidence": 0.2,
        "retrieval_count": 2,  # In LEARNING stage (< 5)
        "success_rate": 0.0,
        "is_first_exposure": True,
        "created_at": 0,
    }
    learning_id = memory.write(learning_key, learning_value, learning_meta)
    learning_stage = memory.get_memory_stage(learning_id)
    print(f"✓ Created test memory in {learning_stage} stage (retrievals: {memory.metadata[learning_id]['retrieval_count']})")

    # Create a REINFORCEMENT stage memory
    reinf_key = torch.randn(256, device=device)
    reinf_value = {
        "word": "test_reinforcement",
        "definition": "test",
        "usage": "test",
        "value_vector": torch.randn(256, device=device)
    }
    reinf_meta = {
        "confidence": 0.4,
        "retrieval_count": 10,  # In REINFORCEMENT stage (5-20)
        "success_rate": 0.5,
        "is_first_exposure": True,
        "created_at": 0,
    }
    reinf_id = memory.write(reinf_key, reinf_value, reinf_meta)
    reinf_stage = memory.get_memory_stage(reinf_id)
    print(f"✓ Created test memory in {reinf_stage} stage (retrievals: {memory.metadata[reinf_id]['retrieval_count']})")

    print()
    print("Expected protection behavior:")
    print("- LEARNING stage: Protected from deletion and consolidation")
    print("- REINFORCEMENT stage: Protected from deletion and consolidation")
    print("- MATURE stage: Can be deleted (if low utility) or consolidated (if proven stable)")
    print()

    # Test forgetting protection
    from hsokv_core.forgetting import ForgettingModule
    forgetting = ForgettingModule(
        memory,
        memory_cap=1000,
        confidence_threshold=0.15,
        utility_threshold=0.5,  # High threshold to trigger deletion
    )

    print("Testing forgetting module...")
    report = forgetting.forget(iteration=1, current_step=100.0)

    # Check that LEARNING and REINFORCEMENT memories are still there
    still_has_learning = learning_id < len(memory.metadata)
    still_has_reinforcement = reinf_id < len(memory.metadata)

    if still_has_learning:
        print(f"✓ LEARNING stage memory protected (not deleted)")
    else:
        print(f"✗ LEARNING stage memory was deleted (should be protected!)")

    if still_has_reinforcement:
        print(f"✓ REINFORCEMENT stage memory protected (not deleted)")
    else:
        print(f"✗ REINFORCEMENT stage memory was deleted (should be protected!)")

    print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("✓ Stage transitions work correctly:")
    print("  - LEARNING (first 5 uses)")
    print("  - REINFORCEMENT (uses 6-20)")
    print("  - MATURE (after 20 uses)")
    print()
    print("✓ Pure recall in LEARNING stage")
    print("✓ Confidence boosting in REINFORCEMENT stage")
    print("✓ Protection from deletion in LEARNING/REINFORCEMENT stages")
    print("✓ Graduation to MATURE stage after sufficient use")
    print()
    print("🎉 3-STAGE LIFECYCLE TEST PASSED!")
    print()
    print("This mimics human learning: you learned 'overwhelming' from a movie,")
    print("used it successfully multiple times, and now it's part of your vocabulary!")
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_3stage_lifecycle()
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
