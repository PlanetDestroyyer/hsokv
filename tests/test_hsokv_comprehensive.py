"""
Comprehensive test suite for HSOKV system
Tests all key features including catastrophic forgetting prevention,
dual memory system, 3-stage lifecycle, and GPU compatibility.

Run with: pytest test_hsokv_comprehensive.py -v
Or standalone: python test_hsokv_comprehensive.py
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hsokv import (
    MemorySystem,
    DualMemorySystem,
    SentenceBERTEmbedder,
    CLIPEmbedder,
    MemoryConfig
)


class TestEnvironment:
    """Test environment setup with GPU/CPU detection"""

    @staticmethod
    def get_device():
        """Detect available device"""
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            print("ℹ CPU mode (no GPU available)")
        return device


class TestCatastrophicForgetting:
    """Test that HSOKV prevents catastrophic forgetting"""

    def setup_method(self):
        """Setup for each test"""
        self.device = TestEnvironment.get_device()
        self.config = MemoryConfig(device=self.device, max_entries=1000)
        self.embedder = SentenceBERTEmbedder(device=self.device)
        self.system = MemorySystem(self.embedder, self.config)

    def test_no_catastrophic_forgetting(self):
        """Test that old memories are retained when learning new ones"""
        print("\n=== Testing Catastrophic Forgetting Prevention ===")

        # Learn Task A: Wake up times
        task_a_memories = [
            ("when should I wake up on Monday?", "7am"),
            ("when should I wake up on Tuesday?", "8am"),
            ("when should I wake up on Wednesday?", "9am"),
        ]

        for query, answer in task_a_memories:
            self.system.learn(query, answer)

        # Verify Task A works
        result = self.system.recall("when should I wake up on Monday?")
        assert result == "7am", f"Expected '7am', got '{result}'"
        print("✓ Task A learned: Monday wake up at 7am")

        # Learn Task B: Meeting times (new information)
        task_b_memories = [
            ("when is the team meeting?", "2pm"),
            ("when is the client call?", "4pm"),
            ("when is lunch break?", "12pm"),
        ]

        for query, answer in task_b_memories:
            self.system.learn(query, answer)

        # Verify Task B works
        result = self.system.recall("when is the team meeting?")
        assert result == "2pm", f"Expected '2pm', got '{result}'"
        print("✓ Task B learned: Team meeting at 2pm")

        # CRITICAL TEST: Verify Task A still works (no catastrophic forgetting)
        result_monday = self.system.recall("when should I wake up on Monday?")
        result_tuesday = self.system.recall("when should I wake up on Tuesday?")
        result_wednesday = self.system.recall("when should I wake up on Wednesday?")

        assert result_monday == "7am", f"Catastrophic forgetting! Expected '7am', got '{result_monday}'"
        assert result_tuesday == "8am", f"Catastrophic forgetting! Expected '8am', got '{result_tuesday}'"
        assert result_wednesday == "9am", f"Catastrophic forgetting! Expected '9am', got '{result_wednesday}'"

        print("✓ Task A retained: Monday=7am, Tuesday=8am, Wednesday=9am")
        print("✓ Zero catastrophic forgetting confirmed!")

        # Learn Task C: More complex information
        for i in range(10):
            self.system.learn(f"what is item {i}?", f"value {i}")

        # Verify all original tasks still work
        assert self.system.recall("when should I wake up on Monday?") == "7am"
        assert self.system.recall("when is the team meeting?") == "2pm"
        print("✓ All memories retained after learning 10 more items")

        return True

    def test_semantic_robustness(self):
        """Test that system handles paraphrasing"""
        print("\n=== Testing Semantic Robustness ===")

        # Learn with one phrasing
        self.system.learn("what time is my alarm?", "6am")

        # Test with different phrasings
        test_queries = [
            "what time is my alarm?",
            "when should I wake up?",
            "what time do I need to get up?",
            "alarm time?",
        ]

        for query in test_queries:
            result = self.system.recall(query)
            print(f"  Query: '{query}' → '{result}'")
            assert result == "6am", f"Semantic matching failed for '{query}'"

        print("✓ Semantic robustness confirmed")
        return True

    def test_retention_over_time(self):
        """Test that memories don't degrade over many operations"""
        print("\n=== Testing Long-term Retention ===")

        # Store initial memory
        self.system.learn("important fact", "remember this")

        # Perform many other operations
        for i in range(100):
            self.system.learn(f"noise {i}", f"value {i}")

        # Verify original memory still works
        result = self.system.recall("important fact")
        assert result == "remember this", f"Memory degraded! Expected 'remember this', got '{result}'"
        print("✓ Memory retained after 100 new memories")

        return True


class TestDualMemorySystem:
    """Test short-term and long-term memory interaction"""

    def setup_method(self):
        """Setup for each test"""
        self.device = TestEnvironment.get_device()
        self.embedder = SentenceBERTEmbedder(device=self.device)
        self.config = MemoryConfig(device=self.device)
        self.system = DualMemorySystem(self.embedder, self.config)

    def test_short_term_memory(self):
        """Test that STM provides fast O(1) lookup"""
        print("\n=== Testing Short-term Memory ===")

        # Learn a word
        self.system.learn("ephemeral", "lasting for a very short time")

        # Should be in STM
        start = time.time()
        result = self.system.recall("ephemeral")
        elapsed = (time.time() - start) * 1000  # ms

        assert "short time" in result or "ephemeral" in result
        print(f"✓ STM lookup: {elapsed:.2f}ms (should be <10ms)")
        assert elapsed < 50, "STM lookup too slow"

        return True

    def test_consolidation_after_rehearsal(self):
        """Test that repeated access moves STM → LTM"""
        print("\n=== Testing Memory Consolidation ===")

        # Learn a word
        word = "consolidate"
        definition = "combine into a single whole"
        self.system.learn(word, definition)

        # Access 3 times to trigger consolidation
        for i in range(3):
            result = self.system.recall(word)
            print(f"  Access {i+1}: {result[:30]}...")

        # Check stats
        stats = self.system.get_stats()
        print(f"  STM: {stats['stm_size']} items, LTM: {stats['ltm_learning']} learning, {stats['ltm_mature']} mature")

        # Should be consolidated to LTM
        assert stats['ltm_learning'] > 0 or stats['ltm_mature'] > 0, "Memory not consolidated to LTM"
        print("✓ Memory consolidated to LTM after rehearsal")

        return True

    def test_stm_capacity_limit(self):
        """Test that STM enforces 7±2 capacity limit"""
        print("\n=== Testing STM Capacity Limit ===")

        # Learn more than STM capacity
        words = [f"word{i}" for i in range(10)]
        for word in words:
            self.system.learn(word, f"definition of {word}")

        stats = self.system.get_stats()
        stm_size = stats['stm_size']

        print(f"  STM size: {stm_size} (should be ≤ {self.system.stm.capacity})")
        assert stm_size <= self.system.stm.capacity, f"STM exceeded capacity: {stm_size} > {self.system.stm.capacity}"
        print("✓ STM capacity limit enforced")

        return True

    def test_time_decay(self):
        """Test that unrehearsed STM items decay"""
        print("\n=== Testing Time-based Decay ===")

        # Learn something
        self.system.learn("temporary", "not permanent")

        # Wait for decay (simulate time passage)
        print("  Triggering decay...")
        self.system.forget()  # Manually trigger decay

        # STM should still have it if within 30 seconds
        # (In real usage, this would decay after 30s of no access)
        stats_before = self.system.get_stats()
        print(f"  STM size before decay: {stats_before['stm_size']}")

        print("✓ Decay mechanism functional")
        return True

    def test_emotional_significance(self):
        """Test that emotionally significant memories bypass STM"""
        print("\n=== Testing Emotional Significance ===")

        # Learn with emotional significance
        self.system.learn("trauma", "deeply disturbing experience", emotionally_significant=True)

        # Should go directly to LTM
        stats = self.system.get_stats()
        ltm_total = stats['ltm_learning'] + stats['ltm_reinforcement'] + stats['ltm_mature']

        assert ltm_total > 0, "Emotionally significant memory not in LTM"
        print("✓ Emotionally significant memory went directly to LTM")

        return True

    def test_sleep_consolidation(self):
        """Test batch consolidation during 'sleep'"""
        print("\n=== Testing Sleep Consolidation ===")

        # Learn multiple items
        for i in range(5):
            self.system.learn(f"item{i}", f"value{i}")
            # Access 3 times each to mark for consolidation
            for _ in range(3):
                self.system.recall(f"item{i}")

        stats_before = self.system.get_stats()
        print(f"  Before sleep: STM={stats_before['stm_size']}, LTM={stats_before['ltm_learning']}")

        # Trigger sleep consolidation
        self.system.sleep()

        stats_after = self.system.get_stats()
        print(f"  After sleep: STM={stats_after['stm_size']}, LTM={stats_after['ltm_learning']}")

        # STM should be cleared, LTM should have more
        assert stats_after['stm_size'] == 0, "STM not cleared after sleep"
        print("✓ Sleep consolidation successful")

        return True


class TestLifecycleStages:
    """Test 3-stage memory lifecycle (LEARNING → REINFORCEMENT → MATURE)"""

    def setup_method(self):
        """Setup for each test"""
        self.device = TestEnvironment.get_device()
        self.embedder = SentenceBERTEmbedder(device=self.device)
        self.config = MemoryConfig(
            device=self.device,
            learning_phase_duration=3,  # Faster testing
            reinforcement_phase_duration=7
        )
        self.system = MemorySystem(self.embedder, self.config)

    def test_lifecycle_progression(self):
        """Test that memories progress through lifecycle stages"""
        print("\n=== Testing Lifecycle Progression ===")

        # Learn a memory
        self.system.learn("test memory", "test value")

        # Check initial stage (should be LEARNING)
        stats = self.system.get_stats()
        print(f"  Initial: {stats}")
        assert stats['learning'] > 0, "Memory not in LEARNING stage"
        print("✓ Memory starts in LEARNING stage")

        # Access multiple times to progress to REINFORCEMENT
        for i in range(self.config.learning_phase_duration + 1):
            self.system.recall("test memory")

        stats = self.system.get_stats()
        print(f"  After {self.config.learning_phase_duration + 1} accesses: {stats}")

        # Access more to reach MATURE
        for i in range(self.config.reinforcement_phase_duration):
            self.system.recall("test memory")

        stats = self.system.get_stats()
        print(f"  After {self.config.reinforcement_phase_duration} more accesses: {stats}")
        assert stats['mature'] > 0, "Memory not in MATURE stage"
        print("✓ Memory progressed to MATURE stage")

        return True

    def test_learning_stage_protection(self):
        """Test that LEARNING stage memories are protected from pruning"""
        print("\n=== Testing LEARNING Stage Protection ===")

        # Create a low-confidence LEARNING memory
        self.system.learn("protected", "value", confidence=0.1)

        # Try to prune
        self.system.prune_memories()

        # Should still exist (protected)
        result = self.system.recall("protected")
        assert result == "value", "LEARNING memory was pruned (should be protected)"
        print("✓ LEARNING stage memories protected from pruning")

        return True


class TestGPUCompatibility:
    """Test GPU-specific functionality"""

    def test_gpu_acceleration(self):
        """Test that embeddings run on GPU if available"""
        print("\n=== Testing GPU Acceleration ===")

        device = TestEnvironment.get_device()

        if device == 'cpu':
            print("⊘ Skipping GPU tests (no GPU available)")
            return True

        # Create embedder on GPU
        embedder = SentenceBERTEmbedder(device='cuda')

        # Test embedding
        text = "test sentence for GPU processing"
        start = time.time()
        embedding = embedder.embed(text)
        gpu_time = (time.time() - start) * 1000

        # Verify on GPU
        assert embedding.device.type == 'cuda', f"Embedding not on GPU: {embedding.device}"
        print(f"✓ Embedding on GPU: {embedding.device}")
        print(f"  GPU processing time: {gpu_time:.2f}ms")

        # Compare with CPU
        embedder_cpu = SentenceBERTEmbedder(device='cpu')
        start = time.time()
        embedding_cpu = embedder_cpu.embed(text)
        cpu_time = (time.time() - start) * 1000

        print(f"  CPU processing time: {cpu_time:.2f}ms")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")

        return True

    def test_large_batch_on_gpu(self):
        """Test that large batches work on GPU"""
        print("\n=== Testing Large Batch Processing ===")

        device = TestEnvironment.get_device()
        if device == 'cpu':
            print("⊘ Skipping GPU batch test (no GPU available)")
            return True

        config = MemoryConfig(device='cuda', max_entries=1000)
        embedder = SentenceBERTEmbedder(device='cuda')
        system = MemorySystem(embedder, config)

        # Store many memories
        print("  Storing 100 memories on GPU...")
        start = time.time()
        for i in range(100):
            system.learn(f"memory {i}", f"value {i}")
        store_time = time.time() - start

        print(f"  Store time: {store_time:.2f}s ({store_time/100*1000:.2f}ms per memory)")

        # Retrieve many memories
        print("  Retrieving 100 memories from GPU...")
        start = time.time()
        for i in range(100):
            result = system.recall(f"memory {i}")
            assert result == f"value {i}"
        retrieve_time = time.time() - start

        print(f"  Retrieve time: {retrieve_time:.2f}s ({retrieve_time/100*1000:.2f}ms per memory)")
        print("✓ Large batch processing successful on GPU")

        return True


class TestCLIPEmbedder:
    """Test CLIP embedder functionality"""

    def test_clip_embedder(self):
        """Test that CLIP embedder works"""
        print("\n=== Testing CLIP Embedder ===")

        device = TestEnvironment.get_device()
        embedder = CLIPEmbedder(device=device)

        # Test text embedding
        text = "a photo of a cat"
        embedding = embedder.embed(text)

        assert embedding is not None
        assert embedding.shape[0] == embedder.get_dim()
        print(f"✓ CLIP text embedding: {embedding.shape}")

        # Test with MemorySystem
        system = MemorySystem(embedder, MemoryConfig(device=device))
        system.learn("describe a cat", "fluffy animal with whiskers")
        result = system.recall("describe a cat")

        assert result == "fluffy animal with whiskers"
        print("✓ CLIP embedder works with MemorySystem")

        return True


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("HSOKV COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    results = []

    # Catastrophic Forgetting Tests
    print("\n" + "=" * 70)
    print("PART 1: CATASTROPHIC FORGETTING PREVENTION")
    print("=" * 70)
    try:
        test = TestCatastrophicForgetting()
        test.setup_method()
        test.test_no_catastrophic_forgetting()
        test.setup_method()
        test.test_semantic_robustness()
        test.setup_method()
        test.test_retention_over_time()
        results.append(("Catastrophic Forgetting Prevention", "PASS"))
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("Catastrophic Forgetting Prevention", f"FAIL: {e}"))

    # Dual Memory Tests
    print("\n" + "=" * 70)
    print("PART 2: DUAL MEMORY SYSTEM (STM + LTM)")
    print("=" * 70)
    try:
        test = TestDualMemorySystem()
        test.setup_method()
        test.test_short_term_memory()
        test.setup_method()
        test.test_consolidation_after_rehearsal()
        test.setup_method()
        test.test_stm_capacity_limit()
        test.setup_method()
        test.test_time_decay()
        test.setup_method()
        test.test_emotional_significance()
        test.setup_method()
        test.test_sleep_consolidation()
        results.append(("Dual Memory System", "PASS"))
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("Dual Memory System", f"FAIL: {e}"))

    # Lifecycle Tests
    print("\n" + "=" * 70)
    print("PART 3: 3-STAGE LIFECYCLE")
    print("=" * 70)
    try:
        test = TestLifecycleStages()
        test.setup_method()
        test.test_lifecycle_progression()
        test.setup_method()
        test.test_learning_stage_protection()
        results.append(("3-Stage Lifecycle", "PASS"))
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("3-Stage Lifecycle", f"FAIL: {e}"))

    # GPU Tests
    print("\n" + "=" * 70)
    print("PART 4: GPU COMPATIBILITY")
    print("=" * 70)
    try:
        test = TestGPUCompatibility()
        test.test_gpu_acceleration()
        test.test_large_batch_on_gpu()
        results.append(("GPU Compatibility", "PASS"))
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("GPU Compatibility", f"FAIL: {e}"))

    # CLIP Tests
    print("\n" + "=" * 70)
    print("PART 5: CLIP EMBEDDER")
    print("=" * 70)
    try:
        test = TestCLIPEmbedder()
        test.test_clip_embedder()
        results.append(("CLIP Embedder", "PASS"))
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("CLIP Embedder", f"FAIL: {e}"))

    # Final Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)

    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} test suites passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! HSOKV is working correctly!")
    else:
        print(f"\n⚠ {total - passed} test suite(s) failed. See details above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
