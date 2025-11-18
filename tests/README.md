# HSOKV Test Suite

Comprehensive test suite to verify that HSOKV really works as advertised.

## What Gets Tested

### 1. Catastrophic Forgetting Prevention
- ✓ Learn Task A, learn Task B, verify Task A still works
- ✓ 100+ sequential learning operations with zero forgetting
- ✓ Semantic robustness (paraphrasing)
- ✓ Long-term retention

### 2. Dual Memory System
- ✓ Short-term memory (STM): Fast O(1) lookup
- ✓ Long-term memory (LTM): Semantic search
- ✓ Consolidation: 3+ accesses → STM to LTM
- ✓ Capacity limits: 7±2 items in STM
- ✓ Time decay: Unrehearsed items forgotten
- ✓ Emotional significance: Direct LTM storage
- ✓ Sleep consolidation: Batch processing

### 3. Three-Stage Lifecycle
- ✓ LEARNING → REINFORCEMENT → MATURE progression
- ✓ Protection from pruning in early stages
- ✓ Confidence boosting per stage

### 4. GPU Compatibility
- ✓ Automatic GPU detection
- ✓ Embeddings on GPU
- ✓ Large batch processing
- ✓ GPU vs CPU speedup measurement

### 5. CLIP Embedder
- ✓ Text embedding functionality
- ✓ Integration with MemorySystem

## How to Run

### Option 1: Simple Runner (Recommended)
```bash
python run_tests.py
```

No dependencies beyond the main HSOKV requirements.

### Option 2: With pytest
```bash
# Install pytest first
pip install pytest

# Run all tests
pytest tests/test_hsokv_comprehensive.py -v

# Run specific test class
pytest tests/test_hsokv_comprehensive.py::TestCatastrophicForgetting -v

# Run with output
pytest tests/test_hsokv_comprehensive.py -v -s
```

### Option 3: Direct Execution
```bash
cd tests
python test_hsokv_comprehensive.py
```

## GPU Testing

The test suite automatically detects if CUDA is available:

**With GPU:**
```
✓ GPU detected: NVIDIA GeForce RTX 3090
  Memory: 24.00 GB
```

**Without GPU:**
```
ℹ CPU mode (no GPU available)
⊘ Skipping GPU tests (no GPU available)
```

GPU-specific tests will be skipped on CPU-only systems.

## Expected Output

```
======================================================================
HSOKV COMPREHENSIVE TEST SUITE
======================================================================

✓ GPU detected: NVIDIA GeForce RTX 3090
  Memory: 24.00 GB

======================================================================
PART 1: CATASTROPHIC FORGETTING PREVENTION
======================================================================

=== Testing Catastrophic Forgetting Prevention ===
✓ Task A learned: Monday wake up at 7am
✓ Task B learned: Team meeting at 2pm
✓ Task A retained: Monday=7am, Tuesday=8am, Wednesday=9am
✓ Zero catastrophic forgetting confirmed!
✓ All memories retained after learning 10 more items

=== Testing Semantic Robustness ===
  Query: 'what time is my alarm?' → '6am'
  Query: 'when should I wake up?' → '6am'
  Query: 'what time do I need to get up?' → '6am'
  Query: 'alarm time?' → '6am'
✓ Semantic robustness confirmed

=== Testing Long-term Retention ===
✓ Memory retained after 100 new memories

======================================================================
PART 2: DUAL MEMORY SYSTEM (STM + LTM)
======================================================================

=== Testing Short-term Memory ===
✓ STM lookup: 2.34ms (should be <10ms)

=== Testing Memory Consolidation ===
  Access 1: combine into a single whole...
  Access 2: combine into a single whole...
  Access 3: combine into a single whole...
  STM: 1 items, LTM: 1 learning, 0 mature
✓ Memory consolidated to LTM after rehearsal

... [more tests] ...

======================================================================
FINAL RESULTS
======================================================================
✓ Catastrophic Forgetting Prevention: PASS
✓ Dual Memory System: PASS
✓ 3-Stage Lifecycle: PASS
✓ GPU Compatibility: PASS
✓ CLIP Embedder: PASS

======================================================================
TOTAL: 5/5 test suites passed
======================================================================

🎉 ALL TESTS PASSED! HSOKV is working correctly!
```

## Test Structure

```
tests/
├── README.md                          # This file
└── test_hsokv_comprehensive.py        # Main test suite
    ├── TestEnvironment                # GPU/CPU detection
    ├── TestCatastrophicForgetting     # Core innovation tests
    ├── TestDualMemorySystem           # STM + LTM tests
    ├── TestLifecycleStages            # 3-stage lifecycle
    ├── TestGPUCompatibility           # GPU acceleration
    └── TestCLIPEmbedder               # CLIP embedder tests
```

## Performance Benchmarks

The tests measure actual performance:

**Short-term Memory (STM):**
- Lookup: < 10ms (O(1) dict access)

**Long-term Memory (LTM):**
- Semantic search: 50-200ms (depends on size)
- Batch storage (100 items): ~2-5s
- Batch retrieval (100 items): ~5-10s

**GPU Acceleration:**
- Embedding speedup: 2-10x faster than CPU
- Batch processing: Even larger speedup

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors on GPU:
```python
# Reduce batch size in tests
# Or use CPU mode
config = MemoryConfig(device='cpu')
```

### Import Errors
Make sure you're in the correct directory:
```bash
cd /path/to/hsokv
python run_tests.py
```

### Model Download Issues
First run downloads models from HuggingFace:
- SentenceBERT: ~80MB
- CLIP: ~600MB

Ensure internet connection for initial download.

## Adding New Tests

To add your own tests:

```python
class TestMyFeature:
    def setup_method(self):
        self.device = TestEnvironment.get_device()
        self.embedder = SentenceBERTEmbedder(device=self.device)
        self.system = MemorySystem(self.embedder)

    def test_my_feature(self):
        print("\n=== Testing My Feature ===")

        # Your test code here
        self.system.learn("query", "answer")
        result = self.system.recall("query")

        assert result == "answer", "Test failed!"
        print("✓ My feature works!")
        return True
```

Then add to `run_all_tests()` function.

## CI/CD Integration

For continuous integration:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python run_tests.py
```

## License

Same as HSOKV (MIT License)
