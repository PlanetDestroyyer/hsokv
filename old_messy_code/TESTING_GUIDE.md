# Testing Guide: How to Validate the 3-Stage Memory Lifecycle

## 🚨 Current Problem

**Observation:**
- Loss drops to 0.00 after a few steps
- KV hit rate is only 0.01-0.02 (should be 60%+)
- Model appears to memorize everything

**Root Cause:**
The synthetic dataset is too simple:
- Only 20 rare words
- Small vocabulary (~150 words)
- Model memorizes everything in transformer weights
- **KV memory is not needed or used**

**This means:** The 3-stage lifecycle is NOT being tested properly!

---

## ✅ Solution: Test with Harder Scenarios

### **Option 1: REAL Benchmarks** (RECOMMENDED)

These test your system on actual NLP tasks with thousands of examples.

#### Test 1: GLUE SST-2 (Sentiment Analysis)
```bash
python hsokv.py --preset research --benchmark glue --glue-task sst2 \
  --allow-download --multi-gpu --visualize
```

**What this tests:**
- Few-shot learning with 16 examples per class
- Large vocabulary (10,000+ words)
- Real sentiment classification
- **Forces model to use KV memory** for rare training examples

**Expected results if 3-stage lifecycle works:**
- KV hit rate: 40-60%
- One-shot accuracy: 70%+
- Should see memories transitioning: LEARNING → REINFORCEMENT → MATURE

#### Test 2: GLUE MNLI (Natural Language Inference)
```bash
python hsokv.py --preset research --benchmark glue --glue-task mnli \
  --allow-download --multi-gpu --visualize
```

**What this tests:**
- Harder task (3-way classification: entailment/neutral/contradiction)
- Longer sequences
- More complex reasoning

#### Test 3: CIFAR-10 (Continual Learning)
```bash
python hsokv.py --preset research --benchmark cifar \
  --allow-download --multi-gpu --visualize
```

**What this tests:**
- **Continual learning** across different classes
- Memory retention over time
- Tests if MATURE memories persist while learning new classes
- Tests if consolidation prevents catastrophic forgetting

---

### **Option 2: Language Modeling Task**

Use the language modeling task instead of classification:

```bash
# Generate corpus and train
python hsokv.py --task language_model --preset research \
  --corpus-size large --multi-gpu --visualize
```

**What this tests:**
- Next-token prediction (harder than classification)
- Larger vocabulary
- More realistic use case

---

### **Option 3: Modify Synthetic Dataset**

If you want to stick with synthetic data, make it much harder:

#### Step 1: Expand the rare words list
Edit `hsokv_core/data.py` line 184 and add 100+ more rare words.

Or use the generated `harder_dataset.json`:

```python
# In hsokv_core/data.py, replace RARE_WORD_SPECS with:
import json
with open("harder_dataset.json") as f:
    data = json.load(f)
RARE_WORD_SPECS = data["rare_words"][:100]  # Use 100 words
```

#### Step 2: Reduce training examples per word
Edit line 252 in `hsokv_core/data.py`:
```python
# OLD:
train_count = random.randint(1, 5)

# NEW (harder):
train_count = random.randint(1, 2)  # Only 1-2 examples per word!
```

#### Step 3: Increase distractors
Edit line 292 in `hsokv_core/data.py`:
```python
# OLD:
for _ in range(len(RARE_WORD_SPECS) * CONFIG["retention_distractor_factor"]):

# NEW (harder):
for _ in range(len(RARE_WORD_SPECS) * 10):  # 10× more distractors
```

Then run:
```bash
python hsokv.py --preset research --multi-gpu --visualize
```

---

## 📊 What to Look For in Results

### **Signs the 3-Stage Lifecycle is Working:**

1. **KV Hit Rate: 40-60%+**
   - Current: 0.01-0.02 (model not using KV memory)
   - Target: 40-60% (model actively using KV memory)

2. **Loss Curve:**
   - Should NOT drop to 0.00 immediately
   - Should show gradual learning over 5-10 iterations
   - If loss = 0.00 after 2 steps → Task too easy!

3. **One-Shot Accuracy: 60-75%**
   - Tests if model learns from 1 example
   - This is where 3-stage lifecycle helps

4. **Retention: 95-98%**
   - Tests if MATURE memories persist
   - Tests if consolidation works

5. **Memory Stage Transitions:**
   - Check logs for messages about:
     - "LEARNING stage" (first 5 uses)
     - "REINFORCEMENT stage" (uses 6-20)
     - "MATURE stage" (20+ uses)
     - Consolidation events
     - Forgetting events

### **Signs it's NOT Working (current state):**

❌ Loss = 0.00 after few steps
❌ KV hit rate < 5%
❌ Model memorizing in weights, not using KV
❌ No memory stage transitions in logs

---

## 🎯 Recommended Testing Sequence

**Day 1: Quick Validation (30 mins)**
```bash
# Test 1: GLUE SST-2 (fastest real benchmark)
python hsokv.py --preset demo --benchmark glue --glue-task sst2 \
  --allow-download --visualize

# Look for: KV hit rate 40%+, one-shot accuracy 60%+
```

**Day 2: Full Evaluation (2 hours)**
```bash
# Test 2: Research preset on GLUE
python hsokv.py --preset research --benchmark glue --glue-task sst2 \
  --allow-download --multi-gpu --visualize

# Test 3: MNLI (harder task)
python hsokv.py --preset research --benchmark glue --glue-task mnli \
  --allow-download --multi-gpu --visualize

# Test 4: CIFAR-10 (continual learning)
python hsokv.py --preset research --benchmark cifar \
  --allow-download --multi-gpu --visualize
```

**Day 3: Ablation Study (3 hours)**
```bash
# Test 5: Run full ablation suite
python hsokv.py --preset research --run-ablations \
  --benchmark glue --glue-task sst2 --allow-download
```

---

## 🔬 Debug: Check if KV Memory is Being Used

Add this debugging code to see what's happening:

```python
# In hsokv_core/training.py, after line 100 (in training loop):

print(f"\n[DEBUG] Iteration {iteration + 1}:")
print(f"  KV memory size: {len(model.kv_memory.memory)}")
print(f"  Avg gate value: {info['gate_values'].mean():.3f}")
print(f"  KV hit rate: {kv_hit_rate:.3f}")

# Print first 3 memories
for i, (key, meta) in enumerate(list(model.kv_memory.memory.items())[:3]):
    stage = model.kv_memory.get_memory_stage(key, meta)
    print(f"  Memory {i}: stage={stage}, confidence={meta['confidence']:.2f}, uses={meta['retrieval_count']}")
```

**What you should see:**
- Memory size growing over iterations
- Gate values around 0.3-0.7 (model using KV memory)
- Memories transitioning through stages

**What you're probably seeing now:**
- Memory size = 0 or very small
- Gate values ≈ 0.0 (model ignoring KV memory)
- No stage transitions

---

## 💡 Quick Fix: Increase Rare Words Now

**Fastest way to make synthetic dataset harder:**

```bash
# 1. Backup current data.py
cp hsokv_core/data.py hsokv_core/data.py.backup

# 2. Count current rare words
grep -c '"word":' hsokv_core/data.py  # Shows 20

# 3. You need to manually add 80-180 more rare words to line 184-205
# Or use the harder_dataset.json I created

# 4. Reduce training examples
# Edit line 252: change random.randint(1, 5) to random.randint(1, 2)

# 5. Test
python hsokv.py --preset research --multi-gpu --visualize
```

---

## 📈 Success Metrics

Your 3-stage lifecycle works if you see:

| Metric | Current (Broken) | Target (Working) |
|--------|------------------|------------------|
| Loss after 5 steps | 0.00 | 0.5-1.5 |
| KV hit rate | 0.01-0.02 | 40-60% |
| One-shot accuracy | ??? | 60-75% |
| Retention | ??? | 95-98% |
| Memory size | 0-20 | 100-400 |
| Gate avg | ≈0.0 | 0.3-0.7 |

If you hit these targets on **GLUE SST-2** or **CIFAR-10**, your idea is validated! ✅

---

## 🚀 Next Steps After Validation

1. **Run experiments/paper_experiments.py** - Generate paper-ready results
2. **Run with multiple seeds** - Check reproducibility
3. **Compare to baselines** - Show 3-stage lifecycle beats standard approaches
4. **Write up results** - Document the "overwhelming" word example
5. **Submit paper** 🎉
