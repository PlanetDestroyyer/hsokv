# H-SOKV Complete Execution Walkthrough

## Table of Contents
1. [Overview](#overview)
2. [Main Execution Flow](#main-execution-flow)
3. [Training Process](#training-process)
4. [Memory Operations](#memory-operations)
5. [Baseline Training](#baseline-training)
6. [Key Calculations](#key-calculations)
7. [Potential Issues](#potential-issues)

---

## Overview

**Entry Point:** `hsokv.py` → `main()` → `run_experiment()`

**Key Files:**
- `hsokv.py`: Main entry point, orchestration
- `hsokv_core/training.py`: Training loops
- `hsokv_core/model.py`: Transformer + KV memory
- `hsokv_core/memory.py`: Key-value episodic memory
- `hsokv_core/swarm.py`: Swarm optimizer
- `hsokv_core/config.py`: Configuration

---

## Main Execution Flow

### Step 1: Initialization (hsokv.py:907-913)
```python
def main():
    print_system_info()        # Print GPU/device info
    args = parse_args()        # Parse command-line args
    run_experiment(args)       # Main logic
```

**What happens:**
1. Prints PyTorch version, CUDA status, device
2. Parses arguments like `--preset research`
3. Calls `run_experiment()`

---

### Step 2: Configuration Setup (hsokv.py:337-382)
```python
def run_experiment(args):
    # 1. Load preset config
    preset_name = args.preset or "demo"  # "research" in your case
    base_config = override_config(CONFIG, {"preset": preset_name})
    base_config = override_config(base_config, PRESET_CONFIGS[preset_name])

    # 2. Apply overrides from args
    # 3. Set random seed
    set_seed(config["seed"])  # seed=42
```

**What happens:**
1. Loads base CONFIG from `config.py`
2. Applies `research` preset:
   ```python
   "meta_iterations": 10,     # 10 swarm iterations
   "agents_per_manager": 5,   # 5 agents per manager
   "agent_steps": 50,         # 50 training steps per agent
   "batch_size": 32,
   "num_managers": 2,         # 2 managers
   ```
3. Sets seed for reproducibility

---

### Step 3: Dataset Generation (hsokv.py:398-428)
```python
# For classification task (default):
dataset, tokenizer, word_counts = generate_dataset(
    tokenizer=None,
    fit_tokenizer=True,
)
```

**What happens (in `data.py`):**
1. **Loads 100 rare words** from `RARE_WORD_SPECS`
2. **Creates samples:**
   - Train: Each word appears 1-2 times (one-shot learning)
   - Test: Word + definition + 5 distractors
3. **Tokenizes** using SimpleTokenizer:
   - Splits into words/punctuation
   - Creates vocab (word → ID mapping)
4. **Returns:**
   - `dataset`: Train/test/retention splits
   - `tokenizer`: Fitted tokenizer
   - `word_counts`: {word: count} dict

---

### Step 4: Main Training - train_hsokv() (training.py:105-260)

This is the **core function**. Let me break it down:

#### 4.1 Model Initialization (training.py:130-149)
```python
# Create probe model to estimate FLOPs
probe_model = TransformerWithKV(vocab_size, num_labels, tokenizer, config).to(device)
flops_per_step = estimate_model_flops(probe_model, config)
del probe_model

# Create model factory
def model_factory():
    model = TransformerWithKV(vocab_size, num_labels, tokenizer, config)
    model.to(device)
    return model
```

**What happens:**
1. Creates temporary model to estimate computation cost (FLOPs)
2. Deletes probe model
3. Defines factory function to create fresh models for swarm

---

#### 4.2 Dataloader Preparation (training.py:152-153)
```python
dataloaders = prepare_dataloaders(dataset, tokenizer, config)
```

**Creates 3 dataloaders:**
- `train_loader`: Training data (batch_size=32)
- `test_loader`: Test data
- `retention_loader`: Retention test with distractors

---

#### 4.3 Swarm Training Loop (training.py:155-223)

This is **where the magic happens**:

```python
if config.get("use_swarm", True):
    # Initialize swarm supervisor
    supervisor = Supervisor(
        agents_per_manager=config["agents_per_manager"],  # 5
        num_managers=config["num_managers"],              # 2
        config=config,
    )

    # Meta-iteration loop (10 iterations)
    for iteration in tqdm(range(config["meta_iterations"]), desc="Meta-iterations"):
        # 1. Run swarm iteration
        reward, best_agent_config = supervisor.run_iteration(
            model_factory=model_factory,
            dataloaders=dataloaders,
            word_counts=word_counts,
            agent_steps=config["agent_steps"],  # 50 steps
        )

        # 2. Evaluate after iteration
        model = supervisor.get_best_model()
        test_metrics = evaluate_model(model, test_loader, device, ...)
        retention = evaluate_retention(model, retention_loader, device)

        # 3. Run consolidation (if enabled)
        if config.get("use_consolidation") and iteration % consolidation_interval == 0:
            consolidation_module.consolidate(model, train_loader, device)

        # 4. Run forgetting (if enabled)
        if config.get("use_forgetting") and iteration % forgetting_interval == 0:
            forgetting_module.prune_memory(model.kv_memory)
```

**Detailed breakdown:**

##### 4.3.1 Swarm Iteration (swarm.py)
```python
supervisor.run_iteration():
    # For each manager (2 managers):
    for manager in managers:
        # For each agent (5 agents per manager):
        for agent in manager.agents:
            # 1. Create fresh model
            model = model_factory()

            # 2. Set agent's hyperparameters
            optimizer = agent.optimizer_type(  # SGD, Adam, RMSprop, or random
                model.parameters(),
                lr=agent.learning_rate  # Random from 1e-5 to 1e-3
            )
            top_k = agent.kv_top_k  # Random from 1-10

            # 3. Train for 50 steps
            for step in range(50):
                batch = next(train_loader)

                # Forward pass
                logits, info = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    top_k=top_k
                )

                # Compute loss
                loss = criterion(logits, batch["labels"])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update KV memory confidence
                for idx, success in zip(retrieved_indices, correctness):
                    model.kv_memory.update_confidence(idx, success)

            # 4. Evaluate agent
            agent_reward = evaluate_model(model, test_loader, ...)
            agent.update_reward(agent_reward)

        # Manager aggregates agent rewards
        manager.aggregate_rewards()

    # Supervisor selects best configuration
    best_config = supervisor.select_best()
    return reward, best_config
```

---

##### 4.3.2 Forward Pass (model.py:TransformerWithKV.forward)

**This is KEY - let's trace it line by line:**

```python
def forward(self, input_ids, attention_mask, top_k=5):
    # Step 1: Embed tokens
    x = self.embedding(input_ids) * math.sqrt(self.d_model)  # Scale embeddings
    x = self.pos_encoder(x)  # Add positional encoding
    x = self.dropout(x)

    # Step 2: Transformer layers (4 layers)
    x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)

    # Step 3: Pool sequence → single vector
    pooled = x[torch.arange(x.size(0)), (attention_mask.sum(dim=1) - 1).clamp(min=0)]

    # Step 4: Query KV memory
    if self.use_kv:
        # Retrieve top-k similar memories
        kv_output, kv_details = self.kv_memory.retrieve(
            query=pooled,
            top_k=top_k,
            hidden_states=x  # For context-aware retrieval
        )
    else:
        kv_output = torch.zeros_like(pooled)
        kv_details = {...}

    # Step 5: Gated fusion
    gate_logits = self.gate_network(pooled)  # Learn gate weights
    gate = torch.sigmoid(gate_logits)

    fused = gate * kv_output + (1 - gate) * pooled  # Blend memory + internal

    # Step 6: Classification
    logits = self.classifier(fused)

    return logits, {
        "gate_values": gate,
        "kv_details": kv_details,
        ...
    }
```

**Key operations:**
1. **Embedding**: `vocab_id → 256-dim vector`
2. **Transformer**: 4-layer attention → contextualized representations
3. **Pooling**: Sequence → single vector (take last token)
4. **KV Retrieval**: Find top-k similar memories
5. **Gating**: Learn to blend memory vs internal knowledge
6. **Classification**: Fused vector → class logits

---

##### 4.3.3 KV Memory Retrieval (memory.py:KeyValueMemory.retrieve)

**Critical function:**

```python
def retrieve(self, query, top_k=5, hidden_states=None):
    if len(self.keys) == 0:
        # No memories yet
        return zero_output, default_details

    # Step 1: Normalize query (L2 normalization)
    query_norm = F.normalize(query, p=2, dim=-1, eps=1e-12)

    # Step 2: Normalize all keys (L2 normalization)
    keys_tensor = torch.stack(self.keys)
    keys_norm = F.normalize(keys_tensor, p=2, dim=-1, eps=1e-12)

    # Step 3: Compute cosine similarity
    similarities = torch.matmul(query_norm, keys_norm.T)  # [batch, num_memories]

    # Step 4: Apply context boosts (if enabled)
    if self.use_context_retrieval and hidden_states is not None:
        domain_boost = extract_domain_signal(hidden_states)  # 1.5x for matching domain
        recency_boost = 0.95^age  # Decay by age
        emotion_boost = compute_emotion_similarity(...)  # ±0.3

        similarities = similarities * domain_boost * recency_boost + emotion_boost

    # Step 5: Apply confidence weighting
    confidences = torch.tensor([self.metadata[i]["confidence"] for i in range(len(self.keys))])
    weighted_similarities = similarities * confidences

    # Step 6: Select top-k
    topk_values, topk_indices = torch.topk(weighted_similarities, k=min(top_k, len(self.keys)))

    # Step 7: Retrieve value vectors
    retrieved_values = [self.value_vectors[idx] for idx in topk_indices[0]]
    value_output = torch.mean(torch.stack(retrieved_values), dim=0)  # Average

    # Step 8: Update metadata
    for idx in topk_indices[0]:
        self.metadata[idx]["retrieval_count"] += 1
        self.metadata[idx]["last_retrieved"] = current_time

    return value_output, {
        "topk_indices": topk_indices,
        "avg_similarity": topk_values.mean(),
        ...
    }
```

**Key calculations:**
- **Cosine similarity**: `dot(query_norm, key_norm)` → value in [-1, 1]
- **Confidence weighting**: Boost memories that have been correct in the past
- **Top-k selection**: Take k most similar
- **Averaging**: Mean of top-k value vectors

---

##### 4.3.4 Memory Writing (surprise_writing.py)

**When does a sample get written to memory?**

```python
def should_write_to_memory(loss, embedding, existing_memories):
    # Step 1: Compute surprise (prediction uncertainty)
    surprise = loss.item()  # Cross-entropy loss

    # Step 2: Check if this is first exposure to a word
    word_id = get_word_id(sample)
    is_first_exposure = (word_counts[word] == 1)

    # Step 3: Compute novelty (distance to existing memories)
    if len(existing_memories) > 0:
        similarities = [cosine_sim(embedding, mem) for mem in existing_memories]
        max_similarity = max(similarities)
        novelty = 1 - max_similarity
    else:
        novelty = 1.0

    # Step 4: Decision rules
    if is_first_exposure:
        threshold = config["first_exposure_threshold"]  # 0.15 (lower = easier to write)
        if surprise > threshold or novelty > 0.7:
            # Write with confidence boost
            confidence = 0.5 + config["first_exposure_boost"]  # 0.5 + 0.25 = 0.75
            write_to_memory(embedding, confidence)
    else:
        threshold = config["surprise_threshold"]  # 0.3
        if surprise > threshold or novelty > 0.7:
            confidence = 0.5
            write_to_memory(embedding, confidence)
```

**Key logic:**
- **First-exposure words**: Lower threshold (0.15), confidence boost (0.75)
- **Seen words**: Higher threshold (0.3), normal confidence (0.5)
- **Surprise**: High loss → uncertain → write to memory
- **Novelty**: Different from existing → write to memory

---

##### 4.3.5 Consolidation (consolidation.py)

**After iteration X, consolidate high-confidence memories:**

```python
def consolidate(model, train_loader, device):
    # Step 1: Find high-confidence memories
    high_conf_indices = [
        i for i, meta in enumerate(model.kv_memory.metadata)
        if meta["confidence"] > config["consolidation_threshold"]  # 0.85
        and meta["success_rate"] > 0.8
    ]

    if len(high_conf_indices) == 0:
        return

    # Step 2: Generate synthetic training data from memories
    synthetic_data = []
    for idx in high_conf_indices:
        word = model.kv_memory.values[idx]["word"]
        definition = model.kv_memory.values[idx]["definition"]
        synthetic_data.append(create_sample(word, definition))

    # Step 3: Fine-tune model on synthetic data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(3):
        for sample in synthetic_data:
            logits, _ = model(sample["input_ids"], sample["attention_mask"])
            loss = criterion(logits, sample["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Step 4: Validate consolidation
    validation_accuracy = test_on_synthetic_data(model, synthetic_data)
    if validation_accuracy > 0.9:
        # Success! Delete consolidated memories
        for idx in sorted(high_conf_indices, reverse=True):
            del model.kv_memory.keys[idx]
            del model.kv_memory.values[idx]
            del model.kv_memory.metadata[idx]
```

**Key idea:**
- Absorb high-confidence memories into model weights
- Free up memory slots
- Only delete if validation passes (prevent forgetting)

---

##### 4.3.6 Forgetting (forgetting.py)

**Every N iterations, prune low-utility memories:**

```python
def prune_memory(kv_memory):
    # Step 1: Compute utility score for each memory
    utilities = []
    for i, meta in enumerate(kv_memory.metadata):
        # Weighted combination of factors
        confidence_score = meta["confidence"]  # 0-1
        usage_score = min(meta["retrieval_count"] / 100, 1.0)  # Normalize
        success_score = meta["success_rate"]  # 0-1
        recency_score = compute_recency(meta["created_at"])  # Decay over time

        utility = (
            0.4 * confidence_score +
            0.2 * usage_score +
            0.3 * success_score +
            0.1 * recency_score
        )
        utilities.append((i, utility))

    # Step 2: Sort by utility
    sorted_by_utility = sorted(utilities, key=lambda x: x[1])

    # Step 3: Prune bottom X%
    to_delete = [
        i for i, util in sorted_by_utility
        if util < config["forgetting_utility_threshold"]  # 0.10
    ]

    # Step 4: Delete
    for idx in sorted(to_delete, reverse=True):
        del kv_memory.keys[idx]
        del kv_memory.values[idx]
        del kv_memory.metadata[idx]
```

**Key calculation:**
- **Utility = weighted combination of:**
  - 40% confidence (how often correct when retrieved)
  - 20% usage (how often retrieved)
  - 30% success rate (accuracy when used)
  - 10% recency (newer = better)
- **Delete if utility < 0.10**

---

### Step 5: Baseline Training (training.py:262-450)

After H-SOKV trains, **3 baselines** are trained for comparison:

#### 5.1 Baseline-1: Standard Fine-tuning
```python
def train_baseline_standard(dataset, tokenizer, config, num_labels):
    # Create model WITHOUT KV memory
    model = BaselineTransformer(vocab_size, num_labels, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])

    # Train for N epochs (budget-matched to H-SOKV)
    for epoch in range(config["baseline_epochs"]):
        for batch in train_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    test_metrics = evaluate_model(model, test_loader, ...)
    return test_metrics
```

**No KV memory, no swarm - just standard training**

---

#### 5.2 Baseline-2: KV-only (No model updates)
```python
def train_baseline_kv(dataset, tokenizer, config, num_labels):
    # Create model WITH KV memory
    model = TransformerWithKV(vocab_size, num_labels, tokenizer, config)

    # FREEZE model parameters (only update memory)
    for param in model.parameters():
        param.requires_grad = False

    # "Train" by only writing to memory
    model.eval()
    for batch in train_loader:
        logits, info = model(batch["input_ids"], batch["attention_mask"])

        # Write to memory if surprise/novelty high
        if should_write(loss, embedding):
            model.kv_memory.write(embedding, value, confidence)

    # Evaluate
    test_metrics = evaluate_model(model, test_loader, ...)
    return test_metrics
```

**Memory-only learning, no weight updates**

---

#### 5.3 Baseline-3: In-Context Learning (Nearest Neighbor)

**🚨 THIS IS THE CRITICAL FINDING!**

Baseline-3 achieves **86% one-shot accuracy** (better than H-SOKV's 60%) using a **MUCH SIMPLER** approach:

```python
def in_context_learning(dataset, tokenizer, config):
    # Step 1: Create simple embedding layer (64 dims, NO training!)
    embedding_model = nn.Embedding(vocab_size, 64).to(device)

    # Step 2: Embed all TRAINING samples (just once)
    train_embeddings = []
    labels = []
    for sample in dataset["train"]:
        ids = tokenizer.encode(sample["story"])
        with torch.no_grad():  # NO GRADIENTS
            emb = embedding_model(ids)
            # Pool by averaging (with masking)
            pooled = (emb * mask).sum(dim=0) / mask.sum()
        train_embeddings.append(pooled)
        labels.append(sample["word_id"])

    # Step 3: At test time, find nearest neighbor
    for test_sample in dataset["test"]:
        test_emb = embed_and_pool(test_sample)

        # Compute similarity to ALL training samples
        similarities = test_emb @ train_embeddings.T  # Dot product

        # Predict: label of most similar training sample
        nearest_idx = similarities.argmax()
        prediction = labels[nearest_idx]

    return accuracy
```

**What this does:**
1. **NO transformer** (just raw embeddings!)
2. **NO training** (embeddings are random/untrained!)
3. **NO KV memory**
4. **NO swarm optimizer**
5. Just **nearest-neighbor search** on raw embeddings

**Results:**
- One-shot accuracy: **86%** (vs H-SOKV's 60%!)
- Retention: **87%** (vs H-SOKV's 93% - slightly worse)
- KV hit rate: **15%** (vs H-SOKV's 9%)

---

### 🔴 Why is Baseline-3 Better?

**The simple nearest-neighbor approach works better because:**

1. **No overfitting**: Untrained embeddings are generic
2. **Perfect memory**: Stores ALL training samples (not selective)
3. **No forgetting**: Never deletes memories
4. **No consolidation bugs**: No complex memory lifecycle
5. **Direct similarity**: Cosine similarity on raw text is effective

**This reveals a FUNDAMENTAL ISSUE with H-SOKV:**

The complex machinery (swarm, consolidation, forgetting, surprise thresholds) is **HURTING performance** rather than helping!

---

## Key Calculations Summary

### 1. **One-Shot Accuracy**
```python
one_shot_accuracy = correct_on_first_exposure_words / total_first_exposure_words
```
- Only counts words that appeared 1 time in training
- Measures true one-shot learning

### 2. **KV Hit Rate**
```python
kv_hit_rate = mean(cosine_similarities_of_retrieved_memories)
```
- Average similarity score of top-k retrieved memories
- Higher = better memory retrieval

### 3. **Retention**
```python
retention = correct_on_distractor_test / total_distractor_samples
```
- Tests memory of previously seen words with distractors
- Measures catastrophic forgetting

### 4. **Convergence Steps**
```python
convergence = first_step_where_accuracy >= 0.8
```
- How quickly the model learns
- Faster = better

---

## Potential Issues Found

### 🔴 Issue 1: Memory is Being Deleted Too Aggressively

**Evidence:**
- KV hit rate: Only 9% (very low!)
- One-shot accuracy: 60% (should be 85-95%)
- Baseline-3 (simple nearest-neighbor): 86% with 15% hit rate

**Root Cause:**
Consolidation and/or forgetting are deleting memories that are still needed.

**Specific Problems:**

#### 1.1 Consolidation Threshold Too Strict
```python
"consolidation_threshold": 0.85  # Requires 85% accuracy before consolidating
```

Problem: Few memories reach 85% confidence early in training. But consolidation still runs and might delete anyway.

#### 1.2 Forgetting Deletes First-Exposure Memories
```python
"forgetting_utility_threshold": 0.10  # Delete if utility < 0.10
```

Problem: First-exposure memories (one-shot words) might have:
- Low retrieval_count (just seen once!)
- Low confidence initially
- → Gets deleted before it can be used!

#### 1.3 First-Exposure Protection Window Too Short
```python
"first_exposure_boost": 0.25,           # Only +0.25 confidence
"first_exposure_retrieval_window": 20, # Protected for 20 retrievals
```

Problem: If a memory is only retrieved 5-10 times during training, the boost decays before consolidation.

---

### 🔴 Issue 2: Surprise Threshold Still Too High

**Current setting:**
```python
"surprise_threshold": 0.3,              # Write if loss > 0.3
"first_exposure_threshold": 0.15,      # Write if loss > 0.15 (for novel words)
```

**Problem:**
- Cross-entropy loss can be low even when prediction is wrong!
- Example: If model predicts class A with 60% confidence but correct answer is B:
  - Loss = -log(0.4) = 0.92 (would write)
- But if model is 50/50 uncertain:
  - Loss = -log(0.5) = 0.69 (would write)
- But if model already learned and gets it right with 70% confidence:
  - Loss = -log(0.7) = 0.36 (would STILL write!)

**This causes:**
- Writing too many "already learned" memories
- Wasting memory slots on duplicates
- Not enough room for truly novel words

---

### 🔴 Issue 3: KV Retrieval Is Ineffective

**Evidence:**
- KV hit rate: 9% (extremely low!)
- Baseline-3 (no KV): 86% accuracy
- H-SOKV (with KV): 60% accuracy

**Possible Causes:**

#### 3.1 L2 Normalization Hurts Similarity
```python
# In memory.py:retrieve()
query_norm = F.normalize(query, p=2, dim=-1, eps=1e-12)
keys_norm = F.normalize(keys_tensor, p=2, dim=-1, eps=1e-12)
```

Problem: L2 normalization forces all vectors to unit length, which:
- Loses magnitude information
- Can make dissimilar vectors appear similar
- Baseline-3 uses RAW dot product (no normalization) and works better!

#### 3.2 Context Boosts Are Noisy
```python
domain_boost = extract_domain_signal(hidden_states)  # 1.5x boost
recency_boost = 0.95^age
emotion_boost = compute_emotion_similarity(...)
```

Problem: These signals are extracted from a 4-layer transformer trained for only 50 steps:
- Domain signal is unreliable (model hasn't learned domains yet)
- Emotion signal is random noise
- Boosts can HURT similarity rather than help

#### 3.3 Top-K Averaging Dilutes Signal
```python
# Retrieve top-5 memories and average them
retrieved_values = [value_vectors[idx] for idx in topk_indices]
value_output = torch.mean(torch.stack(retrieved_values), dim=0)
```

Problem:
- Top-1 memory might be perfect match
- Top-5 includes 4 noisy matches
- Averaging dilutes the good signal with noise!

---

### 🔴 Issue 4: Swarm Optimizer Adds Noise

**Current setup:**
- 2 managers × 5 agents = 10 agents
- Each tries different:
  - Optimizer (SGD, Adam, RMSprop, random)
  - Learning rate (1e-5 to 1e-3)
  - Top-k (1 to 10)

**Problem:**
- Only 50 training steps per agent (very short!)
- Most agents waste steps on bad hyperparameters
- Best agent's model is used, but its memory is contaminated by bad retrievals
- Swarm diversity metric causes supervisor to AVOID converging on best config

**Evidence:**
- Convergence = 1 step (model already knows answer from swarm exploration!)
- But one-shot accuracy still only 60% (swarm found wrong solution)

---

### 🔴 Issue 5: Gate Network Learns to Ignore Memory

```python
gate = torch.sigmoid(gate_network(pooled))  # Learn when to use KV
fused = gate * kv_output + (1 - gate) * pooled
```

**Problem:**
If KV retrieval is noisy (Issues #3.1-3.3), the gate network learns to:
- Set gate ≈ 0 (ignore memory)
- Set gate ≈ 1 only for internal representation

**Evidence:**
- KV hit rate: 9% (memory barely used!)
- Model relies on internal representation instead

---

## Recommended Fixes

### Fix #1: Disable Consolidation and Forgetting COMPLETELY
```python
"use_consolidation": False,
"use_forgetting": False,
```

**Why:** These are DELETING the very memories we need for one-shot learning!

**Expected improvement:** One-shot accuracy 60% → 70-80%

---

### Fix #2: Lower Surprise Threshold Further
```python
"surprise_threshold": 0.1,           # Was 0.3
"first_exposure_threshold": 0.05,   # Was 0.15
```

**Why:** Write more aggressively, especially for novel words.

**Expected improvement:** KV hit rate 9% → 20%+

---

### Fix #3: Remove L2 Normalization from Retrieval
```python
# Change this:
query_norm = F.normalize(query, p=2, dim=-1, eps=1e-12)
similarities = torch.matmul(query_norm, keys_norm.T)

# To this:
similarities = torch.matmul(query, keys_tensor.T)  # Raw dot product
```

**Why:** Baseline-3 uses raw similarity and works better!

**Expected improvement:** KV hit rate 9% → 30%+

---

### Fix #4: Use Top-1 Instead of Top-K Averaging
```python
# Change this:
value_output = torch.mean(torch.stack(retrieved_values), dim=0)

# To this:
value_output = retrieved_values[0]  # Just use best match!
```

**Why:** Averaging dilutes the signal.

**Expected improvement:** Sharper memory retrieval

---

### Fix #5: Disable Swarm (Use Fixed Hyperparameters)
```python
"use_swarm": False,
"agent_steps": 500,  # More steps since no swarm exploration
```

**Why:** Swarm adds noise with only 50 steps per agent.

**Expected improvement:** Cleaner training, less noise

---

### Fix #6: Disable Context Boosts
```python
"use_context_retrieval": False,
```

**Why:** Domain/emotion signals are noisy from undertrained model.

**Expected improvement:** Less noisy retrieval

---

## Quick Test: Replicate Baseline-3's Success

**Try this minimal configuration:**

```python
# Minimal H-SOKV that mimics Baseline-3:
CONFIG = {
    "use_swarm": False,
    "use_consolidation": False,
    "use_forgetting": False,
    "use_context_retrieval": False,
    "surprise_threshold": 0.0,  # Write EVERYTHING
    "top_k": 1,  # Only use best match
    "meta_iterations": 1,
    "agent_steps": 100,
}
```

**Expected result:** Should approach Baseline-3's 86% one-shot accuracy!

If this works, we know the complex features are the problem.

