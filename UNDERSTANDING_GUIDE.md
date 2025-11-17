# Understanding HSOKV - Simple Guide

## Project Structure (Clean!)

```
hsokv/
├── README.md                     # Documentation
├── setup.py                      # Installation
├── requirements.txt              # Dependencies
│
├── examples/                     # How to use it
│   ├── alarm_assistant.py        # Simple example
│   └── human_memory_demo.py      # Dual memory example
│
└── hsokv/                        # Core code (7 files)
    ├── __init__.py               # Public API
    ├── config.py                 # Configuration
    ├── dual_memory.py            # Short-term + Long-term
    ├── embedders.py              # Frozen embedders
    ├── lifecycle.py              # 3-stage lifecycle
    ├── memory.py                 # KeyValueMemory
    └── memory_system.py          # Main system
```

## Core Concepts (Read in Order)

### 1. Start Here: `config.py` (60 lines)
**What:** Settings for the memory system
**Key settings:**
- `learning_phase_duration = 5` (LEARNING stage threshold)
- `reinforcement_phase_duration = 20` (MATURE stage threshold)
- `max_entries = 1000` (memory capacity)

### 2. Then: `lifecycle.py` (120 lines)
**What:** 3-stage memory lifecycle
**Stages:**
- LEARNING (0-5 uses): Pure recall, max protection
- REINFORCEMENT (5-20 uses): Gradual blending, high protection
- MATURE (20+ uses): Standard retrieval, can be pruned

### 3. Then: `embedders.py` (120 lines)
**What:** Frozen embedders (no training!)
**Types:**
- `SentenceBERTEmbedder`: For text
- `CLIPEmbedder`: For images + text

### 4. Core: `memory.py` (330 lines)
**What:** Vector-based memory storage
**Key methods:**
- `store()`: Save key-value pair
- `retrieve()`: Find similar memories
- `prune()`: Remove low-confidence memories

### 5. Simple API: `memory_system.py` (180 lines)
**What:** Easy-to-use wrapper
**Methods:**
- `learn(query, answer)`: Store fact
- `recall(query)`: Retrieve fact

### 6. Advanced: `dual_memory.py` (460 lines)
**What:** Human-like dual memory
**Components:**
- `ShortTermMemory`: Dict, 7±2 capacity, 30s decay
- `LongTermMemory`: Vector DB, unlimited, permanent
- `DualMemorySystem`: Combined system

## How It Works

### Simple Memory System
```python
from hsokv import MemorySystem, SentenceBERTEmbedder

# Setup
embedder = SentenceBERTEmbedder()
system = MemorySystem(embedder)

# Learn
system.learn("what is AI?", "Artificial Intelligence")

# Recall
answer = system.recall("what is AI?")  # → "Artificial Intelligence"
```

### Dual Memory System (Human-like)
```python
from hsokv import DualMemorySystem, SentenceBERTEmbedder

# Setup
embedder = SentenceBERTEmbedder()
system = DualMemorySystem(embedder)

# Learn (→ short-term)
system.learn("ephemeral", "temporary")

# Recall from short-term (O(1) - fast!)
system.recall("ephemeral")

# Recall 3+ times → consolidates to long-term
system.recall("ephemeral")
system.recall("ephemeral")
system.recall("ephemeral")  # Now in long-term!

# Semantic recall works
system.recall("temporary")  # Finds "ephemeral"!
```

## Key Innovations

### 1. Frozen Embeddings
- Embeddings NEVER change
- Prevents catastrophic forgetting
- Monday embedding = Wednesday embedding

### 2. 3-Stage Lifecycle
- LEARNING: New memory (max protection)
- REINFORCEMENT: Practicing (boosted confidence)
- MATURE: Established (can be pruned)

### 3. Dual Memory
- Short-term: Fast dict (7±2 items)
- Long-term: Vector DB (unlimited)
- Consolidation: Rehearsal → long-term

## Read the Code

### Start with examples:
1. `examples/alarm_assistant.py` - Simple usage
2. `examples/human_memory_demo.py` - Dual memory

### Then read core modules:
1. `hsokv/config.py` - Settings
2. `hsokv/lifecycle.py` - 3 stages
3. `hsokv/embedders.py` - Frozen embedders
4. `hsokv/memory.py` - Core storage
5. `hsokv/memory_system.py` - Simple API
6. `hsokv/dual_memory.py` - Advanced API

## Install and Test

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run examples
python examples/alarm_assistant.py
python examples/human_memory_demo.py
```

## Architecture Overview

```
User Code
    ↓
MemorySystem or DualMemorySystem (API layer)
    ↓
KeyValueMemory (storage layer)
    ↓
MemoryLifecycle (3-stage logic)
    ↓
FrozenEmbedder (encoding layer)
```

## That's It!

The code is clean and focused. Each file has one clear purpose.

Start with the examples, then read the code files in order above.
