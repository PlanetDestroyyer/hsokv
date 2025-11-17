# Clean Rebuild Plan - HSOKV Memory System

## Current Problem
- 19+ Python files scattered in root
- 13+ markdown documentation files
- Messy patches on top of patches
- Mix of broken training code + working memory code
- Unclear what the actual system is

## New Clean Structure

```
hsokv/
├── setup.py                          # Professional package setup
├── pyproject.toml                    # Modern Python packaging
├── README.md                         # Clear, focused documentation
├── requirements.txt                  # Minimal dependencies
├── LICENSE                           # MIT License
│
├── hsokv/                            # Main package
│   ├── __init__.py                   # Public API
│   ├── memory_system.py              # Main MemorySystem class
│   ├── memory.py                     # KeyValueMemory (cleaned up)
│   ├── embedders.py                  # Frozen embedders (CLIP, Sentence-BERT)
│   ├── lifecycle.py                  # 3-stage lifecycle logic
│   └── config.py                     # Configuration
│
├── examples/                         # Clean examples
│   ├── alarm_assistant.py            # Your alarm example
│   ├── cifar_continual.py            # CIFAR-10 continual learning
│   └── text_memory.py                # Text-based memory demo
│
├── tests/                            # Proper tests
│   ├── test_memory.py                # Memory tests
│   ├── test_lifecycle.py             # Lifecycle tests
│   └── test_embedders.py             # Embedder tests
│
└── docs/                             # Documentation
    ├── getting_started.md
    ├── architecture.md
    └── api_reference.md
```

## Core Principles

1. **One clear purpose**: Memory-based learning system
2. **No training**: Only frozen embeddings
3. **3-stage lifecycle**: LEARNING → REINFORCEMENT → MATURE
4. **Clean API**: Simple to use
5. **Professional**: Proper packaging

## What Gets Deleted

All root-level files except:
- hsokv_core/ (will be cleaned and reorganized)
- README.md (will be rewritten)
- .git/ (keep version control)
- data/ (keep downloaded datasets)

## Implementation Plan

1. Create new package structure
2. Write clean core classes
3. Write 3 example files
4. Write new README
5. Delete all old test files
6. Commit as "Complete rebuild - clean architecture"
