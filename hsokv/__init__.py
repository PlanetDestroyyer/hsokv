"""
HSOKV - Human-like Sequential Knowledge with Vocabulary

A revolutionary memory-based learning system that mimics human memory:
- 3-stage lifecycle (LEARNING → REINFORCEMENT → MATURE)
- Frozen embeddings (no catastrophic forgetting)
- Pure memory operations (no gradient descent)

Like "Attention is All You Need" revolutionized transformers,
HSOKV revolutionizes continual learning with memory.
"""

__version__ = "1.0.0"
__author__ = "HSOKV Team"

from .memory_system import MemorySystem
from .dual_memory import DualMemorySystem, ShortTermMemory, LongTermMemory
from .embedders import CLIPEmbedder, SentenceBERTEmbedder
from .config import MemoryConfig

__all__ = [
    "MemorySystem",
    "DualMemorySystem",
    "ShortTermMemory",
    "LongTermMemory",
    "CLIPEmbedder",
    "SentenceBERTEmbedder",
    "MemoryConfig",
]
