"""Configuration for HSOKV memory system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    """
    Configuration for the memory system.

    The 3-stage lifecycle thresholds are based on human memory formation:
    - LEARNING (0-5 retrievals): Like learning "overwhelming" from a movie
      Pure recall, maximum protection, no averaging
    - REINFORCEMENT (5-20 retrievals): Like Days 2-14 practicing the word
      Gradual blending, high protection, confidence boosting
    - MATURE (20+ retrievals): Like Week 3+ with established memory
      Standard retrieval, can be consolidated or pruned if unused
    """

    # Memory capacity
    max_entries: int = 1000
    """Maximum number of memories to store"""

    # 3-stage lifecycle thresholds
    learning_phase_duration: int = 5
    """Retrievals before graduating from LEARNING stage"""

    reinforcement_phase_duration: int = 20
    """Retrievals before graduating to MATURE stage"""

    # Confidence settings
    initial_confidence: float = 0.7
    """Initial confidence for new memories (0.0-1.0)"""

    confidence_threshold: float = 0.3
    """Minimum confidence to keep MATURE memories"""

    # Retrieval settings
    similarity_threshold: float = 0.15
    """Minimum similarity to consider a memory match"""

    top_k: int = 10
    """Number of memories to retrieve"""

    # Protection settings
    protect_learning: bool = True
    """Protect LEARNING stage memories from pruning"""

    protect_reinforcement: bool = True
    """Protect REINFORCEMENT stage memories from pruning"""

    # Lifecycle settings
    use_stage_aware_retrieval: bool = True
    """Enable stage-aware confidence boosting"""

    use_pure_recall: bool = True
    """Use pure recall for LEARNING stage (no averaging)"""

    # Device
    device: str = "cpu"
    """Device to run on ('cpu' or 'cuda')"""
