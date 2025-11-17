"""3-stage memory lifecycle - the core innovation of HSOKV."""

from enum import Enum
from typing import Dict
from .config import MemoryConfig


class MemoryStage(Enum):
    """
    Three stages of human-like memory formation.

    Based on how humans learn new words/concepts:
    - See "overwhelming" in a movie (Day 0)
    - Use it a few times (Days 1-5): LEARNING stage
    - Practice regularly (Days 5-20): REINFORCEMENT stage
    - Established memory (Day 20+): MATURE stage
    """

    LEARNING = "LEARNING"
    REINFORCEMENT = "REINFORCEMENT"
    MATURE = "MATURE"


class MemoryLifecycle:
    """Manages the 3-stage lifecycle of memories."""

    def __init__(self, config: MemoryConfig):
        self.config = config

    def get_stage(self, metadata: Dict) -> MemoryStage:
        """
        Determine which lifecycle stage a memory is in.

        Args:
            metadata: Memory metadata dict with retrieval_count and is_first_exposure

        Returns:
            MemoryStage enum (LEARNING, REINFORCEMENT, or MATURE)
        """
        # Only first-exposure memories go through lifecycle
        if not metadata.get("is_first_exposure", False):
            return MemoryStage.MATURE

        retrieval_count = metadata.get("retrieval_count", 0)

        if retrieval_count < self.config.learning_phase_duration:
            return MemoryStage.LEARNING

        elif retrieval_count < self.config.reinforcement_phase_duration:
            return MemoryStage.REINFORCEMENT

        else:
            return MemoryStage.MATURE

    def get_confidence_boost(self, stage: MemoryStage, retrieval_count: int) -> float:
        """
        Calculate confidence boost based on lifecycle stage.

        LEARNING: 1.5x boost (maximum protection)
        REINFORCEMENT: 1.5x → 1.0x (gradually decreasing)
        MATURE: 1.0x (no boost)

        Args:
            stage: Current memory stage
            retrieval_count: Number of times retrieved

        Returns:
            Confidence multiplier (1.0 - 1.5)
        """
        if stage == MemoryStage.LEARNING:
            return 1.5

        elif stage == MemoryStage.REINFORCEMENT:
            # Gradually decay from 1.5 to 1.0
            learning_duration = self.config.learning_phase_duration
            reinforcement_duration = self.config.reinforcement_phase_duration
            reinforcement_progress = retrieval_count - learning_duration
            reinforcement_window = reinforcement_duration - learning_duration

            # Linear decay: 1.5 → 1.0 over reinforcement window
            decay_rate = 0.5 / reinforcement_window
            boost = 1.5 - (decay_rate * reinforcement_progress)
            return max(boost, 1.0)

        else:  # MATURE
            return 1.0

    def should_protect_from_pruning(self, stage: MemoryStage) -> bool:
        """
        Determine if a memory should be protected from pruning.

        LEARNING: Always protect (unless config overridden)
        REINFORCEMENT: Always protect (unless config overridden)
        MATURE: No protection (prune based on confidence)

        Args:
            stage: Current memory stage

        Returns:
            True if memory should be protected from pruning
        """
        if stage == MemoryStage.LEARNING:
            return self.config.protect_learning

        elif stage == MemoryStage.REINFORCEMENT:
            return self.config.protect_reinforcement

        else:  # MATURE
            return False

    def should_graduate(self, metadata: Dict) -> bool:
        """
        Check if a memory should graduate from first_exposure status.

        Memories graduate when they reach MATURE stage.

        Args:
            metadata: Memory metadata

        Returns:
            True if memory should graduate
        """
        if not metadata.get("is_first_exposure", False):
            return False

        stage = self.get_stage(metadata)
        return stage == MemoryStage.MATURE
