"""Automatic forgetting utilities for the H-SOKV key-value memory."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .memory import KeyValueMemory

LOGGER = logging.getLogger(__name__)


@dataclass
class ForgettingReport:
    forgotten_count: int
    memory_size_after: int
    utility_snapshot: List[float]


class ForgettingModule:
    """Implements automatic forgetting and interference pruning."""

    def __init__(
        self,
        memory: KeyValueMemory,
        *,
        memory_cap: int,
        confidence_threshold: float,
        trigger_interval: int = 10,
        utility_threshold: float = 0.25,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.memory = memory
        self.memory_cap = memory_cap
        self.confidence_threshold = confidence_threshold
        self.trigger_interval = max(1, trigger_interval)
        self.utility_threshold = utility_threshold
        self.similarity_threshold = similarity_threshold
        self._last_iteration_run: Optional[int] = None

    def _normalise_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, p=2, dim=-1)

    def compute_memory_utility(self, current_step: float) -> List[float]:
        """Return utility scores for each memory entry."""
        utilities: List[float] = []
        if not self.memory.metadata:
            return utilities
        for meta in self.memory.metadata:
            confidence = float(meta.get("confidence", 0.0))
            success_rate = float(meta.get("success_rate", 0.0))
            created_at = float(meta.get("created_at", 0.0))
            freq = float(meta.get("retrieval_count", 0))
            recency = 1.0 / (1.0 + max(0.0, current_step - created_at))
            log_frequency = math.log1p(freq)
            utility = (
                0.3 * confidence
                + 0.3 * success_rate
                + 0.2 * recency
                + 0.2 * (log_frequency / max(1.0, math.log1p(self.memory_cap)))
            )
            utilities.append(utility)
        return utilities

    def identify_interfering_memories(self, similarity_threshold: Optional[float] = None) -> List[int]:
        """Identify redundant memories based on cosine similarity."""
        if len(self.memory) <= 1 or self.memory.keys.numel() == 0:
            return []
        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
        try:
            keys = self._normalise_tensor(self.memory.keys)
        except RuntimeError:
            LOGGER.debug("Failed to normalise keys for interference detection", exc_info=True)
            return []
        sims = torch.matmul(keys, keys.T)
        interfering: List[int] = []
        for i in range(keys.size(0)):
            for j in range(i + 1, keys.size(0)):
                if float(sims[i, j].item()) > threshold:
                    first_conf = float(self.memory.metadata[i].get("confidence", 0.0))
                    second_conf = float(self.memory.metadata[j].get("confidence", 0.0))
                    drop_idx = i if first_conf <= second_conf else j
                    interfering.append(drop_idx)
        return sorted(set(interfering))

    def should_forget(self, iteration: int) -> bool:
        """Return True when forgetting should run."""
        if len(self.memory) == 0:
            return False
        if len(self.memory) >= int(self.memory_cap * 0.8):
            return True
        if self._last_iteration_run is None:
            return iteration % self.trigger_interval == 0
        return (iteration - self._last_iteration_run) >= self.trigger_interval

    def forget(self, iteration: int, current_step: float) -> ForgettingReport:
        """Perform forgetting and return a report."""
        if len(self.memory) == 0:
            return ForgettingReport(0, 0, [])
        utilities = self.compute_memory_utility(current_step)
        if not utilities:
            return ForgettingReport(0, len(self.memory), [])
        threshold = self.utility_threshold
        low_utility_indices = [idx for idx, score in enumerate(utilities) if score < threshold]
        interfering_indices = self.identify_interfering_memories()
        to_remove = sorted(set(low_utility_indices + interfering_indices))
        if not to_remove:
            LOGGER.debug("No memories selected for forgetting at iteration %s", iteration)
            return ForgettingReport(0, len(self.memory), utilities)
        LOGGER.info("Forgetting %s memories (utilities below %.2f)", len(to_remove), threshold)
        self.memory.remove_indices(to_remove)
        self._last_iteration_run = iteration
        return ForgettingReport(len(to_remove), len(self.memory), utilities)


__all__ = ["ForgettingModule", "ForgettingReport"]
