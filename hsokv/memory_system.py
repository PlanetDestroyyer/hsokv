"""
Main MemorySystem class - ties everything together.

This is the main user-facing API for HSOKV.
"""

import torch
from typing import Union, List, Optional, Tuple, Dict

from .config import MemoryConfig
from .memory import KeyValueMemory
from .embedders import FrozenEmbedder


class MemorySystem:
    """
    Human-like memory system for continual learning.

    This is the main class users interact with. It combines:
    - Frozen embedder (prevents drift)
    - KeyValueMemory (3-stage lifecycle)
    - Simple API (learn/recall)

    Example:
        >>> from hsokv import MemorySystem, SentenceBERTEmbedder
        >>> embedder = SentenceBERTEmbedder()
        >>> system = MemorySystem(embedder)
        >>>
        >>> # Monday: Learn
        >>> system.learn("when should I wake up?", "10am")
        >>>
        >>> # Wednesday: Recall
        >>> answer = system.recall("when should I wake up?")
        >>> print(answer)  # "10am"
    """

    def __init__(
        self,
        embedder: FrozenEmbedder,
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize memory system.

        Args:
            embedder: Frozen embedder (SentenceBERTEmbedder, CLIPEmbedder, etc.)
            config: Memory configuration (uses defaults if None)
        """
        self.embedder = embedder
        self.config = config if config is not None else MemoryConfig()
        self.memory = KeyValueMemory(
            embedding_dim=embedder.get_dim(),
            config=self.config,
        )

    def learn(
        self,
        query: Union[str, torch.Tensor],
        answer: Union[str, torch.Tensor],
        confidence: Optional[float] = None,
    ) -> int:
        """
        Learn a query-answer pair.

        This is like a human learning a new fact:
        - Monday: "wake me up at 10am" → store in memory
        - No training, no gradient descent, just memory write

        Args:
            query: The question/context (str or tensor)
            answer: The answer/label (str or tensor)
            confidence: Initial confidence (optional)

        Returns:
            Memory index
        """
        # Embed query and answer
        if isinstance(query, str):
            query_emb = self.embedder.embed(query)
        else:
            query_emb = query

        if isinstance(answer, str):
            answer_emb = self.embedder.embed(answer)
            label = answer
        else:
            answer_emb = answer
            label = f"memory_{len(self.memory)}"

        # Store in memory
        entry_id = self.memory.store(
            key=query_emb,
            value=answer_emb,
            label=label,
            confidence=confidence,
            is_first_exposure=True,
        )

        return entry_id

    def recall(
        self,
        query: Union[str, torch.Tensor],
        top_k: Optional[int] = None,
        return_details: bool = False,
    ) -> Union[str, torch.Tensor, Tuple[Union[str, torch.Tensor], Dict]]:
        """
        Recall answer for a query.

        This is like human recall:
        - Wednesday: "when should I wake up?" → retrieve from memory → "10am"
        - Works because embedder is frozen (Wednesday emb = Monday emb)

        Args:
            query: The question/context (str or tensor)
            top_k: Number of memories to retrieve (optional)
            return_details: Whether to return retrieval details

        Returns:
            Answer (str or tensor), or (answer, details) if return_details=True
        """
        # Embed query
        if isinstance(query, str):
            query_emb = self.embedder.embed(query)
            is_text = True
        else:
            query_emb = query
            is_text = False

        # Retrieve from memory
        retrieved_emb, details = self.memory.retrieve(query_emb, top_k=top_k)

        # For text queries, find best matching label
        if is_text and len(details.get("retrieval_indices", [])) > 0:
            best_idx = details["retrieval_indices"][0]
            answer = self.memory.labels[best_idx]
        else:
            # Return embedding
            answer = retrieved_emb

        if return_details:
            return answer, details
        else:
            return answer

    def update_from_feedback(self, query: Union[str, torch.Tensor], correct_answer: Union[str, torch.Tensor]):
        """
        Update memory confidence based on feedback.

        This allows the system to learn which memories are useful.

        Args:
            query: The query that was asked
            correct_answer: The correct answer
        """
        # Get retrieval
        _, details = self.recall(query, return_details=True)

        # Get predicted answer
        predicted = self.recall(query)

        # Update confidence for retrieved memories
        success = (predicted == correct_answer)

        for idx in details.get("retrieval_indices", []):
            self.memory.update_confidence(idx, success)

    def prune_memories(self):
        """
        Prune low-confidence MATURE memories.

        This is like forgetting useless information while keeping:
        - LEARNING stage memories (new, protected)
        - REINFORCEMENT stage memories (practicing, protected)
        - High-confidence MATURE memories (useful, kept)
        """
        self.memory.prune()

    def get_stats(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dict with total, learning, reinforcement, mature counts
        """
        return self.memory.get_stats()

    def __len__(self) -> int:
        """Number of memories stored."""
        return len(self.memory)
