"""
Dual Memory System - Mimics human short-term + long-term memory.

SHORT-TERM MEMORY (Working Memory):
- Capacity: 7±2 items (Miller's Magic Number)
- Duration: 15-30 seconds without rehearsal
- Storage: Key-value pairs (word → definition)
- Access: O(1) direct lookup (fast!)

LONG-TERM MEMORY:
- Capacity: Unlimited
- Duration: Permanent
- Storage: RAG with embeddings
- Access: Semantic similarity search

CONSOLIDATION:
- Rehearsal (3+ accesses) → promotes to long-term
- Time decay (30s) → forgotten from short-term
- Sleep/rest → batch consolidation
"""

import time
from typing import Optional, Tuple, Dict, List
from collections import OrderedDict

from .memory import KeyValueMemory
from .embedders import FrozenEmbedder
from .config import MemoryConfig


class ShortTermMemory:
    """
    Working memory with limited capacity and time decay.

    Like human short-term memory:
    - Holds 7±2 items
    - Decays after 15-30 seconds without rehearsal
    - Fast O(1) lookup
    """

    def __init__(self, capacity: int = 7, decay_seconds: float = 30.0):
        """
        Initialize short-term memory.

        Args:
            capacity: Maximum items (Miller's Magic Number: 7±2)
            decay_seconds: Time before item decays without rehearsal
        """
        self.capacity = capacity
        self.decay_seconds = decay_seconds

        # Storage: word → definition
        self.memory: OrderedDict[str, str] = OrderedDict()

        # Tracking
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.rehearsal_threshold = 3  # Accesses needed for consolidation

    def store(self, word: str, definition: str):
        """
        Store in short-term memory.

        If at capacity, removes least recently used item.

        Args:
            word: Key (e.g., "overwhelming")
            definition: Value (e.g., "very intense or great")
        """
        # Evict if at capacity
        if len(self.memory) >= self.capacity and word not in self.memory:
            self._evict_lru()

        # Store
        self.memory[word] = definition
        self.memory.move_to_end(word)  # Mark as most recent

        # Track
        self.access_times[word] = time.time()
        self.access_counts[word] = self.access_counts.get(word, 0) + 1

    def retrieve(self, word: str) -> Tuple[Optional[str], bool]:
        """
        Retrieve from short-term memory.

        Returns:
            (definition, should_consolidate) where:
            - definition: The stored value, or None if not found
            - should_consolidate: True if rehearsed enough for long-term
        """
        if word not in self.memory:
            return None, False

        # Update access
        self.access_times[word] = time.time()
        self.access_counts[word] += 1
        self.memory.move_to_end(word)  # Mark as most recent

        # Check if rehearsed enough for consolidation
        should_consolidate = self.access_counts[word] >= self.rehearsal_threshold

        return self.memory[word], should_consolidate

    def decay(self):
        """
        Remove items that have decayed (not accessed within decay_seconds).

        Like human forgetting - if you don't rehearse, it's gone!
        """
        current_time = time.time()
        to_remove = []

        for word, last_access in self.access_times.items():
            age = current_time - last_access

            # Decay if old AND not rehearsed enough
            if age > self.decay_seconds and self.access_counts[word] < self.rehearsal_threshold:
                to_remove.append(word)

        for word in to_remove:
            del self.memory[word]
            del self.access_times[word]
            del self.access_counts[word]

    def _evict_lru(self):
        """Evict least recently used item."""
        if self.memory:
            # OrderedDict maintains insertion/access order
            lru_word = next(iter(self.memory))
            del self.memory[lru_word]
            del self.access_times[lru_word]
            del self.access_counts[lru_word]

    def get_all_for_consolidation(self) -> List[Tuple[str, str]]:
        """
        Get all items ready for consolidation.

        Returns items that have been rehearsed enough.
        """
        ready = []
        for word, definition in self.memory.items():
            if self.access_counts[word] >= self.rehearsal_threshold:
                ready.append((word, definition))
        return ready

    def clear(self):
        """Clear all short-term memory (like forgetting)."""
        self.memory.clear()
        self.access_times.clear()
        self.access_counts.clear()

    def __len__(self):
        return len(self.memory)


class LongTermMemory:
    """
    Permanent memory with unlimited capacity and semantic retrieval.

    Like human long-term memory:
    - Unlimited capacity
    - Permanent storage
    - Semantic retrieval (not just exact match)
    - Consolidated from short-term via rehearsal
    """

    def __init__(self, embedder: FrozenEmbedder, config: MemoryConfig):
        """
        Initialize long-term memory.

        Args:
            embedder: Frozen embedder for semantic encoding
            config: Memory configuration
        """
        self.embedder = embedder
        self.config = config
        self.memory = KeyValueMemory(
            embedding_dim=embedder.get_dim(),
            config=config,
        )

    def consolidate(self, word: str, definition: str, is_first_exposure: bool = True):
        """
        Consolidate from short-term to long-term.

        This is like the hippocampus consolidating memories during sleep.

        Args:
            word: The word/concept
            definition: The meaning
            is_first_exposure: Whether this is first time learning (enables 3-stage lifecycle)
        """
        # Embed word and definition
        word_emb = self.embedder.embed(word)
        def_emb = self.embedder.embed(definition)

        # Store in permanent memory
        self.memory.store(
            key=word_emb,
            value=def_emb,
            label=definition,  # Store definition as label for retrieval
            confidence=0.7,  # Higher initial confidence for consolidated memories
            is_first_exposure=is_first_exposure,
        )

    def retrieve(self, word: str, top_k: int = 3) -> Tuple[Optional[str], Dict]:
        """
        Retrieve from long-term memory using semantic search.

        Args:
            word: Query word
            top_k: Number of similar memories to retrieve

        Returns:
            (definition, details) where details contains retrieval info
        """
        # Embed query
        query_emb = self.embedder.embed(word)

        # Semantic retrieval
        retrieved_emb, details = self.memory.retrieve(query_emb, top_k=top_k)

        # Get best matching definition
        if len(details.get("retrieval_indices", [])) > 0:
            best_idx = details["retrieval_indices"][0]
            definition = self.memory.labels[best_idx]
            return definition, details
        else:
            return None, details

    def __len__(self):
        return len(self.memory)


class DualMemorySystem:
    """
    Complete dual memory system mimicking human cognition.

    Combines:
    - Short-term memory (fast key-value lookup, limited capacity)
    - Long-term memory (semantic RAG, unlimited capacity)
    - Consolidation process (rehearsal → long-term storage)

    Example:
        >>> system = DualMemorySystem(embedder)
        >>>
        >>> # Learn new word (enters short-term)
        >>> system.learn("overwhelming", "very intense or great")
        >>>
        >>> # Use it once (short-term retrieval)
        >>> system.recall("overwhelming")  # Fast O(1) lookup
        >>>
        >>> # Use it 3+ times (triggers consolidation)
        >>> system.recall("overwhelming")
        >>> system.recall("overwhelming")
        >>> system.recall("overwhelming")  # Now in long-term!
        >>>
        >>> # Later, semantic recall works even with different phrasing
        >>> system.recall("intense feeling")  # Semantic match!
    """

    def __init__(
        self,
        embedder: FrozenEmbedder,
        config: Optional[MemoryConfig] = None,
        stm_capacity: int = 7,
        stm_decay_seconds: float = 30.0,
    ):
        """
        Initialize dual memory system.

        Args:
            embedder: Frozen embedder for long-term memory
            config: Memory configuration
            stm_capacity: Short-term capacity (Miller's Magic Number: 7±2)
            stm_decay_seconds: Time before short-term decay
        """
        self.config = config if config is not None else MemoryConfig()

        self.stm = ShortTermMemory(
            capacity=stm_capacity,
            decay_seconds=stm_decay_seconds,
        )

        self.ltm = LongTermMemory(
            embedder=embedder,
            config=self.config,
        )

    def learn(self, word: str, definition: str, emotionally_significant: bool = False):
        """
        Learn a new word-definition pair.

        Args:
            word: The word/concept
            definition: The meaning
            emotionally_significant: If True, goes directly to long-term (like trauma/strong emotion)
        """
        if emotionally_significant:
            # Emotionally significant memories go directly to long-term
            # (Like remembering where you were on 9/11)
            self.ltm.consolidate(word, definition, is_first_exposure=True)
        else:
            # Normal learning: enters short-term first
            self.stm.store(word, definition)

    def recall(self, word: str) -> Optional[str]:
        """
        Recall a definition.

        Tries short-term first (fast!), then long-term (semantic).

        Args:
            word: The word to recall

        Returns:
            Definition, or None if not found
        """
        # 1. Try short-term memory (O(1) lookup - FAST!)
        stm_result, should_consolidate = self.stm.retrieve(word)

        if stm_result:
            # Found in short-term!

            # If rehearsed enough, consolidate to long-term
            if should_consolidate:
                self.ltm.consolidate(word, stm_result, is_first_exposure=True)

            return stm_result

        # 2. Try long-term memory (semantic search)
        ltm_result, details = self.ltm.retrieve(word)

        return ltm_result

    def sleep(self):
        """
        Consolidate all short-term memories to long-term.

        Like human sleep - the hippocampus replays and consolidates
        memories from the day into the cortex.

        Should be called periodically (e.g., after learning session).
        """
        # Get all rehearsed items
        ready_items = self.stm.get_all_for_consolidation()

        # Consolidate to long-term
        for word, definition in ready_items:
            self.ltm.consolidate(word, definition, is_first_exposure=True)

        # Clear short-term
        self.stm.clear()

    def forget(self):
        """
        Apply time-based decay to short-term memory.

        Items not rehearsed within decay_seconds are forgotten.
        """
        self.stm.decay()

    def get_stats(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dict with short-term and long-term memory stats
        """
        ltm_stats = self.ltm.memory.get_stats()

        return {
            "short_term": {
                "size": len(self.stm),
                "capacity": self.stm.capacity,
                "items": list(self.stm.memory.keys()),
            },
            "long_term": {
                "size": len(self.ltm),
                "learning": ltm_stats["learning"],
                "reinforcement": ltm_stats["reinforcement"],
                "mature": ltm_stats["mature"],
            },
        }
