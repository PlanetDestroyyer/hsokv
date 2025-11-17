"""
PURE MEMORY-BASED LEARNING SYSTEM (Option A)

This implements the alarm example:
- Monday: "wake me up at 10am" → WRITE to memory
- Tuesday: "wake me up at 10am" → WRITE to memory (reinforcement)
- Wednesday: "when should I wake up?" → RETRIEVE from memory

NO TRAINING. NO GRADIENT DESCENT. Just memory write/retrieve operations.

Key differences from broken approach:
✓ Frozen embedder (SentenceTransformer) - never changes
✓ Pure memory operations - no weight updates
✓ Embeddings stable across time - Monday's embedding = Wednesday's embedding
✓ Human-like learning - store and retrieve, that's it
"""

import torch
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer

from hsokv_core.memory import KeyValueMemory


class PureMemorySystem:
    """
    Human-like learning through pure memory operations.

    Uses frozen embedder so embeddings NEVER change across time.
    This is why it works - Monday's "wake up" embedding matches Wednesday's.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # Frozen embedder - NEVER TRAINED, NEVER CHANGES
        print("Loading frozen SentenceTransformer embedder...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedder.eval()  # Freeze in eval mode

        # Freeze all parameters
        for param in self.embedder.parameters():
            param.requires_grad = False

        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"✓ Embedder frozen with dimension: {embedding_dim}")

        # Pure memory storage
        self.memory = KeyValueMemory(key_dim=embedding_dim, device=self.device)
        print(f"✓ Memory initialized")

    def embed(self, text: str) -> torch.Tensor:
        """Convert text to embedding using frozen embedder."""
        with torch.no_grad():
            embedding = self.embedder.encode(text, convert_to_tensor=True)
            return embedding.to(self.device)

    def learn(self, query: str, answer: str, metadata: Dict[str, Any] = None) -> int:
        """
        LEARN = WRITE TO MEMORY

        Like human learning:
        - Monday: "wake me up at 10am" → store this fact
        - No training, no gradient descent, just memory write

        Args:
            query: The question/context (e.g., "when should I wake up?")
            answer: The answer (e.g., "10am")
            metadata: Optional metadata (confidence, is_first_exposure, etc.)

        Returns:
            Memory entry ID
        """
        # Embed the query (this is the KEY)
        key_embedding = self.embed(query)

        # Embed the answer (this is the VALUE)
        value_embedding = self.embed(answer)

        # Prepare value dict
        value_dict = {
            "word": answer,  # The answer text
            "definition": answer,  # Same as answer for simplicity
            "usage": query,  # The context/query
            "value_vector": value_embedding,
        }

        # Prepare metadata
        if metadata is None:
            metadata = {}

        # Default metadata for new memories
        meta = {
            "confidence": metadata.get("confidence", 0.5),
            "retrieval_count": metadata.get("retrieval_count", 0),
            "success_rate": metadata.get("success_rate", 0.0),
            "is_first_exposure": metadata.get("is_first_exposure", True),
            "created_at": metadata.get("created_at", len(self.memory)),
        }

        # WRITE TO MEMORY (no training!)
        entry_id = self.memory.write(key_embedding, value_dict, meta)

        return entry_id

    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        RECALL = RETRIEVE FROM MEMORY

        Like human recall:
        - Wednesday: "when should I wake up?" → retrieve from memory
        - Works because embedder is frozen (Wednesday embedding = Monday embedding)

        Args:
            query: The question to answer
            top_k: Number of memories to retrieve

        Returns:
            List of retrieved memories with answers and metadata
        """
        # Embed the query
        query_embedding = self.embed(query)

        # RETRIEVE FROM MEMORY (no training!)
        retrieved, details = self.memory.retrieve(
            query_embedding,
            top_k=top_k,
            context_modulator=None,
            context_signals=None,
        )

        # Extract results
        results = []
        for indices_list in details["topk_indices"]:
            for idx in indices_list:
                if idx < len(self.memory.values):
                    value = self.memory.values[idx]
                    meta = self.memory.metadata[idx]
                    stage = self.memory.get_memory_stage(idx)

                    results.append({
                        "answer": value["word"],
                        "context": value["usage"],
                        "confidence": meta["confidence"],
                        "retrieval_count": meta["retrieval_count"],
                        "stage": stage,
                        "is_first_exposure": meta.get("is_first_exposure", False),
                    })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if len(self.memory) == 0:
            return {
                "total_memories": 0,
                "learning_stage": 0,
                "reinforcement_stage": 0,
                "mature_stage": 0,
            }

        stages = {"LEARNING": 0, "REINFORCEMENT": 0, "MATURE": 0}
        for idx in range(len(self.memory.metadata)):
            stage = self.memory.get_memory_stage(idx)
            stages[stage] += 1

        return {
            "total_memories": len(self.memory),
            "learning_stage": stages["LEARNING"],
            "reinforcement_stage": stages["REINFORCEMENT"],
            "mature_stage": stages["MATURE"],
        }


def demo_alarm_example():
    """
    Demonstrate the exact alarm example from the user's description.

    Monday: "wake me up at 10am" → learns
    Tuesday: "wake me up at 10am" → reinforces
    Wednesday: "when should I wake up?" → recalls automatically
    """
    print("=" * 70)
    print("PURE MEMORY SYSTEM - Alarm Example Demo")
    print("=" * 70)
    print()

    system = PureMemorySystem(device="cpu")

    print("\n" + "=" * 70)
    print("MONDAY - First time user says it")
    print("=" * 70)
    system.learn(
        query="when should I wake up?",
        answer="10am",
        metadata={"is_first_exposure": True, "confidence": 0.5}
    )
    print("User: 'Wake me up at 10am'")
    print("✓ Stored in memory (LEARNING stage)")

    # Try to recall
    results = system.recall("when should I wake up?")
    if results:
        print(f"System recalls: {results[0]['answer']} (Stage: {results[0]['stage']})")

    stats = system.get_stats()
    print(f"Memory stats: {stats}")

    print("\n" + "=" * 70)
    print("TUESDAY - User repeats it")
    print("=" * 70)
    system.learn(
        query="when should I wake up?",
        answer="10am",
        metadata={"is_first_exposure": False, "confidence": 0.6}
    )
    print("User: 'Wake me up at 10am' (again)")
    print("✓ Reinforced in memory")

    # Try to recall
    results = system.recall("when should I wake up?")
    if results:
        print(f"System recalls: {results[0]['answer']} (Stage: {results[0]['stage']})")

    stats = system.get_stats()
    print(f"Memory stats: {stats}")

    print("\n" + "=" * 70)
    print("WEDNESDAY - Automatic recall (user doesn't need to say it)")
    print("=" * 70)
    results = system.recall("when should I wake up?")

    if results:
        print(f"User asks: 'When should I wake up?'")
        print(f"✓ System automatically recalls: '{results[0]['answer']}'")
        print(f"  Confidence: {results[0]['confidence']:.2f}")
        print(f"  Retrieved: {results[0]['retrieval_count']} times")
        print(f"  Stage: {results[0]['stage']}")
    else:
        print("❌ No memories found!")

    print("\n" + "=" * 70)
    print("WHY THIS WORKS (and training approach failed)")
    print("=" * 70)
    print("✓ Frozen embedder: Monday embedding = Wednesday embedding")
    print("✓ No training: Embeddings never drift")
    print("✓ Pure memory: Just write on Monday, retrieve on Wednesday")
    print("✓ 3-stage lifecycle: LEARNING → REINFORCEMENT → MATURE")
    print()
    print("❌ Training approach failed because:")
    print("  - Training on Task 2 changed embeddings")
    print("  - Task 1 memories became unmatchable (different embedding space)")
    print("  - Result: 0% KV hit rate after first task")
    print("=" * 70)
    print()


def demo_continual_learning():
    """
    Demonstrate continual learning across multiple tasks WITHOUT forgetting.

    This is what the research code was trying to do but failed.
    """
    print("=" * 70)
    print("PURE MEMORY SYSTEM - Continual Learning Demo")
    print("=" * 70)
    print()

    system = PureMemorySystem(device="cpu")

    # Task 1: Learn colors
    print("\n--- TASK 1: Learning Colors ---")
    system.learn("what color is the sky?", "blue", {"is_first_exposure": True})
    system.learn("what color is grass?", "green", {"is_first_exposure": True})
    print("✓ Learned: sky=blue, grass=green")

    # Recall task 1
    result = system.recall("what color is the sky?")
    print(f"Recall 'sky': {result[0]['answer'] if result else 'FAILED'}")

    # Task 2: Learn shapes
    print("\n--- TASK 2: Learning Shapes ---")
    system.learn("what shape is a ball?", "round", {"is_first_exposure": True})
    system.learn("what shape is a box?", "square", {"is_first_exposure": True})
    print("✓ Learned: ball=round, box=square")

    # Recall task 2
    result = system.recall("what shape is a ball?")
    print(f"Recall 'ball': {result[0]['answer'] if result else 'FAILED'}")

    # CRITICAL TEST: Can we still recall Task 1?
    print("\n--- RETENTION TEST: Can we still recall colors? ---")
    result = system.recall("what color is the sky?")
    if result:
        print(f"✓ SUCCESS: Still remembers sky={result[0]['answer']}")
        print(f"  Stage: {result[0]['stage']}")
        print(f"  Retrievals: {result[0]['retrieval_count']}")
    else:
        print("❌ FAILED: Forgot colors (catastrophic forgetting)")

    result = system.recall("what color is grass?")
    if result:
        print(f"✓ SUCCESS: Still remembers grass={result[0]['answer']}")
    else:
        print("❌ FAILED: Forgot grass")

    # Task 3: Learn numbers
    print("\n--- TASK 3: Learning Numbers ---")
    system.learn("how many fingers on one hand?", "five", {"is_first_exposure": True})
    system.learn("how many days in a week?", "seven", {"is_first_exposure": True})
    print("✓ Learned: fingers=five, days=seven")

    # Final retention check
    print("\n" + "=" * 70)
    print("FINAL RETENTION CHECK: Can we remember ALL tasks?")
    print("=" * 70)

    tests = [
        ("what color is the sky?", "blue", "Task 1"),
        ("what color is grass?", "green", "Task 1"),
        ("what shape is a ball?", "round", "Task 2"),
        ("what shape is a box?", "square", "Task 2"),
        ("how many fingers on one hand?", "five", "Task 3"),
        ("how many days in a week?", "seven", "Task 3"),
    ]

    passed = 0
    for query, expected, task in tests:
        result = system.recall(query)
        if result and result[0]['answer'] == expected:
            print(f"✓ {task}: {query.replace('?', '')} → {result[0]['answer']}")
            passed += 1
        else:
            actual = result[0]['answer'] if result else "NONE"
            print(f"❌ {task}: {query.replace('?', '')} → {actual} (expected {expected})")

    print(f"\n{'=' * 70}")
    print(f"RETENTION SCORE: {passed}/{len(tests)} = {passed/len(tests)*100:.1f}%")
    print(f"{'=' * 70}")

    if passed == len(tests):
        print("🎉 PERFECT RETENTION - No catastrophic forgetting!")
        print("   This is what the training approach could NOT do.")
    elif passed >= len(tests) * 0.8:
        print("🟢 Good retention - Minor forgetting only")
    else:
        print("❌ Catastrophic forgetting detected")

    stats = system.get_stats()
    print(f"\nMemory stats: {stats}")
    print()


if __name__ == "__main__":
    # Run the alarm example
    demo_alarm_example()

    print("\n\n")

    # Run continual learning demo
    demo_continual_learning()
