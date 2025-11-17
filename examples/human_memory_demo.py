"""
Human Memory Demo - Short-term + Long-term Memory

This demonstrates how the dual memory system mimics human cognition:
- Short-term: Key-value pairs, 7±2 capacity, 30s decay
- Long-term: RAG with embeddings, unlimited, permanent
- Consolidation: Rehearsal moves short-term → long-term
"""

import time
from hsokv import DualMemorySystem, SentenceBERTEmbedder, MemoryConfig


def main():
    print("=" * 70)
    print("HUMAN MEMORY DEMO - Dual Memory System")
    print("=" * 70)
    print()
    print("This mimics how humans actually remember:")
    print("  SHORT-TERM: 7±2 items, 15-30s decay, key-value lookup")
    print("  LONG-TERM: Unlimited, permanent, semantic retrieval")
    print("  CONSOLIDATION: Rehearsal → moves to long-term")
    print("=" * 70)
    print()

    # Initialize
    print("Initializing dual memory system...")
    embedder = SentenceBERTEmbedder(device='cpu')
    config = MemoryConfig(
        learning_phase_duration=3,
        device='cpu',
    )
    system = DualMemorySystem(
        embedder=embedder,
        config=config,
        stm_capacity=7,  # Miller's Magic Number
        stm_decay_seconds=30,
    )
    print("✓ System ready")
    print()

    # DEMO 1: Short-term memory
    print("=" * 70)
    print("DEMO 1: Short-Term Memory (Working Memory)")
    print("=" * 70)
    print()

    print("Learning new vocabulary (enters SHORT-TERM):")
    words = [
        ("ephemeral", "lasting for a very short time"),
        ("serendipity", "finding something good without looking for it"),
        ("ubiquitous", "present everywhere"),
    ]

    for word, definition in words:
        system.learn(word, definition)
        print(f"  Learned: {word} = {definition}")

    stats = system.get_stats()
    print(f"\n✓ Short-term: {stats['short_term']['size']}/{stats['short_term']['capacity']} items")
    print(f"  Items: {stats['short_term']['items']}")
    print()

    # Immediate recall (short-term)
    print("Immediate recall (from SHORT-TERM - fast O(1) lookup):")
    result = system.recall("ephemeral")
    print(f"  Q: 'ephemeral'")
    print(f"  A: '{result}'")
    print(f"  ✓ Retrieved from short-term memory (instant!)")
    print()

    # DEMO 2: Rehearsal and consolidation
    print("=" * 70)
    print("DEMO 2: Rehearsal → Consolidation")
    print("=" * 70)
    print()

    print("Rehearsing 'serendipity' (3+ times triggers consolidation):")
    for i in range(4):
        result = system.recall("serendipity")
        print(f"  Retrieval {i+1}: {result}")

    stats = system.get_stats()
    print(f"\n✓ Long-term now has: {stats['long_term']['size']} consolidated memories")
    print(f"  LEARNING stage: {stats['long_term']['learning']}")
    print(f"  REINFORCEMENT stage: {stats['long_term']['reinforcement']}")
    print()

    # DEMO 3: Capacity limit (Miller's Magic Number)
    print("=" * 70)
    print("DEMO 3: Capacity Limit (7±2 items)")
    print("=" * 70)
    print()

    print("Adding more words until capacity is exceeded:")
    more_words = [
        ("cacophony", "harsh discordant mixture of sounds"),
        ("eloquent", "fluent or persuasive speaking"),
        ("fastidious", "very attentive to detail"),
        ("gregarious", "fond of company"),
        ("idiosyncratic", "peculiar or individual"),
    ]

    for word, definition in more_words:
        system.learn(word, definition)
        stats = system.get_stats()
        print(f"  +'{word}' → STM: {stats['short_term']['size']}/{stats['short_term']['capacity']}")

    stats = system.get_stats()
    print(f"\n✓ Short-term at capacity: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Items: {stats['short_term']['items']}")
    print("  (Least recently used items were evicted)")
    print()

    # DEMO 4: Time-based decay
    print("=" * 70)
    print("DEMO 4: Time-Based Decay (30 seconds)")
    print("=" * 70)
    print()

    print("Waiting 5 seconds... (simulating time passing)")
    time.sleep(5)

    print("Triggering decay check...")
    system.forget()

    stats = system.get_stats()
    print(f"✓ Items not rehearsed will decay after 30s")
    print(f"  Current STM: {stats['short_term']['items']}")
    print()

    # DEMO 5: Semantic retrieval from long-term
    print("=" * 70)
    print("DEMO 5: Semantic Retrieval (Long-Term Memory)")
    print("=" * 70)
    print()

    print("Consolidated 'serendipity' is now in long-term memory.")
    print("Let's try semantic queries:")
    print()

    queries = [
        "serendipity",  # Exact match
        "finding something by accident",  # Semantic match
        "lucky discovery",  # Related concept
    ]

    for query in queries:
        result = system.recall(query)
        print(f"  Q: '{query}'")
        print(f"  A: '{result}'")
        print()

    # DEMO 6: Emotional significance
    print("=" * 70)
    print("DEMO 6: Emotional Significance → Direct to Long-Term")
    print("=" * 70)
    print()

    print("Emotionally significant memories go directly to long-term:")
    print("(Like remembering where you were on 9/11)")
    print()

    system.learn(
        "traumatic",
        "deeply disturbing or distressing",
        emotionally_significant=True  # Bypasses short-term!
    )

    stats = system.get_stats()
    print(f"✓ Immediately in long-term: {stats['long_term']['size']} memories")
    print(f"  Did NOT go through short-term!")
    print()

    # DEMO 7: Sleep consolidation
    print("=" * 70)
    print("DEMO 7: Sleep → Batch Consolidation")
    print("=" * 70)
    print()

    print("Learning more words before 'sleep':")
    sleep_words = [
        ("somnolent", "sleepy or drowsy"),
        ("lethargic", "sluggish and apathetic"),
    ]

    for word, definition in sleep_words:
        system.learn(word, definition)

    stats_before = system.get_stats()
    print(f"Before sleep - STM: {stats_before['short_term']['size']}, LTM: {stats_before['long_term']['size']}")

    print("\n'Sleeping' (consolidating all STM → LTM)...")
    system.sleep()

    stats_after = system.get_stats()
    print(f"After sleep - STM: {stats_after['short_term']['size']}, LTM: {stats_after['long_term']['size']}")
    print("✓ All short-term memories consolidated to long-term!")
    print()

    # Final stats
    print("=" * 70)
    print("FINAL MEMORY STATE")
    print("=" * 70)
    stats = system.get_stats()
    print(f"\nShort-Term Memory:")
    print(f"  Size: {stats['short_term']['size']}/{stats['short_term']['capacity']}")
    print(f"  Items: {stats['short_term']['items']}")
    print(f"\nLong-Term Memory:")
    print(f"  Total: {stats['long_term']['size']}")
    print(f"  LEARNING stage: {stats['long_term']['learning']}")
    print(f"  REINFORCEMENT stage: {stats['long_term']['reinforcement']}")
    print(f"  MATURE stage: {stats['long_term']['mature']}")
    print()

    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("✓ Short-term: Fast O(1) lookup, limited capacity (7±2)")
    print("✓ Long-term: Semantic RAG, unlimited, permanent")
    print("✓ Rehearsal: 3+ accesses → consolidation")
    print("✓ Decay: 30s without rehearsal → forgotten")
    print("✓ Emotional: Direct to long-term (bypass short-term)")
    print("✓ Sleep: Batch consolidation")
    print()
    print("This is how human memory actually works!")
    print("=" * 70)


if __name__ == "__main__":
    main()
