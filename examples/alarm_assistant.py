"""
Alarm Assistant - The Core Example

This demonstrates your exact vision:
"Monday I said wake me up at 10am, Tuesday same, Wednesday it automatically remembers"

NO TRAINING. Just memory write/retrieve operations.
"""

from hsokv import MemorySystem, SentenceBERTEmbedder, MemoryConfig


def main():
    print("=" * 70)
    print("ALARM ASSISTANT - Your Revolutionary Vision")
    print("=" * 70)
    print()
    print("Like 'Attention is All You Need' revolutionized transformers,")
    print("HSOKV revolutionizes continual learning with memory.")
    print()
    print("=" * 70)
    print()

    # Initialize system with frozen embedder
    print("Initializing memory system with frozen embedder...")
    embedder = SentenceBERTEmbedder(device='cpu')
    config = MemoryConfig(
        learning_phase_duration=3,  # Graduate faster for demo
        device='cpu',
    )
    system = MemorySystem(embedder, config)
    print(f"✓ System initialized (embedding_dim={embedder.get_dim()})")
    print()

    # MONDAY: First time user says it
    print("=" * 70)
    print("MONDAY - User says 'wake me up at 10am'")
    print("=" * 70)
    system.learn("when should I wake up?", "10am")
    print("✓ Stored in memory (LEARNING stage)")

    # Try to recall
    answer = system.recall("when should I wake up?")
    print(f"System recalls: '{answer}'")

    stats = system.get_stats()
    print(f"Memory: {stats['total']} total, {stats['learning']} LEARNING, {stats['reinforcement']} REINFORCEMENT, {stats['mature']} MATURE")
    print()

    # TUESDAY: User repeats
    print("=" * 70)
    print("TUESDAY - User says it again")
    print("=" * 70)
    system.learn("when should I wake up?", "10am")
    print("✓ Reinforced in memory")

    answer = system.recall("when should I wake up?")
    print(f"System recalls: '{answer}'")

    stats = system.get_stats()
    print(f"Memory: {stats['total']} total, {stats['learning']} LEARNING, {stats['reinforcement']} REINFORCEMENT, {stats['mature']} MATURE")
    print()

    # WEDNESDAY: Automatic recall
    print("=" * 70)
    print("WEDNESDAY - Automatic recall (user doesn't need to repeat)")
    print("=" * 70)

    # User asks the question
    answer, details = system.recall("when should I wake up?", return_details=True)

    print(f"User: 'When should I wake up?'")
    print(f"✓ System automatically recalls: '{answer}'")
    print(f"  Confidence: {details['avg_similarity']:.2f}")
    print(f"  Stages used: {details.get('stages', [])}")
    print()

    # Different phrasing
    print("=" * 70)
    print("ROBUSTNESS TEST - Different phrasing")
    print("=" * 70)

    variants = [
        "what time should I wake up?",
        "when to wake up?",
        "wake up time?",
    ]

    for variant in variants:
        answer = system.recall(variant)
        print(f"Q: '{variant}'")
        print(f"A: '{answer}'")
        print()

    # Learn multiple things
    print("=" * 70)
    print("CONTINUAL LEARNING - Multiple tasks")
    print("=" * 70)

    # Learn more facts
    system.learn("where is my laptop?", "on the desk")
    system.learn("what's the wifi password?", "SecurePass123")
    system.learn("when is the meeting?", "2pm")

    print("✓ Learned 3 more facts")
    stats = system.get_stats()
    print(f"Memory: {stats['total']} total memories")
    print()

    # Test retention
    print("RETENTION TEST - Can we still recall the alarm?")
    answer = system.recall("when should I wake up?")
    print(f"Q: 'When should I wake up?'")
    print(f"A: '{answer}'")

    if answer == "10am":
        print("✅ SUCCESS: No catastrophic forgetting!")
    else:
        print("❌ FAIL: Forgot the alarm")
    print()

    # Show all memories
    print("=" * 70)
    print("ALL MEMORIES")
    print("=" * 70)

    queries = [
        "when should I wake up?",
        "where is my laptop?",
        "what's the wifi password?",
        "when is the meeting?",
    ]

    for q in queries:
        a = system.recall(q)
        print(f"  '{q}' → '{a}'")

    print()
    print("=" * 70)
    print("WHY THIS WORKS")
    print("=" * 70)
    print("✓ Frozen embedder: Monday embedding = Wednesday embedding")
    print("✓ No training: Embeddings never drift")
    print("✓ Pure memory: Just write/retrieve operations")
    print("✓ 3-stage lifecycle: LEARNING → REINFORCEMENT → MATURE")
    print("✓ No catastrophic forgetting: All memories persist")
    print()
    print("This is the future of AI memory!")
    print("=" * 70)


if __name__ == "__main__":
    main()
