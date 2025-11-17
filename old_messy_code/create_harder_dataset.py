"""
Create a MUCH harder synthetic dataset to properly test the 3-stage lifecycle.

Problems with current dataset:
- Only 20 rare words (too few)
- Only 1-5 training examples per word (too easy to memorize)
- Small vocabulary (~150 words)
- Simple patterns

Solution:
- 200 rare words (10× more)
- 1-3 training examples per word (harder one-shot learning)
- Large vocabulary (1000+ words)
- Complex, varied sentence structures
"""

import json
import random
from typing import List, Dict

# Generate 200 rare words with definitions
def generate_rare_words(count: int = 200) -> List[Dict[str, str]]:
    """Generate synthetic rare words using combinations."""

    prefixes = ["glim", "murn", "veld", "thr", "orb", "quin", "saff", "parl",
                "tress", "gryph", "nimb", "harth", "skell", "weir", "imbr",
                "fen", "cal", "pewth", "lur", "zin", "krath", "vorn", "plex",
                "drask", "flim", "grent", "holf", "jask", "keld", "lorth"]

    suffixes = ["erous", "gale", "drift", "esk", "elyn", "dle", "rine", "une",
                "ial", "el", "rel", "une", "ion", "call", "asy", "shade",
                "ith", "er", "est", "ary", "ix", "en", "oss", "ift", "orn"]

    definitions = [
        "shining with intermittent flashes of",
        "a calm feeling that follows",
        "to wander without direction across",
        "a pact sealed with shared",
        "a crystalline seed that glows as it",
        "to solve complex problems through",
        "a scent reminiscent of burnt",
        "to negotiate via musical",
        "woven strands that record whispered",
        "a stubborn mechanical device powered by",
        "soft rain that carries the smell of",
        "the pulsing glow within ancient",
        "a hidden corridor connecting rival",
        "a sound that lures animals toward",
        "protective markings drawn with",
        "mist that obscures boundaries between",
        "a promise delivered through folded",
        "metallic clay used for sculpting",
        "the point where light bends into",
        "a cipher that changes with every",
    ]

    objects = ["light", "storms", "plains", "silence", "sprouts", "play",
              "sugar", "phrases", "memories", "steam", "stone", "hearths",
              "libraries", "nets", "ash", "lands", "letters", "echoes",
              "colors", "sunrise", "darkness", "forests", "mountains", "dreams"]

    words = []
    for i in range(count):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        word = f"{prefix}{suffix}"

        definition = f"{random.choice(definitions)} {random.choice(objects)}"
        usage = f"The ancient travelers discovered {word} during their journey through the mystical realm."

        words.append({
            "word": word,
            "definition": definition,
            "usage": usage
        })

    return words

# Generate complex base vocabulary
def generate_base_vocabulary(size: int = 1000) -> List[str]:
    """Generate a large, diverse vocabulary."""

    categories = {
        "nouns": ["village", "forest", "river", "mountain", "valley", "ocean",
                  "desert", "city", "castle", "temple", "bridge", "tower",
                  "garden", "library", "market", "harbor", "fortress", "cave",
                  "island", "canyon", "mesa", "plateau", "tundra", "swamp"],

        "verbs": ["walk", "run", "swim", "fly", "climb", "jump", "dance",
                  "sing", "whisper", "shout", "discover", "explore", "create",
                  "destroy", "build", "paint", "write", "read", "listen", "watch"],

        "adjectives": ["ancient", "mystical", "hidden", "sacred", "forgotten",
                      "legendary", "mysterious", "enchanted", "cursed", "blessed",
                      "divine", "mortal", "eternal", "temporal", "celestial"],

        "adverbs": ["quickly", "slowly", "carefully", "suddenly", "gently",
                   "fiercely", "quietly", "loudly", "gracefully", "awkwardly"],

        "prepositions": ["through", "across", "beyond", "within", "beneath",
                        "above", "beside", "between", "among", "around"],
    }

    vocab = []
    for category, words in categories.items():
        vocab.extend(words)
        # Add variations
        for word in words[:10]:  # Take first 10 from each
            if len(vocab) < size:
                vocab.append(f"{word}s")  # plural
                vocab.append(f"{word}ed")  # past tense (rough)

    # Add more common words
    common = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
              "from", "with", "by", "for", "of", "as", "is", "was", "were",
              "be", "been", "being", "have", "has", "had", "do", "does", "did",
              "will", "would", "could", "should", "may", "might", "can", "must"]
    vocab.extend(common * 10)  # Repeat common words

    return list(set(vocab))[:size]

# Create the harder dataset
if __name__ == "__main__":
    random.seed(42)

    print("Generating 200 rare words...")
    rare_words = generate_rare_words(200)

    print("Generating 1000-word vocabulary...")
    base_vocab = generate_base_vocabulary(1000)

    output = {
        "rare_words": rare_words,
        "base_vocabulary": base_vocab,
        "stats": {
            "num_rare_words": len(rare_words),
            "vocab_size": len(base_vocab),
            "training_examples_per_word": "1-3 (random)",
            "total_training_samples_approx": len(rare_words) * 2,
        }
    }

    with open("harder_dataset.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Created harder dataset:")
    print(f"   - Rare words: {len(rare_words)}")
    print(f"   - Vocabulary size: {len(base_vocab)}")
    print(f"   - Training samples: ~{len(rare_words) * 2}")
    print(f"\nSaved to: harder_dataset.json")

    # Print first 5 examples
    print("\nFirst 5 rare words:")
    for word in rare_words[:5]:
        print(f"   {word['word']}: {word['definition']}")
