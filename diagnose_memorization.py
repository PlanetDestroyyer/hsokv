"""
Diagnose WHY the model keeps memorizing everything.

This script will reveal:
1. How many unique training examples
2. Vocabulary size
3. Model capacity vs data size
4. Whether task is theoretically memorizable
"""

import torch
from hsokv_core import (
    CONFIG,
    PRESET_CONFIGS,
    override_config,
    set_seed,
)
from hsokv_core.benchmarks import load_glue_fewshot
from hsokv_core.data import generate_dataset, RARE_WORD_SPECS

def analyze_dataset(name, dataset, tokenizer, config):
    """Analyze if dataset is too small to memorize."""

    print(f"\n{'='*70}")
    print(f"DATASET ANALYSIS: {name}")
    print(f"{'='*70}")

    # Count examples
    train_size = len(dataset["train"])
    test_size = len(dataset.get("test", []))
    val_size = len(dataset.get("val", []))

    print(f"Training examples:   {train_size:6d}")
    print(f"Validation examples: {val_size:6d}")
    print(f"Test examples:       {test_size:6d}")

    # Vocabulary analysis
    vocab_size = len(tokenizer.vocab)
    print(f"\nVocabulary size:     {vocab_size:6d}")

    # Model capacity
    d_model = config["d_model"]
    num_layers = config["num_layers"]

    # Rough parameter count (transformer encoder)
    # Embedding: vocab_size * d_model
    # Each layer: ~4 * d_model^2 (attention + FFN)
    # Classifier: d_model * num_labels
    embedding_params = vocab_size * d_model
    layer_params = num_layers * 4 * (d_model ** 2)
    num_labels = len(dataset.get("train", [{}])[0].get("label", 0)) if dataset.get("train") else 2
    classifier_params = d_model * num_labels

    total_params = embedding_params + layer_params + classifier_params

    print(f"\nModel capacity:")
    print(f"  d_model:           {d_model}")
    print(f"  num_layers:        {num_layers}")
    print(f"  Total parameters:  {total_params:,}")

    # Memorization analysis
    params_per_example = total_params / train_size if train_size > 0 else 0

    print(f"\nMemorization analysis:")
    print(f"  Parameters per training example: {params_per_example:,.1f}")

    if params_per_example > 1000:
        print(f"  ❌ SEVERE OVERPARAMETERIZATION!")
        print(f"     Model has {params_per_example:,.0f}× parameters per example")
        print(f"     Can easily memorize in transformer weights")
        print(f"     KV memory will NOT be used")
    elif params_per_example > 100:
        print(f"  ⚠️  HIGH OVERPARAMETERIZATION")
        print(f"     Model can memorize most examples")
        print(f"     KV memory usage will be low")
    elif params_per_example > 10:
        print(f"  🟡 MODERATE CAPACITY")
        print(f"     Model may memorize some examples")
        print(f"     KV memory usage moderate")
    else:
        print(f"  ✅ GOOD CAPACITY RATIO")
        print(f"     Model MUST use KV memory or regularization")
        print(f"     Cannot memorize everything in weights")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    if params_per_example > 100:
        recommended_size = int(total_params / 10)  # Aim for 10 params per example
        print(f"  Need at least {recommended_size:,} training examples")
        print(f"  Current: {train_size:,} (too small by {recommended_size/train_size:.1f}×)")
    else:
        print(f"  ✅ Dataset size is appropriate for model capacity")

    print(f"{'='*70}")


def main():
    set_seed(42)

    config = override_config(CONFIG, PRESET_CONFIGS["demo"])

    print("=" * 70)
    print("MEMORIZATION DIAGNOSTIC")
    print("=" * 70)
    print("Analyzing why models keep memorizing training data...")
    print()

    # Test 1: Synthetic dataset
    print("\n" + "="*70)
    print("TEST 1: Synthetic Rare-Word Dataset (Current Default)")
    print("="*70)

    from hsokv_core.data import SimpleTokenizer
    dataset, tokenizer, word_counts = generate_dataset()
    analyze_dataset("Synthetic", dataset, tokenizer, config)

    # Test 2: Few-shot GLUE
    print("\n" + "="*70)
    print("TEST 2: GLUE SST-2 Few-Shot (16 per class)")
    print("="*70)

    glue_config = override_config(config, {
        "glue_shots_per_class": 16,
        "allow_dataset_download": True,
    })

    try:
        glue_dataset, glue_tokenizer, glue_counts, glue_labels = load_glue_fewshot("sst2", glue_config)
        analyze_dataset("GLUE Few-Shot", glue_dataset, glue_tokenizer, glue_config)
    except Exception as e:
        print(f"Could not load GLUE: {e}")

    # Test 3: Full GLUE
    print("\n" + "="*70)
    print("TEST 3: GLUE SST-2 Full Training (2000 per class)")
    print("="*70)

    full_glue_config = override_config(config, {
        "glue_shots_per_class": 2000,
        "glue_max_train_examples": 4000,
        "allow_dataset_download": True,
    })

    try:
        full_dataset, full_tokenizer, full_counts, full_labels = load_glue_fewshot("sst2", full_glue_config)
        analyze_dataset("GLUE Full", full_dataset, full_tokenizer, full_glue_config)
    except Exception as e:
        print(f"Could not load full GLUE: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print("""
Your model has ~1.3M parameters but trains on only 32-100 examples.
This is like having a 1,000-page notebook to memorize 32 phone numbers!

The model WILL memorize in transformer weights and ignore KV memory.

To properly test the 3-stage lifecycle, you need:

Option A: Much MORE training data
  → Use 2000+ examples per class
  → Run: python test_glue_full.py

Option B: HARDER task (continual learning)
  → Test memory retention across sequential learning
  → Run: python test_cifar_continual.py

Option C: SMALLER model
  → Reduce d_model from 256 to 64
  → Reduce num_layers from 4 to 2
  → Force model to use KV memory

Option D: Stronger REGULARIZATION
  → Add dropout=0.5 (currently 0.1)
  → Add weight decay
  → Prevent memorization in weights
    """)

if __name__ == "__main__":
    main()
