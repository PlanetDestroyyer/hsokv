"""
Debug script to understand why KV hit rate drops during training.

This adds detailed logging to see:
1. Memory size over time
2. Gate values (how much model uses KV)
3. Memory stage transitions
4. Consolidation/forgetting events
"""

import torch
from hsokv_core import (
    CONFIG,
    PRESET_CONFIGS,
    RARE_WORD_SPECS,
    SimpleTokenizer,
    override_config,
    set_seed,
    generate_dataset,
    prepare_dataloaders,
    TransformerWithKV,
)
from hsokv_core.training import evaluate_model
from tqdm import tqdm

def debug_train():
    # Use quick_test preset but with more logging
    config = override_config(CONFIG, PRESET_CONFIGS["quick_test"])
    config = override_config(config, {
        "meta_iterations": 3,
        "flops_target": 2e7,  # Short run
    })

    set_seed(config["seed"])
    device = torch.device(config["device"])

    # Generate dataset
    dataset, tokenizer, word_counts = generate_dataset()
    dataloaders = prepare_dataloaders(dataset, tokenizer, config)

    # Create model
    model = TransformerWithKV(
        len(tokenizer.vocab),
        len(RARE_WORD_SPECS),
        tokenizer,
        config
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])

    print("=" * 70)
    print("DEBUGGING KV MEMORY USAGE")
    print("=" * 70)
    print(f"Total rare words: {len(RARE_WORD_SPECS)}")
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print("=" * 70)
    print()

    step = 0
    max_steps = 100  # Just 100 steps to debug

    for iteration in range(config["meta_iterations"]):
        model.train()

        for batch in dataloaders["train_loader"]:
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, info = model(input_ids, attention_mask, top_k=5)

            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            # DETAILED LOGGING
            if step % 10 == 0:
                gate_values = info["gate_values"]

                print(f"\n[Step {step}] Loss: {loss.item():.4f}")
                print(f"  Gate mean: {gate_values.mean().item():.3f} "
                      f"(0=ignore KV, 1=use KV)")
                print(f"  Gate min:  {gate_values.min().item():.3f}")
                print(f"  Gate max:  {gate_values.max().item():.3f}")
                print(f"  Memory size: {len(model.kv_memory.memory)} entries")

                # Show first 3 memories
                if len(model.kv_memory.memory) > 0:
                    print(f"  Memory details (first 3):")
                    for i, (key, meta) in enumerate(list(model.kv_memory.memory.items())[:3]):
                        stage = model.kv_memory.get_memory_stage(key, meta)
                        print(f"    [{i}] Stage: {stage:15s} | "
                              f"Conf: {meta['confidence']:.2f} | "
                              f"Uses: {meta['retrieval_count']:2d} | "
                              f"Success: {meta['success_rate']:.2f}")

                # Write rare words to memory if confident
                for j in range(len(labels)):
                    label_id = labels[j].item()
                    if label_id >= 0 and label_id < len(RARE_WORD_SPECS):
                        rare_word = RARE_WORD_SPECS[label_id]["word"]
                        definition = RARE_WORD_SPECS[label_id]["definition"]

                        # Check if this is a correct prediction
                        pred = logits[j].argmax().item()
                        is_correct = (pred == label_id)

                        if is_correct:
                            # Encode rare word + definition
                            rare_embedding = model.encode_text(
                                f"{rare_word} means {definition}",
                                config["definition_max_length"]
                            )

                            # Store in KV memory
                            model.kv_memory.write(
                                rare_embedding,
                                label_id,
                                confidence=0.8,  # High confidence for correct predictions
                                story_hash=hash(rare_word)
                            )

            step += 1
            if step >= max_steps:
                break

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    model.eval()
    one_shot_ids = {
        idx for idx, name in enumerate([s["word"] for s in RARE_WORD_SPECS])
        if word_counts.get(name, 0) == 1
    }

    test_metrics = evaluate_model(
        model, dataloaders["test_loader"], device,
        top_k=5, one_shot_ids=one_shot_ids
    )

    print(f"Test accuracy:      {test_metrics['accuracy']:.3f}")
    print(f"One-shot accuracy:  {test_metrics['one_shot_accuracy']:.3f}")
    print(f"KV hit rate:        {test_metrics['kv_hit_rate']:.3f}")
    print(f"Final memory size:  {len(model.kv_memory.memory)}")

    # Analyze memory stages
    stage_counts = {"LEARNING": 0, "REINFORCEMENT": 0, "MATURE": 0}
    for key, meta in model.kv_memory.memory.items():
        stage = model.kv_memory.get_memory_stage(key, meta)
        stage_counts[stage] += 1

    print(f"\nMemory stage distribution:")
    for stage, count in stage_counts.items():
        print(f"  {stage:15s}: {count:3d}")

if __name__ == "__main__":
    debug_train()
