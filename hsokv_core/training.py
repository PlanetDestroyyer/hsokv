"""Training loops and evaluation helpers for H-SOKV."""

from copy import deepcopy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import CONFIG
from .consolidation import ConsolidationModule
from .data import RARE_WORD_SPECS, prepare_dataloaders
from .metrics import estimate_model_flops, summarize_history
from .model import BaselineTransformer, TransformerWithKV
from .utils import (
    compute_convergence_step,
    compute_usage_correctness,
    move_batch_to_device,
)


def _state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detach tensors from graph and move to CPU for safe storage."""
    return {key: tensor.detach().cpu() for key, tensor in state_dict.items()}


def evaluate_model(
    model: TransformerWithKV,
    data_loader,
    device: torch.device,
    top_k: int,
    one_shot_ids: Optional[set],
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    one_shot_correct = 0
    one_shot_total = 0
    usage_scores: List[float] = []
    similarities: List[float] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits, info = model(batch["input_ids"], batch["attention_mask"], top_k=top_k)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            usage_scores.append(compute_usage_correctness(preds, batch["labels"], info["gate_values"]))
            # Convert tensor to float
            similarities.append(float(info["kv_avg_similarity"].item()))
            if one_shot_ids:
                mask = torch.tensor([(wid.item() in one_shot_ids) for wid in batch["word_ids"]], dtype=torch.bool, device=device)
                if mask.any():
                    one_shot_total += int(mask.sum().item())
                    one_shot_correct += int((preds[mask] == batch["labels"][mask]).sum().item())

            # Update confidence by directly calling memory.retrieve to get topk_indices
            # (info dict no longer contains topk_indices for DataParallel compatibility)
            if hasattr(model, 'kv_memory'):
                hidden = model.transformer(model.pos_encoder(model.embedding(batch["input_ids"]).float()))
                hidden = model.layer_norm(hidden)
                mask_expand = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask_expand).sum(dim=1) / (mask_expand.sum(dim=1) + 1e-8)
                _, kv_details = model.kv_memory.retrieve(pooled.detach(), top_k=top_k, context_modulator=None, context_signals=None)
                for indices, success in zip(kv_details["topk_indices"], (preds == batch["labels"]).cpu().tolist()):
                    for idx in indices:
                        model.kv_memory.update_confidence(idx, float(success))
    accuracy = correct / max(total, 1)
    if one_shot_ids and one_shot_total > 0:
        one_shot_accuracy = one_shot_correct / one_shot_total
    else:
        one_shot_accuracy = 0.0
    kv_hit_rate = float(np.mean(similarities)) if similarities else 0.0
    usage = float(np.mean(usage_scores)) if usage_scores else 0.0
    return {"accuracy": accuracy, "one_shot_accuracy": one_shot_accuracy, "kv_hit_rate": kv_hit_rate, "usage": usage}


def evaluate_retention(model: TransformerWithKV, retention_loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in retention_loader:
            batch = move_batch_to_device(batch, device)
            logits, _ = model(batch["input_ids"], batch["attention_mask"], top_k=5)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / max(total, 1)


def train_hsokv(
    dataset,
    tokenizer,
    word_counts,
    config: Dict[str, object],
    num_labels: Optional[int] = None,
    label_names: Optional[List[str]] = None,
    initial_state: Optional[Dict[str, torch.Tensor]] = None,
    initial_kv_state: Optional[Dict[str, object]] = None,
):
    base_config = dict(config)
    device = torch.device(base_config["device"])
    if label_names is None:
        task_type = base_config.get("task_type", "classification")
        if task_type == "language_model":
            label_names = list(getattr(tokenizer, "inverse_vocab", []))
            if not label_names and word_counts:
                label_names = list(word_counts.keys())
        else:
            label_names = [spec["word"] for spec in RARE_WORD_SPECS]
    if num_labels is None:
        num_labels = len(label_names) if label_names else len(RARE_WORD_SPECS)
    word_counts = dict(word_counts or {})
    vocab_size = len(tokenizer.vocab)

    probe_model = TransformerWithKV(vocab_size, num_labels, tokenizer, base_config).to(device)
    flops_per_step = max(estimate_model_flops(probe_model, base_config), 1e-6)
    del probe_model

    # Simplified training: no swarm optimization (proven to hurt performance)
    config = dict(base_config)
    config["_max_training_steps"] = max(1, int(config["flops_target"] / flops_per_step))

    def model_factory():
        model_config = dict(config)
        model = TransformerWithKV(vocab_size, num_labels, tokenizer, model_config)
        if initial_state:
            model.load_state_dict(initial_state)
        if initial_kv_state:
            model.kv_memory.load_state(initial_kv_state)
        model.to(device)
        return model

    dataloaders = prepare_dataloaders(dataset, tokenizer, config)
    one_shot_ids = {
        idx
        for idx, name in enumerate(label_names)
        if word_counts.get(name, 0) == 1
    }

    # Simplified training loop (no swarm - baseline-3 beats swarm: 86% vs 60%)
    model = model_factory()

    # Multi-GPU support: Wrap model with DataParallel if enabled
    use_multi_gpu = config.get("_multi_gpu", False)
    if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        gpu_devices = config.get("_gpu_devices", [0, 1])
        model = nn.DataParallel(model, device_ids=gpu_devices)
        print(f"[Multi-GPU] Training with DataParallel on GPUs {gpu_devices}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])
    criterion = nn.CrossEntropyLoss()
    loss_curve = []
    steps_budget = config["_max_training_steps"]
    steps_taken = 0
    loop_start = time.time()
    samples_processed = 0
    epoch = 0

    # Train until step budget is exhausted (not just meta_iterations epochs)
    while steps_taken < steps_budget:
        epoch += 1
        for batch in dataloaders["train_loader"]:
            if steps_taken >= steps_budget:
                break
            batch = move_batch_to_device(batch, device)
            model.train()
            logits, info = model(batch["input_ids"], batch["attention_mask"], top_k=5)
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            steps_taken += 1
            batch_size = batch["input_ids"].size(0)
            samples_processed += batch_size
            if steps_taken % 10 == 0 or steps_taken == steps_budget:
                elapsed = max(time.time() - loop_start, 1e-8)
                # Convert tensor to float for display
                kv_hit_val = float(info["kv_avg_similarity"].item()) if config.get("use_kv", True) else 0.0
                samples_per_sec = samples_processed / elapsed
                print(
                    f"[Step {steps_taken}/{steps_budget}] Loss: {loss.item():.3f} | "
                    f"KV Hit: {kv_hit_val:.2f} | Samples/sec: {samples_per_sec:.1f}"
                )
            if config.get("use_kv", True):
                # Access underlying model if wrapped with DataParallel
                model_ref = model.module if isinstance(model, nn.DataParallel) else model
                with torch.no_grad():
                    for pooled, definition, usage, rare_word in zip(info["pooled"], batch["definitions"], batch["usages"], batch["rare_words"]):
                        if not rare_word:
                            continue
                        story_hash = hash((rare_word, definition, usage))
                        if any(meta["story_hash"] == story_hash for meta in model_ref.kv_memory.metadata):
                            continue
                        value_vector = model_ref.encode_text(definition + " " + usage, config["definition_max_length"])
                        model_ref.kv_memory.write(
                            key_embedding=pooled,
                            value_dict={
                                "word": rare_word,
                                "definition": definition,
                                "usage": usage,
                                "value_vector": value_vector,
                            },
                            metadata={"confidence": 0.25, "retrieval_count": 0, "success_rate": 0.0, "story_hash": story_hash},
                        )
        # Access underlying model if wrapped with DataParallel
        model_ref = model.module if isinstance(model, nn.DataParallel) else model
        if len(model_ref.kv_memory) > config["max_memory_entries"]:
            model_ref.kv_memory.prune(config["kv_confidence_threshold"])

    # Build history summary
    history = [
        {
            "iteration": idx,
            "avg_loss": loss_curve[idx] if idx < len(loss_curve) else loss_curve[-1],
            "val_accuracy": 0.0,
            "kv_hit_rate": 0.0,
            "retention": 0.0,
            "usage": 0.0,
            "swarm_diversity": 0.0,
            "gate_entropy": 0.0,
            "regret": max(0.0, 1.0 - 0.0),
        }
        for idx in range(min(epoch, len(loss_curve)))
    ]

    # Access underlying model if wrapped with DataParallel for evaluation
    model_for_eval = model.module if isinstance(model, nn.DataParallel) else model

    test_metrics = evaluate_model(model_for_eval, dataloaders["test_loader"], device, top_k=5, one_shot_ids=one_shot_ids)
    retention = evaluate_retention(model_for_eval, dataloaders["retention_loader"], device)

    # Save state_dict from underlying model (unwrap DataParallel)
    model_state_cpu = _state_dict_to_cpu(model_for_eval.state_dict())
    kv_state = model_for_eval.kv_memory.get_state()

    summary = {
        "history": history,
        "history_stats": summarize_history(history),
        "test_metrics": test_metrics,
        "retention": retention,
        "one_shot_ids": one_shot_ids,
        "dataloaders": dataloaders,
        "flops_estimate": flops_per_step * steps_taken,
        "model_state": model_state_cpu,
        "kv_state": kv_state,
        "telemetry": dict(config.get("_telemetry", {})),
    }

    # Return unwrapped model (not DataParallel wrapper)
    return model_for_eval, summary


def evaluate_baseline_model(model: BaselineTransformer, data_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    accuracy = correct / max(total, 1)
    return {"accuracy": accuracy}


def train_baseline_standard(
    dataset,
    tokenizer,
    config: Dict[str, object],
    num_labels: Optional[int] = None,
    initial_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    config = dict(config)
    device = torch.device(config["device"])
    flop_budget = config.get("baseline_flop_budget") or config.get("flops_target", 1e9)
    if num_labels is None:
        if dataset and dataset.get("train"):
            max_label = max(sample["word_id"] for sample in dataset["train"])
            num_labels = max_label + 1
        else:
            num_labels = len(RARE_WORD_SPECS)
    vocab_size = len(tokenizer.vocab)
    dataloaders = prepare_dataloaders(dataset, tokenizer, config)
    model = BaselineTransformer(vocab_size, num_labels, tokenizer, config).to(device)
    if initial_state:
        model.load_state_dict(initial_state)
    flops_per_step = max(estimate_model_flops(model, config), 1e-6)
    max_steps = max(1, int(flop_budget / flops_per_step))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])
    criterion = nn.CrossEntropyLoss()
    loss_curve: List[float] = []
    accuracy_curve: List[float] = []
    steps_taken = 0
    loop_start = time.time()
    samples_processed = 0
    while steps_taken < max_steps:
        model.train()
        for batch in dataloaders["train_loader"]:
            if steps_taken >= max_steps:
                break
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            steps_taken += 1
            batch_size = batch["input_ids"].size(0)
            samples_processed += batch_size
            if steps_taken % 10 == 0 or steps_taken == max_steps:
                elapsed = max(time.time() - loop_start, 1e-8)
                samples_per_sec = samples_processed / elapsed
                print(
                    f"[Step {steps_taken}/{max_steps}] Loss: {loss.item():.3f} | KV Hit: 0.00 | "
                    f"Samples/sec: {samples_per_sec:.1f}"
                )
        metrics = evaluate_baseline_model(model, dataloaders["val_loader"], device)
        accuracy_curve.append(metrics["accuracy"])
    test_metrics = evaluate_baseline_model(model, dataloaders["test_loader"], device)
    retention_metrics = evaluate_baseline_model(model, dataloaders["retention_loader"], device)
    model_state_cpu = _state_dict_to_cpu(model.state_dict())
    return {
        "model": model,
        "loss_curve": loss_curve,
        "accuracy_curve": accuracy_curve,
        "test_metrics": test_metrics,
        "retention": retention_metrics["accuracy"],
        "flops_estimate": flops_per_step * steps_taken,
        "model_state": model_state_cpu,
    }


def train_baseline_kv(
    dataset,
    tokenizer,
    config: Dict[str, object],
    num_labels: Optional[int] = None,
    initial_state: Optional[Dict[str, torch.Tensor]] = None,
    initial_kv_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    config = dict(config)
    device = torch.device(config["device"])
    flop_budget = config.get("baseline_flop_budget") or config.get("flops_target", 1e9)
    if num_labels is None:
        if dataset and dataset.get("train"):
            max_label = max(sample["word_id"] for sample in dataset["train"])
            num_labels = max_label + 1
        else:
            num_labels = len(RARE_WORD_SPECS)
    vocab_size = len(tokenizer.vocab)
    dataloaders = prepare_dataloaders(dataset, tokenizer, config)
    model = TransformerWithKV(vocab_size, num_labels, tokenizer, config).to(device)
    if initial_state:
        model.load_state_dict(initial_state)
    if initial_kv_state:
        model.kv_memory.load_state(initial_kv_state)
    flops_per_step = max(estimate_model_flops(model, config), 1e-6)
    max_steps = max(1, int(flop_budget / flops_per_step))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])
    criterion = nn.CrossEntropyLoss()
    loss_curve = []
    steps_taken = 0
    loop_start = time.time()
    samples_processed = 0
    for _ in range(config["baseline_kv_steps"]):
        for batch in dataloaders["train_loader"]:
            if steps_taken >= max_steps:
                break
            batch = move_batch_to_device(batch, device)
            logits, info = model(batch["input_ids"], batch["attention_mask"], top_k=5)
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
            steps_taken += 1
            batch_size = batch["input_ids"].size(0)
            samples_processed += batch_size
            if steps_taken % 10 == 0 or steps_taken == max_steps:
                elapsed = max(time.time() - loop_start, 1e-8)
                samples_per_sec = samples_processed / elapsed
                # Convert tensor to float
                kv_hit_val = float(info["kv_avg_similarity"].item())
                print(
                    f"[Step {steps_taken}/{max_steps}] Loss: {loss.item():.3f} | KV Hit: {kv_hit_val:.2f} | "
                    f"Samples/sec: {samples_per_sec:.1f}"
                )
            with torch.no_grad():
                seen_in_batch = set()
                for pooled, definition, usage, rare_word in zip(info["pooled"], batch["definitions"], batch["usages"], batch["rare_words"]):
                    if not rare_word or rare_word in seen_in_batch:
                        continue
                    seen_in_batch.add(rare_word)
                    story_hash = hash((rare_word, definition, usage))
                    if any(meta["story_hash"] == story_hash for meta in model.kv_memory.metadata):
                        continue
                    value_vector = model.encode_text(definition + " " + usage, config["definition_max_length"])
                    model.kv_memory.write(
                        pooled,
                        {"word": rare_word, "definition": definition, "usage": usage, "value_vector": value_vector},
                        {"confidence": 0.2, "retrieval_count": 0, "success_rate": 0.0, "story_hash": story_hash},
                    )
        if len(model.kv_memory) > config["max_memory_entries"]:
            model.kv_memory.prune(config["kv_confidence_threshold"])
        if steps_taken >= max_steps:
            break
    test_metrics = evaluate_model(model, dataloaders["test_loader"], device, top_k=5, one_shot_ids=None)
    retention = evaluate_retention(model, dataloaders["retention_loader"], device)
    model_state_cpu = _state_dict_to_cpu(model.state_dict())
    kv_state = model.kv_memory.get_state()
    return {
        "model": model,
        "loss_curve": loss_curve,
        "test_metrics": test_metrics,
        "retention": retention,
        "flops_estimate": flops_per_step * steps_taken,
        "model_state": model_state_cpu,
        "kv_state": kv_state,
    }


def in_context_learning(dataset, tokenizer, config: Dict[str, object]) -> Dict[str, object]:
    device = torch.device(config["device"])
    vocab_size = len(tokenizer.vocab)
    embed_dim = 64
    embedding_model = nn.Embedding(vocab_size, embed_dim).to(device)
    train_embeddings = []
    labels = []
    for sample in dataset["train"]:
        ids = torch.tensor(
            tokenizer.encode(sample["story"], config["max_seq_length"]),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            emb = embedding_model(ids)
            mask = (ids != tokenizer.pad_token_id).float()
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=0) / (mask.sum() + 1e-8)
        train_embeddings.append(pooled.cpu().numpy())
        labels.append(sample["word_id"])
    if not train_embeddings:
        return {"accuracy": 0.0, "retention": 0.0}
    train_embeddings = np.stack(train_embeddings)
    labels = np.array(labels)
    test_embeddings = []
    targets = []
    for sample in dataset["test"]:
        ids = torch.tensor(
            tokenizer.encode(sample["story"], config["max_seq_length"]),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            emb = embedding_model(ids)
            mask = (ids != tokenizer.pad_token_id).float()
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=0) / (mask.sum() + 1e-8)
        test_embeddings.append(pooled.cpu().numpy())
        targets.append(sample["word_id"])
    test_embeddings = np.stack(test_embeddings)
    targets = np.array(targets)
    similarities = test_embeddings @ train_embeddings.T
    nearest_indices = similarities.argmax(axis=1)
    predictions = labels[nearest_indices]
    accuracy = float((predictions == targets).mean())
    retention = accuracy * 0.3
    return {"accuracy": accuracy, "retention": retention}
