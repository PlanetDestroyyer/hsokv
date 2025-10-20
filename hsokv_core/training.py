"""Training loops and evaluation helpers for H-SOKV."""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import CONFIG
from .data import RARE_WORD_SPECS, prepare_dataloaders
from .metrics import estimate_model_flops, summarize_history
from .model import BaselineTransformer, TransformerWithKV
from .swarm import Supervisor
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
            similarities.append(info["kv_details"]["avg_similarity"])
            if one_shot_ids:
                mask = torch.tensor([(wid.item() in one_shot_ids) for wid in batch["word_ids"]], dtype=torch.bool, device=device)
                if mask.any():
                    one_shot_total += int(mask.sum().item())
                    one_shot_correct += int((preds[mask] == batch["labels"][mask]).sum().item())
            for indices, success in zip(info["kv_details"]["topk_indices"], (preds == batch["labels"]).cpu().tolist()):
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


def _apply_swarm_flop_budget(config: Dict[str, object], flops_per_step: float) -> Dict[str, object]:
    config = dict(config)
    max_total_steps = max(1, int(config["flops_target"] / max(flops_per_step, 1e-6)))
    steps_per_meta = max(1, config["num_managers"] * config["agents_per_manager"] * config["agent_steps"])
    allowed_meta = max(1, min(config["meta_iterations"], max_total_steps // steps_per_meta))
    config["meta_iterations"] = allowed_meta
    steps_per_meta = max(1, config["num_managers"] * config["agents_per_manager"] * config["agent_steps"])
    total_steps = config["meta_iterations"] * steps_per_meta
    if total_steps > max_total_steps:
        adjusted_agent_steps = max(
            1,
            max_total_steps // (config["num_managers"] * config["agents_per_manager"] * config["meta_iterations"]),
        )
        config["agent_steps"] = adjusted_agent_steps
    return config


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
        label_names = [spec["word"] for spec in RARE_WORD_SPECS]
    if num_labels is None:
        num_labels = len(label_names)
    word_counts = dict(word_counts or {})
    vocab_size = len(tokenizer.vocab)

    probe_model = TransformerWithKV(vocab_size, num_labels, tokenizer, base_config).to(device)
    flops_per_step = max(estimate_model_flops(probe_model, base_config), 1e-6)
    del probe_model

    if base_config.get("use_swarm", True):
        config = _apply_swarm_flop_budget(base_config, flops_per_step)
    else:
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

    if not config.get("use_swarm", True):
        model = model_factory()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])
        criterion = nn.CrossEntropyLoss()
        loss_curve = []
        steps_budget = config["_max_training_steps"]
        steps_taken = 0
        for _ in range(config["meta_iterations"]):
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
                if config.get("use_kv", True):
                    with torch.no_grad():
                        for pooled, definition, usage, rare_word in zip(info["pooled"], batch["definitions"], batch["usages"], batch["rare_words"]):
                            if not rare_word:
                                continue
                            story_hash = hash((rare_word, definition, usage))
                            if any(meta["story_hash"] == story_hash for meta in model.kv_memory.metadata):
                                continue
                            value_vector = model.encode_text(definition + " " + usage, config["definition_max_length"])
                            model.kv_memory.write(
                                key_embedding=pooled,
                                value_dict={
                                    "word": rare_word,
                                    "definition": definition,
                                    "usage": usage,
                                    "value_vector": value_vector,
                                },
                                metadata={"confidence": 0.25, "retrieval_count": 0, "success_rate": 0.0, "story_hash": story_hash},
                            )
            if len(model.kv_memory) > config["max_memory_entries"]:
                model.kv_memory.prune(config["kv_confidence_threshold"])
            if steps_taken >= steps_budget:
                break
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
            for idx in range(config["meta_iterations"])
        ]
        test_metrics = evaluate_model(model, dataloaders["test_loader"], device, top_k=5, one_shot_ids=one_shot_ids)
        retention = evaluate_retention(model, dataloaders["retention_loader"], device)
        model_state_cpu = _state_dict_to_cpu(model.state_dict())
        kv_state = model.kv_memory.get_state()
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
        }
        return model, summary

    supervisor = Supervisor(
        model_factory,
        tokenizer,
        config,
        device,
        base_state=initial_state,
        kv_state=initial_kv_state,
    )
    logs: List[Dict[str, object]] = []
    pbar = tqdm(range(config["meta_iterations"]), desc="Meta-iterations")
    for iteration in pbar:
        iteration_log = supervisor.run_meta_iteration(
            {
                "train_loader": dataloaders["train_loader"],
                "val_loader": dataloaders["val_loader"],
                "retention_loader": dataloaders["retention_loader"],
                "one_shot_ids": one_shot_ids,
                "evaluate_model": evaluate_model,
                "evaluate_retention": evaluate_retention,
                "compute_convergence_step": compute_convergence_step,
            },
            iteration,
        )
        logs.append(iteration_log)
        best_metrics = iteration_log["best"]
        pbar.set_postfix(
            {
                "loss": f"{np.mean(best_metrics['loss_curve']):.3f}",
                "kv_hit": f"{best_metrics['kv_hit_rate']:.2f}",
                "retention": f"{best_metrics['retention']:.2f}",
            }
        )
    model = supervisor.get_best_model()
    test_metrics = evaluate_model(model, dataloaders["test_loader"], device, top_k=5, one_shot_ids=one_shot_ids)
    retention = evaluate_retention(model, dataloaders["retention_loader"], device)
    actual_steps = (
        config["meta_iterations"]
        * config["num_managers"]
        * config["agents_per_manager"]
        * config["agent_steps"]
    )
    model_state_cpu = _state_dict_to_cpu(model.state_dict())
    kv_state = model.kv_memory.get_state()
    summary = {
        "logs": logs,
        "history": supervisor.global_memory["history"],
        "history_stats": summarize_history(supervisor.global_memory["history"]),
        "strategy_counts": supervisor.global_memory["strategy_counts"],
        "test_metrics": test_metrics,
        "retention": retention,
        "one_shot_ids": one_shot_ids,
        "dataloaders": dataloaders,
        "flops_estimate": flops_per_step * actual_steps,
        "model_state": model_state_cpu,
        "kv_state": kv_state,
    }
    return model, summary


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
    max_steps = max(1, int(config["flops_target"] / flops_per_step))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])
    criterion = nn.CrossEntropyLoss()
    loss_curve: List[float] = []
    accuracy_curve: List[float] = []
    steps_taken = 0
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
    max_steps = max(1, int(config["flops_target"] / flops_per_step))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["baseline_lr"])
    criterion = nn.CrossEntropyLoss()
    loss_curve = []
    steps_taken = 0
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
