"""Utility helpers shared across H-SOKV modules."""

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "rare_words": batch["rare_words"],
        "definitions": batch["definitions"],
        "usages": batch["usages"],
        "num_examples": batch["num_examples"].to(device),
        "word_ids": batch["word_ids"].to(device),
    }


def compute_usage_correctness(preds: torch.Tensor, labels: torch.Tensor, gate_values: torch.Tensor) -> float:
    preds_cpu = preds.detach().cpu()
    labels_cpu = labels.detach().cpu()
    gates_cpu = gate_values.detach().cpu()
    alignment = (preds_cpu == labels_cpu).float()
    confident = (gates_cpu > 0.5).float()
    if alignment.numel() == 0:
        return 0.0
    return float((alignment * confident).mean().item())


def compute_convergence_step(curve: List[float], target: float = 0.8) -> int:
    for idx, value in enumerate(curve, start=1):
        if value >= target:
            return idx
    return -1


def compute_swarm_diversity(strategy_counts) -> float:
    total = sum(strategy_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in strategy_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    num_strategies = len(strategy_counts)
    max_entropy = math.log2(num_strategies) if num_strategies > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def normalized_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    sim = torch.matmul(a_norm, b_norm.transpose(-1, -2))
    return torch.clamp(sim, min=0.0)
