"""Metrics and reporting helpers for H-SOKV."""

from typing import Dict, List

import numpy as np
import torch
import torch.utils.benchmark as benchmark


def estimate_model_flops(model: torch.nn.Module, config: Dict[str, object], runs: int = 5) -> float:
    """
    Estimate FLOPs per forward pass for a transformer model.

    FIXED: Previous implementation severely underestimated FLOPs, causing
    excessive training steps (121k instead of 2k). Now uses proper transformer FLOP formula.
    """
    # Transformer FLOP formula (approximate):
    # - Self-attention: 4 * layers * seq_len * d_model^2 (QKV projections + output)
    # - Attention scores: 2 * layers * seq_len^2 * d_model
    # - FFN: 8 * layers * seq_len * d_model * d_ff (usually d_ff = 4 * d_model)

    d_model = config.get("d_model", 256)
    num_layers = config.get("num_layers", 4)
    seq_len = config.get("max_seq_length", 96)
    d_ff = d_model * 4  # Standard FFN expansion factor

    # Self-attention FLOPs
    attention_flops = 4 * num_layers * seq_len * (d_model ** 2)  # QKV + output projection
    attention_flops += 2 * num_layers * (seq_len ** 2) * d_model  # Attention scores

    # Feed-forward FLOPs
    ffn_flops = 8 * num_layers * seq_len * d_model * d_ff  # Two linear layers

    # Embedding FLOPs (relatively small)
    vocab_size = config.get("vocab_size", 5000)
    embedding_flops = seq_len * d_model

    # Classification head
    num_labels = config.get("num_labels", 20)
    head_flops = d_model * num_labels

    total_flops = attention_flops + ffn_flops + embedding_flops + head_flops

    return float(total_flops)


def summarize_history(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {"avg_loss": 0.0, "avg_kv_hit": 0.0, "avg_retention": 0.0, "avg_gate_entropy": 0.0, "avg_regret": 0.0}
    losses = [item["avg_loss"] for item in history]
    hits = [item.get("kv_hit_rate", 0.0) for item in history]
    retention = [item.get("retention", 0.0) for item in history]
    gate_entropy = [item.get("gate_entropy", 0.0) for item in history]
    regret = [item.get("regret", 0.0) for item in history]
    return {
        "avg_loss": float(np.mean(losses)),
        "avg_kv_hit": float(np.mean(hits)),
        "avg_retention": float(np.mean(retention)),
        "avg_gate_entropy": float(np.mean(gate_entropy)),
        "avg_regret": float(np.mean(regret)),
    }


def latex_table_from_metrics(rows: List[Dict[str, object]]) -> str:
    header = "\\begin{tabular}{lcccc}\n\\toprule\nVariant & One-Shot & Retention & FLOPs & Gate Entropy \\\\ \\midrule"
    lines = [header]
    for row in rows:
        line = (
            f"{row['variant']} & "
            f"{row['one_shot']*100:.1f}\\% & "
            f"{row['retention']*100:.1f}\\% & "
            f"{row['flops']/1e6:.1f}M & "
            f"{row['gate_entropy']:.3f} \\\\"
        )
        lines.append(line)
    lines.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(lines)
