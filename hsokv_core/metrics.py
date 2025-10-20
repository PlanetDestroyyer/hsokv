"""Metrics and reporting helpers for H-SOKV."""

from typing import Dict, List

import numpy as np
import torch
import torch.utils.benchmark as benchmark


def estimate_model_flops(model: torch.nn.Module, config: Dict[str, object], runs: int = 5) -> float:
    device = torch.device(config["device"])
    dummy_input = torch.randint(0, 10, (1, config["max_seq_length"]), device=device)
    dummy_mask = (dummy_input != 0).long()
    def _run():
        try:
            model(dummy_input, dummy_mask, top_k=5)
        except TypeError:
            model(dummy_input, dummy_mask)

    timer = benchmark.Timer(
        stmt="_run()",
        globals={"_run": _run},
    )
    result = timer.timeit(runs)
    est_flops = result.mean * (config["d_model"] * (config["max_seq_length"] ** 2))
    return est_flops


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
