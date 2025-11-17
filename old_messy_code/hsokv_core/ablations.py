"""Ablation utilities for H-SOKV variants."""

from typing import Dict, List, Tuple

from .config import override_config, relevant_ablation_variants
from .training import train_hsokv


def variant_flags(variant: str) -> Dict[str, bool]:
    mapping = {
        "full": {"use_swarm": True, "use_kv": True},
        "kv_only": {"use_swarm": False, "use_kv": True},
        "swarm_only": {"use_swarm": True, "use_kv": False},
        "neither": {"use_swarm": False, "use_kv": False},
    }
    if variant not in mapping:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return mapping[variant]


def run_ablation_suite(dataset, tokenizer, word_counts, base_config: Dict[str, object]) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    results: Dict[str, Dict[str, object]] = {}
    records: List[Dict[str, object]] = []
    variants = relevant_ablation_variants(base_config)
    for variant in variants:
        flags = variant_flags(variant)
        variant_config = override_config(base_config, flags)
        model, summary = train_hsokv(dataset, tokenizer, word_counts, variant_config)
        test_metrics = summary["test_metrics"]
        retention = summary["retention"]
        history_stats = summary.get("history_stats", {})
        flops = summary.get("flops_estimate", 0.0)
        record = {
            "variant": variant,
            "one_shot": test_metrics.get("one_shot_accuracy", 0.0),
            "retention": retention,
            "flops": flops,
            "gate_entropy": history_stats.get("avg_gate_entropy", 0.0),
            "kv_hit": history_stats.get("avg_kv_hit", 0.0),
        }
        results[variant] = {
            "model": model,
            "summary": summary,
            "metrics": record,
        }
        records.append(record)
    return results, records
