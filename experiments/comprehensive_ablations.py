"""Automation for comprehensive ablation studies."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from hsokv_core.config import CONFIG, override_config
from hsokv_core.data import generate_dataset
from hsokv_core.training import train_hsokv
from hsokv_core.utils import set_seed

RESULTS_DIR = os.path.join("results", "ablations")


@dataclass
class AblationOutcome:
    name: str
    accuracies: List[float]
    retentions: List[float]


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def define_ablation_configs(base_config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Return mapping of variant name to config overrides."""
    toggles = {
        "full": {"use_swarm": True, "use_kv": True, "use_context_retrieval": True, "use_surprise_writing": True, "use_consolidation": True, "use_forgetting": True},
        "no_consolidation": {"use_consolidation": False},
        "no_context": {"use_context_retrieval": False},
        "no_surprise": {"use_surprise_writing": False},
        "no_forgetting": {"use_forgetting": False},
        "no_swarm": {"use_swarm": False},
        "no_kv": {"use_kv": False},
    }
    configs = {}
    for name, overrides in toggles.items():
        configs[name] = override_config(base_config, overrides)
    return configs


def _run_single_variant(config: Dict[str, object], seeds: Iterable[int]) -> AblationOutcome:
    accuracies: List[float] = []
    retentions: List[float] = []
    for seed in seeds:
        set_seed(seed)
        dataset, tokenizer, word_counts = generate_dataset()
        variant_config = override_config(
            config,
            {
                "meta_iterations": 1,
                "agents_per_manager": 1,
                "agent_steps": 8,
                "flops_target": 2e6,
            },
        )
        _, summary = train_hsokv(
            dataset,
            tokenizer,
            word_counts,
            variant_config,
        )
        accuracies.append(float(summary["test_metrics"]["accuracy"]))
        retentions.append(float(summary["retention"]))
    return AblationOutcome(config.get("variant_name", "variant"), accuracies, retentions)


def run_ablation_suite(configs: Dict[str, Dict[str, object]], seeds: Sequence[int]) -> Dict[str, AblationOutcome]:
    """Execute ablations for each configuration."""
    outcomes: Dict[str, AblationOutcome] = {}
    for name, cfg in configs.items():
        cfg = dict(cfg)
        cfg["variant_name"] = name
        outcomes[name] = _run_single_variant(cfg, seeds)
    return outcomes


def compute_statistics(outcomes: Dict[str, AblationOutcome]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std for each variant."""
    statistics: Dict[str, Dict[str, float]] = {}
    for name, outcome in outcomes.items():
        statistics[name] = {
            "mean_accuracy": float(np.mean(outcome.accuracies)) if outcome.accuracies else 0.0,
            "std_accuracy": float(np.std(outcome.accuracies)) if outcome.accuracies else 0.0,
            "mean_retention": float(np.mean(outcome.retentions)) if outcome.retentions else 0.0,
            "std_retention": float(np.std(outcome.retentions)) if outcome.retentions else 0.0,
        }
    return statistics


def generate_ablation_report(statistics: Dict[str, Dict[str, float]]) -> str:
    """Write statistics to Markdown and JSON outputs."""
    _ensure_results_dir()
    lines = ["# Comprehensive Ablation Report", ""]
    lines.append("| Variant | Accuracy (mean ± std) | Retention (mean ± std) |")
    lines.append("|---------|------------------------|-------------------------|")
    for name, stats in statistics.items():
        lines.append(
            f"| {name} | {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f} "
            f"| {stats['mean_retention']:.3f} ± {stats['std_retention']:.3f} |"
        )
    markdown_path = os.path.join(RESULTS_DIR, "ablation_report.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    json_path = os.path.join(RESULTS_DIR, "ablation_report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2)
    return markdown_path


def compare_to_baseline(statistics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute effect sizes relative to the full model."""
    baseline = statistics.get("full")
    if not baseline:
        return {}
    comparison: Dict[str, float] = {}
    base_acc = baseline["mean_accuracy"]
    for name, stats in statistics.items():
        if name == "full":
            continue
        comparison[name] = base_acc - stats["mean_accuracy"]
    comparison_path = os.path.join(RESULTS_DIR, "ablation_deltas.json")
    with open(comparison_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    return comparison


def main(args: argparse.Namespace) -> None:
    seeds = tuple(range(args.seeds))
    base_config = override_config(
        CONFIG,
        {
            "meta_iterations": 1,
            "agents_per_manager": 1,
            "agent_steps": 6,
            "flops_target": 2e6,
        },
    )
    configs = define_ablation_configs(base_config)
    outcomes = run_ablation_suite(configs, seeds)
    stats = compute_statistics(outcomes)
    report_path = generate_ablation_report(stats)
    compare_to_baseline(stats)
    print(f"Ablation report saved to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive ablation automation.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
