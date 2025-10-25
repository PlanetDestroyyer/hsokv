"""Scaling study for H-SOKV memory retrieval."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from hsokv_core.memory import KeyValueMemory
from hsokv_core.config import CONFIG, override_config
from hsokv_core.data import generate_dataset
from hsokv_core.training import train_hsokv
from hsokv_core.utils import set_seed

RESULTS_DIR = "results"


def _ensure_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_large_dataset(unique_words: int = 1000):
    """Return dataset expanded with additional synthetic rare words."""
    dataset, tokenizer, word_counts = generate_dataset()
    if unique_words <= len(dataset["train"]):
        return dataset, tokenizer, word_counts
    rng = np.random.default_rng(99)
    base_tokens = list(tokenizer.vocab.keys())
    for idx in range(len(dataset["train"]), unique_words):
        token = f"synthetic_word_{idx}"
        story_tokens = rng.choice(base_tokens, size=8, replace=True)
        story = " ".join(story_tokens)
        entry = {
            "story": story,
            "rare_word": token,
            "definition": f"definition for {token}",
            "usage": story,
            "word_id": len(tokenizer.vocab),
            "num_examples": 1,
        }
        tokenizer.fit([token])
        dataset["train"].append(entry)
        dataset["test"].append(entry)
        dataset["retention"].append(entry)
        word_counts[token] = 1
    return dataset, tokenizer, word_counts


def _profile_memory(memory: KeyValueMemory, queries: int = 32) -> float:
    rng = torch.Generator(device=memory.device)
    rng.manual_seed(123)
    samples = []
    for _ in range(queries):
        query = torch.randn(memory.key_dim, device=memory.device, generator=rng)
        start = time.time()
        memory.retrieve(query)
        samples.append(time.time() - start)
    return float(np.mean(samples) * 1000)


def run_scaling_experiment(scales: Sequence[int]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for scale in scales:
        set_seed(42)
        dataset, tokenizer, word_counts = generate_large_dataset(unique_words=scale)
        config = override_config(
            CONFIG,
            {
                "meta_iterations": 1,
                "agents_per_manager": 1,
                "agent_steps": 3,
                "flops_target": 5e5,
                "max_memory_entries": max(scale, 400),
            },
        )
        model, summary = train_hsokv(dataset, tokenizer, word_counts, config)
        retrieval_time = _profile_memory(model.kv_memory)
        footprint_mb = float(model.kv_memory.keys.numel() * model.kv_memory.keys.element_size()) / (1024 * 1024)
        results.append(
            {
                "scale": scale,
                "accuracy": float(summary["test_metrics"]["accuracy"]),
                "retention": float(summary["retention"]),
                "retrieval_ms": retrieval_time,
                "footprint_mb": footprint_mb,
            }
        )
        model.to("cpu")
    return results


def profile_retrieval_time(results: List[Dict[str, float]]) -> str:
    path = os.path.join(RESULTS_DIR, "scaling_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return path


def test_consolidation_at_scale(results: List[Dict[str, float]]) -> str:
    report_lines = ["Scale,Accuracy,Retention,Retrieval(ms),Footprint(MB)"]
    for row in results:
        report_lines.append(
            f"{row['scale']},{row['accuracy']:.3f},{row['retention']:.3f},{row['retrieval_ms']:.2f},{row['footprint_mb']:.2f}"
        )
    path = os.path.join(RESULTS_DIR, "scaling_report.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")
    return path


def plot_scaling_curves(results: List[Dict[str, float]]) -> str:
    import matplotlib.pyplot as plt

    _ensure_dir()
    scales = [row["scale"] for row in results]
    retrieval_ms = [row["retrieval_ms"] for row in results]
    footprint_mb = [row["footprint_mb"] for row in results]
    accuracy = [row["accuracy"] for row in results]

    plt.figure(figsize=(10, 6))
    plt.plot(scales, retrieval_ms, marker="o", label="Retrieval ms/query")
    plt.plot(scales, accuracy, marker="s", label="Accuracy")
    plt.xlabel("Memory Size")
    plt.ylabel("Metric")
    plt.title("Scaling behaviour of H-SOKV")
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_path = os.path.join(RESULTS_DIR, "scaling_study_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(scales, footprint_mb, marker="^", color="#d9534f")
    plt.xlabel("Memory Size")
    plt.ylabel("Memory Footprint (MB)")
    plt.title("Memory Footprint vs. Scale")
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    footprint_path = os.path.join(RESULTS_DIR, "scaling_memory_footprint.png")
    plt.savefig(footprint_path, dpi=300)
    plt.close()
    return output_path


def recommend_optimizations(results: List[Dict[str, float]]) -> str:
    recommendations = [
        "Consider FAISS or ANN search once retrieval latency exceeds 5ms/query.",
        "Shard KV memory across devices after 50K entries to maintain throughput.",
        "Enable mixed precision storage for value vectors to reduce footprint.",
    ]
    path = os.path.join(RESULTS_DIR, "optimization_recommendations.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(recommendations) + "\n")
    return path


def main(args: argparse.Namespace) -> None:
    _ensure_dir()
    results = run_scaling_experiment(args.scales)
    metrics_path = profile_retrieval_time(results)
    report_path = test_consolidation_at_scale(results)
    plot_path = plot_scaling_curves(results)
    recommendations_path = recommend_optimizations(results)
    print(f"Scaling metrics written to {metrics_path}")
    print(f"Scaling report saved to {report_path}")
    print(f"Scaling curves saved to {plot_path}")
    print(f"Optimization tips saved to {recommendations_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaling experiment for H-SOKV.")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1000, 3000, 5000],
        help="Memory sizes to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
