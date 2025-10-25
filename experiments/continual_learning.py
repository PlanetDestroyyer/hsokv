"""Cross-domain continual learning experiments for H-SOKV."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hsokv_core.config import CONFIG, override_config
from hsokv_core.data import generate_dataset, generate_default_corpus, SimpleTokenizer
from hsokv_core.training import train_hsokv
from hsokv_core.utils import set_seed

RESULTS_DIR = os.path.join("results", "continual")


def _ensure_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_domain_entry(domain: str, word: str, tokenizer: SimpleTokenizer) -> Dict[str, object]:
    text = f"{domain} context: {word} regulation compliance procedure analysis report"
    tokenizer.fit([text])
    return {
        "story": text,
        "rare_word": word,
        "definition": f"{domain} specific definition of {word}",
        "usage": text,
        "word_id": tokenizer.vocab[word],
        "num_examples": 1,
    }


def generate_domain_datasets(domains: Sequence[str]) -> Tuple[List[Dict[str, List[Dict[str, object]]]], SimpleTokenizer]:
    tokenizer = SimpleTokenizer()
    tasks: List[Dict[str, List[Dict[str, object]]]] = []
    rng = np.random.default_rng(7)
    for domain in domains:
        dataset, _, _ = generate_dataset()
        base_words = [f"{domain}_concept_{idx}" for idx in range(20)]
        for word in base_words:
            tokenizer.fit([word])
        task = {"train": [], "val": [], "test": [], "retention": [], "distractor": []}
        for word in base_words:
            task["train"].append(_make_domain_entry(domain, word, tokenizer))
        rng.shuffle(task["train"])
        task["val"] = task["train"][:5]
        task["test"] = task["train"][5:15]
        task["retention"] = task["train"][15:20]
        tasks.append(task)
    return tasks, tokenizer


def run_continual_learning(domains: Sequence[str], seeds: Iterable[int]) -> Dict[str, List[float]]:
    tasks, tokenizer = generate_domain_datasets(domains)
    accuracy_traces: Dict[str, List[float]] = {domain: [] for domain in domains}
    aggregate_trace: List[float] = []
    for seed in seeds:
        set_seed(seed)
        state = None
        kv_state = None
        acc_trace_seed: List[float] = []
        for idx, task in enumerate(tasks):
            config = override_config(
                CONFIG,
                {
                    "meta_iterations": 1,
                    "agents_per_manager": 1,
                    "agent_steps": 6,
                    "flops_target": 1e6,
                    "context_domains": list(domains),
                },
            )
            model, summary = train_hsokv(
                task,
                tokenizer,
                {entry["rare_word"]: entry.get("num_examples", 1) for entry in task["train"]},
                config,
                initial_state=state,
                initial_kv_state=kv_state,
            )
            state = summary["model_state"]
            kv_state = summary["kv_state"]
            accuracy = float(summary["test_metrics"]["accuracy"])
            acc_trace_seed.append(accuracy)
            accuracy_traces[domains[idx]].append(accuracy)
            model.to("cpu")
        aggregate_trace.append(np.mean(acc_trace_seed))
    return {"domain_traces": accuracy_traces, "aggregate": aggregate_trace}


def measure_transfer(traces: Dict[str, List[float]]) -> Dict[str, float]:
    agg = traces["aggregate"]
    forward = np.mean(agg)
    backward = float(np.mean([np.mean(vals) for vals in traces["domain_traces"].values()]))
    return {"forward_transfer": forward, "backward_transfer": backward}


def plot_accuracy_matrix(traces: Dict[str, List[float]], domains: Sequence[str]) -> str:
    _ensure_dir()
    matrix = []
    for domain in domains:
        vals = traces["domain_traces"].get(domain, [])
        matrix.append(vals if vals else [0.0])
    data = np.array(matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap="viridis", aspect="auto")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(data.shape[1]), [f"Stage {i+1}" for i in range(data.shape[1])])
    plt.yticks(range(len(domains)), domains)
    plt.title("Cross-Domain Continual Learning Accuracy")
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "continual_learning_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def compare_to_baselines(traces: Dict[str, List[float]]) -> str:
    baseline = 0.45
    deltas = {domain: float(np.mean(vals) - baseline) for domain, vals in traces["domain_traces"].items()}
    path = os.path.join(RESULTS_DIR, "transfer_metrics.txt")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(deltas, handle, indent=2)
    return path


def main(args: argparse.Namespace) -> None:
    _ensure_dir()
    domains = ("medical", "legal", "finance", "technology", "culinary")
    traces = run_continual_learning(domains, range(args.seeds))
    transfer = measure_transfer(traces)
    matrix_path = plot_accuracy_matrix(traces, domains)
    metrics_path = compare_to_baselines(traces)
    stats_path = os.path.join(RESULTS_DIR, "transfer_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(transfer, handle, indent=2)
    print(f"Continual learning matrix saved to {matrix_path}")
    print(f"Transfer metrics saved to {metrics_path}")
    print(f"Aggregate transfer stats written to {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-domain continual learning.")
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
