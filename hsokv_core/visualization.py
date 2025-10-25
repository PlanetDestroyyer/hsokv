"""Visualization helpers for inspecting H-SOKV memory dynamics."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .memory import KeyValueMemory

VIS_RESULTS_DIR = os.path.join("results", "visualizations")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_consolidation_timeline(history: Sequence[Dict[str, float]], output_dir: Optional[str] = None) -> str:
    """Plot retention, gate entropy, and regret over time."""
    if output_dir is None:
        output_dir = VIS_RESULTS_DIR
    _ensure_dir(output_dir)
    if not history:
        raise ValueError("History is empty; cannot plot consolidation timeline.")
    iterations = [entry.get("iteration", idx) + 1 for idx, entry in enumerate(history)]
    retention = [entry.get("retention", 0.0) for entry in history]
    gate_entropy = [entry.get("gate_entropy", 0.0) for entry in history]
    regret = [entry.get("regret", 0.0) for entry in history]

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, retention, label="Retention", marker="o")
    plt.plot(iterations, gate_entropy, label="Gate Entropy", marker="s")
    plt.plot(iterations, regret, label="Regret", marker="^")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Consolidation Timeline")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, "consolidation_timeline.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _build_memory_statistics(memory: KeyValueMemory) -> Dict[str, float]:
    confidences = [float(meta.get("confidence", 0.0)) for meta in memory.metadata]
    success = [float(meta.get("success_rate", 0.0)) for meta in memory.metadata]
    retrievals = [int(meta.get("retrieval_count", 0)) for meta in memory.metadata]
    emotions = [float(meta.get("emotion", 0.5)) for meta in memory.metadata]
    return {
        "count": len(memory),
        "confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
        "confidence_std": float(np.std(confidences)) if confidences else 0.0,
        "success_mean": float(np.mean(success)) if success else 0.0,
        "retrieval_mean": float(np.mean(retrievals)) if retrievals else 0.0,
        "emotion_mean": float(np.mean(emotions)) if emotions else 0.5,
    }


def plot_memory_statistics(memory: KeyValueMemory, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Create histogram and scatter plots for memory metadata."""
    if output_dir is None:
        output_dir = VIS_RESULTS_DIR
    _ensure_dir(output_dir)
    paths: Dict[str, str] = {}
    if len(memory) == 0:
        raise ValueError("Memory is empty; cannot plot statistics.")
    confidences = [float(meta.get("confidence", 0.0)) for meta in memory.metadata]
    success = [float(meta.get("success_rate", 0.0)) for meta in memory.metadata]
    retrievals = [int(meta.get("retrieval_count", 0)) for meta in memory.metadata]

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=15, color="#4f81bd", alpha=0.8)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Memory Confidence Distribution")
    plt.grid(True, linestyle="--", alpha=0.4)
    hist_path = os.path.join(output_dir, "memory_confidence_hist.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    paths["confidence_hist"] = hist_path

    plt.figure(figsize=(8, 5))
    plt.scatter(retrievals, confidences, c=success, cmap="viridis", edgecolor="k")
    plt.xlabel("Retrieval Count")
    plt.ylabel("Confidence")
    plt.title("Confidence vs Retrievals (color = success rate)")
    plt.colorbar(label="Success Rate")
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "memory_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    paths["confidence_scatter"] = scatter_path

    stats = _build_memory_statistics(memory)
    stats_path = os.path.join(output_dir, "memory_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    paths["stats"] = stats_path
    return paths


def generate_report(
    history: Sequence[Dict[str, float]],
    memory: KeyValueMemory,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate a consolidated visualization report."""
    if output_dir is None:
        output_dir = VIS_RESULTS_DIR
    _ensure_dir(output_dir)
    report: Dict[str, str] = {}
    try:
        report["timeline"] = plot_consolidation_timeline(history, output_dir)
    except ValueError:
        report["timeline"] = ""
    try:
        stats_paths = plot_memory_statistics(memory, output_dir)
        report.update(stats_paths)
    except ValueError:
        report["confidence_hist"] = ""
        report["confidence_scatter"] = ""
    return report


__all__ = ["plot_consolidation_timeline", "plot_memory_statistics", "generate_report"]
