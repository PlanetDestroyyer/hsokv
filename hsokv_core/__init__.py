"""Core package exposing H-SOKV components for reuse across stages."""

from .ablations import run_ablation_suite, variant_flags
from .benchmarks import run_glue_benchmark, run_split_cifar_benchmark
from .config import CONFIG, PRESET_CONFIGS, override_config, relevant_ablation_variants
from .data import (
    RARE_WORD_SPECS,
    SimpleTokenizer,
    generate_dataset,
    generate_default_corpus,
    generate_language_model_dataset,
    prepare_dataloaders,
)
from .hf_adapter import HFSwarmConfig, HFSwarmTrainer
from .distributed import run_distributed_swarm
from .memory import KeyValueMemory
from .model import BaselineTransformer, TransformerWithKV
from .metrics import estimate_model_flops, latex_table_from_metrics, summarize_history
from .context_retrieval import ContextualRetrievalModule
from .consolidation import ConsolidationModule
from .forgetting import ForgettingModule
from .visualization import plot_consolidation_timeline, plot_memory_statistics, generate_report as generate_visualization_report
from .swarm import Agent, Manager, Supervisor, compute_swarm_diversity
from .training import (
    compute_convergence_step,
    compute_usage_correctness,
    evaluate_baseline_model,
    evaluate_model,
    evaluate_retention,
    in_context_learning,
    train_baseline_kv,
    train_baseline_standard,
    train_hsokv,
)
from .utils import move_batch_to_device, set_seed

__all__ = [
    "CONFIG",
    "PRESET_CONFIGS",
    "run_ablation_suite",
    "variant_flags",
    "run_glue_benchmark",
    "run_split_cifar_benchmark",
    "override_config",
    "relevant_ablation_variants",
    "ConsolidationModule",
    "ForgettingModule",
    "ContextualRetrievalModule",
    "plot_consolidation_timeline",
    "plot_memory_statistics",
    "generate_visualization_report",
    "RARE_WORD_SPECS",
    "SimpleTokenizer",
    "generate_dataset",
    "generate_default_corpus",
    "generate_language_model_dataset",
    "prepare_dataloaders",
    "run_distributed_swarm",
    "HFSwarmConfig",
    "HFSwarmTrainer",
    "KeyValueMemory",
    "BaselineTransformer",
    "TransformerWithKV",
    "estimate_model_flops",
    "summarize_history",
    "latex_table_from_metrics",
    "Agent",
    "Manager",
    "Supervisor",
    "compute_swarm_diversity",
    "compute_convergence_step",
    "compute_usage_correctness",
    "evaluate_baseline_model",
    "evaluate_model",
    "evaluate_retention",
    "in_context_learning",
    "train_baseline_kv",
    "train_baseline_standard",
    "train_hsokv",
    "move_batch_to_device",
    "set_seed",
]
