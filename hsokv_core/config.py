"""Configuration dictionary and ablation flags for H-SOKV."""

from typing import Dict, List

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


CONFIG: Dict[str, object] = {
    "seed": 42,
    "device": _default_device(),
    "pin_memory": True if torch.cuda.is_available() else False,
    "num_workers": 2 if torch.cuda.is_available() else 0,
    "allow_dataset_download": False,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_length": 96,
    "definition_max_length": 48,
    "batch_size": 8,
    "meta_iterations": 10,
    "agent_steps": 50,
    "num_managers": 2,
    "agents_per_manager": 5,
    "kv_top_k_range": (1, 10),
    "learning_rate_range": (1e-5, 1e-3),
    "kv_confidence_threshold": 0.15,
    "max_memory_entries": 400,
    "memory_cap": 1000,
    "results_dir": "results",
    "retention_distractor_factor": 5,
    "baseline_epochs": 6,
    "baseline_lr": 2e-4,
    "baseline_kv_steps": 150,
    "use_swarm": True,
    "use_kv": True,
    "ablate": ["full", "kv_only", "swarm_only", "neither"],  # stage 1 scaffolding
    "benchmark": "synthetic",
    "glue_data_dir": "data/glue",
    "glue_task": "sst2",
    "glue_shots_per_class": 16,
    "glue_val_samples": 64,
    "glue_test_samples": 128,
    "glue_retention_samples": 64,
    "glue_max_train_examples": 50000,
    "cifar_data_dir": "data/cifar",
    "cifar_split_scheme": "pair",
    "cifar_max_train_per_class": 200,
    "cifar_max_test_per_class": 200,
    "cifar_val_ratio": 0.1,
    "cifar_retention_per_class": 80,
    "cifar_token_grid": 4,
    "cifar_token_bins": 16,
    "task_type": "classification",
    "lm_corpus_path": "data/lm/sample.txt",
    "lm_seq_length": 96,
    "lm_stride": 32,
    "lm_max_sequences": 20000,
    "lm_train_split": 0.8,
    "lm_val_split": 0.1,
    "distributed_node_counts": [1, 2, 4],
    "distributed_agents_per_node": 5,
    "distributed_episodes": 24,
    "distributed_steps": 40,
    "distributed_gamma": 0.95,
    "distributed_learning_rate": 0.05,
    "distributed_backend": "auto",
    "num_nodes": 1,
    "hf_model_name": "hsokv-base",
    "flops_target": 1e9,
    "preset": "demo",
    "lm_corpus_preset": "medium",
    "lm_min_samples": 50,
    "use_context_retrieval": True,
    "context_recency_decay": 0.95,
    "context_domain_boost": 1.5,
    "context_emotion_scale": 0.3,
    "context_importance_scale": 0.5,
    "context_domains": ["general", "medical", "legal", "finance", "technology", "culinary"],
}

PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "quick_test": {
        "meta_iterations": 2,
        "agents_per_manager": 1,
        "agent_steps": 10,
        "batch_size": 16,
        "num_managers": 1,
        "lm_seq_length": 32,
        "lm_stride": 16,
        "flops_target": 5e6,
    },
    "demo": {
        "meta_iterations": 5,
        "agents_per_manager": 2,
        "agent_steps": 25,
        "batch_size": 16,
        "num_managers": 2,
        "lm_seq_length": 48,
        "lm_stride": 24,
    },
    "research": {
        "meta_iterations": 10,
        "agents_per_manager": 5,
        "agent_steps": 50,
        "batch_size": 32,
        "num_managers": 2,
        "lm_seq_length": 64,
        "lm_stride": 32,
    },
}


def override_config(config: Dict[str, object], overrides: Dict[str, object]) -> Dict[str, object]:
    """Return a copy of config with overrides applied."""
    merged = dict(config)
    merged.update(overrides)
    return merged


def relevant_ablation_variants(config: Dict[str, object]) -> List[str]:
    ablate = config.get("ablate", [])
    if isinstance(ablate, str):
        return [ablate]
    return list(ablate)
