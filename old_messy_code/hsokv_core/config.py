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
    "kv_confidence_threshold": 0.15,
    "max_memory_entries": 400,
    "memory_cap": 1000,
    "use_consolidation": True,  # Re-enabled: needed for memory lifecycle management
    "use_forgetting": True,  # Re-enabled: needed to prevent memory overflow
    "forgetting_interval": 10,
    "forgetting_similarity_threshold": 0.8,
    "forgetting_utility_threshold": 0.10,  # FIXED: Lowered from 0.25 to preserve memories longer during extended training
    "results_dir": "results",
    "retention_distractor_factor": 5,
    "baseline_epochs": 6,
    "baseline_lr": 2e-4,
    "baseline_kv_steps": 150,
    "use_kv": True,
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
    "hf_model_name": "hsokv-base",
    "flops_target": 1e9,
    "preset": "demo",
    "lm_corpus_preset": "medium",
    "lm_min_samples": 50,
    # 3-STAGE MEMORY LIFECYCLE (Human-inspired learning: "overwhelming" example)
    "memory_learning_phase_duration": 5,  # STAGE 1: First 5 retrievals - pure recall, maximum protection
    "memory_reinforcement_phase_duration": 20,  # STAGE 2: Next 15 retrievals - boosted, high protection
    "min_uses_before_consolidation": 5,  # Require 5 successful uses before permanent storage
    "min_success_rate_for_consolidation": 0.8,  # Require 80% correct usage
    "consolidation_confidence_threshold": 0.85,  # Require 85% confidence
    "protect_during_learning": True,  # Never delete memories in LEARNING stage
    "protect_during_reinforcement": True,  # Never delete memories in REINFORCEMENT stage
    "use_pure_recall_for_new_words": True,  # No averaging during learning - return exact best match
    "use_stage_aware_retrieval": True,  # Enable 3-stage retrieval strategy
}

PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "quick_test": {
        "meta_iterations": 2,
        "batch_size": 16,
        "lm_seq_length": 32,
        "lm_stride": 16,
        "flops_target": 5e6,
    },
    "demo": {
        "meta_iterations": 5,
        "batch_size": 16,
        "lm_seq_length": 48,
        "lm_stride": 24,
    },
    "research": {
        "meta_iterations": 10,
        "batch_size": 32,
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
