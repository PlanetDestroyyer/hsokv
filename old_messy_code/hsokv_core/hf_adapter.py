"""Hugging Face compatible wrapper for H-SOKV training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .config import CONFIG, override_config
from .data import SimpleTokenizer, generate_dataset, prepare_dataloaders
from .training import train_hsokv
from .model import TransformerWithKV
from .utils import set_seed


@dataclass
class HFSwarmConfig:
    """Configuration container mirroring HF TrainingArguments essentials."""

    output_dir: str = "hsokv_checkpoint"
    seed: int = CONFIG["seed"]
    device: Optional[str] = None
    max_seq_length: int = CONFIG["max_seq_length"]
    meta_iterations: int = CONFIG["meta_iterations"]
    use_swarm: bool = True
    use_kv: bool = True
    extras: Dict[str, object] = field(default_factory=dict)


class HFSwarmTrainer:
    """Simple Trainer-like wrapper exposing train/evaluate/save APIs."""

    def __init__(
        self,
        model: Optional[TransformerWithKV] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        config: Optional[HFSwarmConfig] = None,
        dataset: Optional[Dict[str, object]] = None,
        word_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.config = config or HFSwarmConfig()
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataset
        self.word_counts = word_counts
        self.history = None
        self.training_summary: Optional[Dict[str, object]] = None

    def _ensure_resources(self) -> None:
        set_seed(self.config.seed)
        if self.tokenizer is None or self.dataset is None or self.word_counts is None:
            self.dataset, self.tokenizer, self.word_counts = generate_dataset()
        if self.model is None:
            hf_config = override_config(
                CONFIG,
                {
                    "device": self.config.device or CONFIG["device"],
                    "max_seq_length": self.config.max_seq_length,
                    "meta_iterations": self.config.meta_iterations,
                    "use_swarm": self.config.use_swarm,
                    "use_kv": self.config.use_kv,
                    **self.config.extras,
                },
            )
            device = torch.device(hf_config["device"])
            self.model = TransformerWithKV(len(self.tokenizer.vocab), len(self.tokenizer.vocab), self.tokenizer, hf_config)
            self.model.to(device)

    def train(self) -> Dict[str, object]:
        """Run swarm training loop and return summary metrics."""
        self._ensure_resources()
        hf_config = override_config(
            CONFIG,
            {
                "device": self.config.device or CONFIG["device"],
                "max_seq_length": self.config.max_seq_length,
                "meta_iterations": self.config.meta_iterations,
                "use_swarm": self.config.use_swarm,
                "use_kv": self.config.use_kv,
                **self.config.extras,
            },
        )
        self.model, summary = train_hsokv(
            self.dataset,
            self.tokenizer,
            self.word_counts,
            hf_config,
        )
        self.training_summary = summary
        self.history = summary["history"]
        return summary

    def evaluate(self) -> Dict[str, float]:
        """Return held-out metrics from the most recent training run."""
        if not self.training_summary:
            raise RuntimeError("Call train() before evaluate().")
        return self.training_summary["test_metrics"]

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Persist model/tokenizer state to disk."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model/tokenizer to save.")
        save_path = output_dir or self.config.output_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)

