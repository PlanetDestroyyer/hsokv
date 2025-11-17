"""Memory consolidation module transferring stable KV entries into model weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .memory import KeyValueMemory

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationMetrics:
    consolidated_count: int
    memory_freed: int
    avg_loss: float


class _ConsolidationDataset(Dataset):
    """Lightweight dataset wrapping synthetic consolidation samples."""

    def __init__(self, samples: Sequence[Tuple[torch.Tensor, torch.Tensor, int]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, label = self.samples[idx]
        return {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ConsolidationModule:
    """Implements memory consolidation for the TransformerWithKV model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, object],
        tokenizer,
        label_names: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
        max_steps: int = 50,
        learning_rate: float = 1e-5,
    ) -> None:
        self.model = model
        self.memory: KeyValueMemory = getattr(model, "kv_memory")
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or torch.device(config.get("device", "cpu"))
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.max_seq_length = int(config.get("max_seq_length", 96))
        self.label_names = list(label_names) if label_names else None
        self._label_index = (
            {label: idx for idx, label in enumerate(self.label_names)} if self.label_names else {}
        )

    # ----------------------------
    # Candidate Identification
    # ----------------------------
    def identify_candidates(self) -> List[int]:
        """
        Return indices of KV entries that qualify for consolidation.

        Respects 3-stage memory lifecycle (human-inspired "overwhelming" example):
        - LEARNING: NEVER consolidate (still learning, maximum protection)
        - REINFORCEMENT: NEVER consolidate (reinforcing through use, high protection)
        - MATURE: Can consolidate if proven stable through repeated successful use
        """
        indices: List[int] = []

        # Get thresholds from config (human-inspired requirements)
        threshold_conf = float(self.config.get("consolidation_confidence_threshold", 0.85))
        threshold_retrieval = int(self.config.get("min_uses_before_consolidation", 5))
        threshold_success = float(self.config.get("min_success_rate_for_consolidation", 0.8))

        skipped_learning = 0
        skipped_reinforcement = 0

        for idx, meta in enumerate(self.memory.metadata):
            confidence = float(meta.get("confidence", 0.0))
            retrieval_count = int(meta.get("retrieval_count", 0))
            success_rate = float(meta.get("success_rate", 0.0))

            # CHECK MEMORY STAGE: Only consolidate MATURE memories
            stage = self.memory.get_memory_stage(idx)

            if stage == "LEARNING":
                # NEVER consolidate during LEARNING (like Day 0-1 with "overwhelming")
                skipped_learning += 1
                logger.debug(f"Skipping LEARNING stage memory at index {idx} from consolidation")
                continue

            elif stage == "REINFORCEMENT":
                # NEVER consolidate during REINFORCEMENT (like Days 2-14 with "overwhelming")
                skipped_reinforcement += 1
                logger.debug(f"Skipping REINFORCEMENT stage memory at index {idx} from consolidation")
                continue

            # MATURE STAGE: Check if proven stable for consolidation (like Week 3+ with "overwhelming")
            if (
                confidence >= threshold_conf
                and retrieval_count >= threshold_retrieval
                and success_rate >= threshold_success
            ):
                word = self.memory.values[idx].get("word", "unknown")
                logger.info(
                    f"Consolidation candidate '{word}': stage={stage}, confidence={confidence:.2f}, "
                    f"uses={retrieval_count}, success={success_rate:.2f}"
                )
                indices.append(idx)

        if skipped_learning > 0 or skipped_reinforcement > 0:
            logger.info(
                f"Protected {skipped_learning} LEARNING + {skipped_reinforcement} REINFORCEMENT "
                f"stage memories from premature consolidation"
            )

        logger.debug("Consolidation candidates identified: %s", indices)
        return indices

    # ----------------------------
    # Dataset Construction
    # ----------------------------
    def create_consolidation_dataset(self, candidate_indices: Iterable[int]) -> _ConsolidationDataset:
        """Generate synthetic context-definition samples from consolidated memories."""
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        for idx in candidate_indices:
            try:
                value = self.memory.values[idx]
            except IndexError:
                continue
            metadata = self.memory.metadata[idx] if idx < len(self.memory.metadata) else {}
            word = value.get("word")
            usage = value.get("usage") or value.get("definition") or ""
            definition = value.get("definition") or ""
            if not usage:
                # Without usage text there is no meaningful context.
                continue
            label_id = self._infer_label_id(word, metadata)
            if label_id is None:
                logger.debug("Skipping consolidation candidate without label mapping: %s", word)
                continue
            text = f"{usage}\nDefinition: {definition}"
            token_ids = self.tokenizer.encode(text, self.max_seq_length)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            pad_id = getattr(self.tokenizer, "pad_token_id", 0)
            attention_mask = (input_ids != pad_id).long()
            samples.append((input_ids, attention_mask, label_id))
        if not samples:
            logger.debug("No consolidation samples could be created.")
        return _ConsolidationDataset(samples)

    def _infer_label_id(self, word: Optional[str], metadata: Dict[str, object]) -> Optional[int]:
        if word is None:
            return None
        if word in self._label_index:
            return self._label_index[word]
        label_id = metadata.get("label_id")
        if isinstance(label_id, int):
            return label_id
        if self.tokenizer and hasattr(self.tokenizer, "vocab"):
            return self.tokenizer.vocab.get(word)
        return None

    # ----------------------------
    # Consolidation Routine
    # ----------------------------
    def consolidate(self) -> ConsolidationMetrics:
        candidate_indices = self.identify_candidates()
        if not candidate_indices:
            logger.info("No memories qualified for consolidation.")
            return ConsolidationMetrics(0, 0, 0.0)

        dataset = self.create_consolidation_dataset(candidate_indices)
        if len(dataset) == 0:
            logger.info("No usable samples derived from candidate memories.")
            return ConsolidationMetrics(0, 0, 0.0)

        batch_size_cfg = int(self.config.get("batch_size", 8))
        batch_size = max(1, min(batch_size_cfg, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        losses: List[float] = []
        step_iter = 0
        iterator = iter(loader)
        while step_iter < self.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "labels": batch["labels"].to(self.device),
            }
            optimizer.zero_grad(set_to_none=True)
            logits, _ = self.model(batch["input_ids"], batch["attention_mask"], top_k=5)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            step_iter += 1
        self.model.eval()

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # FIXED: Validate consolidation before deleting memories
        validation_correct = 0
        validation_total = 0
        min_accuracy_threshold = 0.85  # FIXED: Raised from 0.75 to 0.85 for safer consolidation during extended training

        with torch.no_grad():
            # Test on up to 50 samples from the consolidation dataset
            test_size = min(len(dataset), 50)
            for sample_idx in range(test_size):
                sample = dataset[sample_idx]
                input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                label = sample["labels"].unsqueeze(0).to(self.device)

                logits, _ = self.model(input_ids, attention_mask, top_k=5)
                pred = logits.argmax(dim=-1)
                validation_correct += int((pred == label).sum().item())
                validation_total += 1

        validation_accuracy = validation_correct / max(validation_total, 1)

        if validation_accuracy >= min_accuracy_threshold:
            # Consolidation successful - safe to delete
            self._remove_memories(candidate_indices)
            logger.info(
                "Consolidation validated (acc=%.2f). Removed %d memories.",
                validation_accuracy,
                len(candidate_indices),
            )
            metrics = ConsolidationMetrics(
                consolidated_count=len(candidate_indices),
                memory_freed=len(candidate_indices),
                avg_loss=avg_loss,
            )
        else:
            # Consolidation FAILED - keep memories
            logger.warning(
                "Consolidation FAILED (acc=%.2f < %.2f threshold). Keeping memories in KV store.",
                validation_accuracy,
                min_accuracy_threshold,
            )
            metrics = ConsolidationMetrics(
                consolidated_count=0,  # Didn't actually consolidate
                memory_freed=0,
                avg_loss=avg_loss,
            )

        return metrics

    def _remove_memories(self, indices: Sequence[int]) -> None:
        if not indices:
            return
        self.memory.remove_indices(indices)
        # Update metadata for consolidated entries if needed
        logger.debug("Removed %d consolidated memories from KV store.", len(indices))
