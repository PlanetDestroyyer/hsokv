"""Surprise-based selective memory writing utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class SurpriseBasedWriter:
    """Filter KV writes using prediction surprise and contextual novelty."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.use_surprise_writing = bool(config.get("use_surprise_writing", True))
        self.surprise_threshold = float(config.get("surprise_threshold", 0.5))
        self.novelty_threshold = float(config.get("novelty_threshold", 0.7))
        self.min_confidence = float(config.get("surprise_min_confidence", 0.05))
        self.metrics = {
            "surprise_scores": [],  # type: List[float]
            "novelty_scores": [],  # type: List[float]
            "write_count": 0,
            "skip_count": 0,
        }
        self.write_step = 0

    def reset_metrics(self) -> None:
        self.metrics["surprise_scores"].clear()
        self.metrics["novelty_scores"].clear()
        self.metrics["write_count"] = 0
        self.metrics["skip_count"] = 0

    def compute_prediction_error(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return per-sample cross entropy surprise scores."""
        if logits.size(0) != labels.size(0):
            raise ValueError("logits and labels must share the batch dimension")
        surprise = F.cross_entropy(logits.detach(), labels.detach(), reduction="none")
        return surprise

    def compute_novelty(self, pooled: torch.Tensor, memory) -> torch.Tensor:
        """Estimate novelty as 1 - max cosine similarity vs. existing memory keys."""
        if not len(memory):
            return torch.ones(pooled.size(0), device=pooled.device)
        normalized = F.normalize(pooled.detach(), p=2, dim=-1)
        keys = memory.keys
        if keys.device != normalized.device:
            keys = keys.to(normalized.device)
        similarities = torch.matmul(normalized, keys.t()).clamp(min=-1.0, max=1.0)
        max_sim = similarities.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(min=0.0, max=1.0)
        return novelty

    def should_write(self, surprise: torch.Tensor, novelty: torch.Tensor) -> torch.Tensor:
        if surprise.size() != novelty.size():
            raise ValueError("surprise and novelty tensors must match in shape")
        write_mask = (surprise > self.surprise_threshold) | (novelty > self.novelty_threshold)
        return write_mask

    def _update_metrics(self, surprise: Iterable[float], novelty: Iterable[float], writes: int, skips: int) -> None:
        self.metrics["surprise_scores"].extend(float(s) for s in surprise)
        self.metrics["novelty_scores"].extend(float(n) for n in novelty)
        self.metrics["write_count"] += int(writes)
        self.metrics["skip_count"] += int(skips)

    def selective_write(
        self,
        *,
        model,
        memory,
        batch: Dict[str, object],
        pooled: torch.Tensor,
        logits: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        """Main entry point: perform surprise-based filtered writes."""
        if not self.use_surprise_writing:
            # Fall back to legacy always-write behaviour.
            return self._legacy_write(model=model, memory=memory, batch=batch, pooled=pooled, cache=cache)

        if pooled.size(0) != logits.size(0):
            raise ValueError("pooled and logits must share the batch dimension")

        labels: torch.Tensor = batch["labels"]
        surprise_scores = self.compute_prediction_error(logits, labels)
        novelty_scores = self.compute_novelty(pooled, memory)
        write_mask = self.should_write(surprise_scores, novelty_scores)

        writes = 0
        skips = 0
        seen_in_batch = set()
        existing_words = {
            entry.get("word")
            for entry in getattr(memory, "values", [])
            if isinstance(entry, dict) and entry.get("word")
        }
        surprise_list: List[float] = surprise_scores.detach().cpu().tolist()
        novelty_list: List[float] = novelty_scores.detach().cpu().tolist()

        for idx, should_store in enumerate(write_mask):
            rare_word = batch["rare_words"][idx]
            definition = batch["definitions"][idx]
            usage = batch["usages"][idx]
            if not rare_word or rare_word in seen_in_batch:
                skips += 1
                continue
            seen_in_batch.add(rare_word)

            story_hash = hash((rare_word, definition, usage))
            if any(meta.get("story_hash") == story_hash for meta in memory.metadata):
                skips += 1
                continue

            should_store_flag = bool(should_store.item() if torch.is_tensor(should_store) else should_store)
            if rare_word not in existing_words:
                should_store_flag = True

            if not should_store_flag:
                skips += 1
                continue

            value_vector = self._get_value_vector(model, cache, rare_word, definition, usage)
            confidence = float(torch.clamp(1.0 - surprise_scores[idx], self.min_confidence, 1.0).item())
            metadata = {
                "confidence": confidence,
                "retrieval_count": 0,
                "success_rate": 0.0,
                "story_hash": story_hash,
                "created_at": float(self.write_step),
                "domain": "general",
                "emotion": 0.5,
            }
            try:
                memory.write(
                    key_embedding=pooled[idx],
                    value_dict={
                        "word": rare_word,
                        "definition": definition,
                        "usage": usage,
                        "value_vector": value_vector,
                    },
                    metadata=metadata,
                )
                writes += 1
                existing_words.add(rare_word)
            except Exception as exc:
                skips += 1
                LOGGER.warning("Selective write failed for '%s': %s", rare_word, exc)

        self.write_step += 1
        self._update_metrics(surprise_list, novelty_list, writes, skips)
        return {
            "write_mask": write_mask.detach().cpu(),
            "surprise": surprise_scores.detach().cpu(),
            "novelty": novelty_scores.detach().cpu(),
            "writes": writes,
            "skips": skips,
        }

    def _legacy_write(
        self,
        *,
        model,
        memory,
        batch: Dict[str, object],
        pooled: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        writes = 0
        seen_in_batch = set()
        for vector, definition, usage, rare_word in zip(
            pooled,
            batch["definitions"],
            batch["usages"],
            batch["rare_words"],
        ):
            if not rare_word or rare_word in seen_in_batch:
                continue
            seen_in_batch.add(rare_word)
            story_hash = hash((rare_word, definition, usage))
            if any(meta.get("story_hash") == story_hash for meta in memory.metadata):
                continue
            value_vector = self._get_value_vector(model, cache, rare_word, definition, usage)
            metadata = {
                "confidence": 0.25,
                "retrieval_count": 0,
                "success_rate": 0.0,
                "story_hash": story_hash,
            }
            memory.write(
                key_embedding=vector,
                value_dict={"word": rare_word, "definition": definition, "usage": usage, "value_vector": value_vector},
                metadata=metadata,
            )
            writes += 1
        return {"write_mask": None, "surprise": None, "novelty": None, "writes": writes, "skips": 0}

    def _get_value_vector(
        self,
        model,
        cache: Dict[str, torch.Tensor],
        rare_word: str,
        definition: str,
        usage: str,
    ) -> torch.Tensor:
        """Cache-accelerated encoder for value vectors."""
        cache_key = rare_word
        if cache_key in cache:
            return cache[cache_key]
        encoded = model.encode_text(f"{definition} {usage}".strip(), self.config["definition_max_length"]).detach()
        cache[cache_key] = encoded
        return encoded


__all__ = ["SurpriseBasedWriter"]
