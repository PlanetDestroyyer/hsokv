"""Context-aware retrieval modulator for KeyValueMemory."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


class ContextualRetrievalModule:
    """Applies contextual modulation to memory retrieval similarities."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.decay_base = float(config.get("context_recency_decay", 0.95))
        self.domain_boost = float(config.get("context_domain_boost", 1.5))
        self.emotion_scale = float(config.get("context_emotion_scale", 0.3))
        self.importance_scale = float(config.get("context_importance_scale", 0.5))
        self.available_domains: Sequence[str] = tuple(
            config.get(
                "context_domains",
                ["general", "medical", "legal", "finance", "technology", "culinary"],
            )
        )
        if not self.available_domains:
            self.available_domains = ("general",)
        self.step_counter: int = 0

    def next_step(self) -> int:
        self.step_counter += 1
        return self.step_counter

    def extract_context_signals(
        self, hidden_states: torch.Tensor, pooled_states: torch.Tensor, current_step: int
    ) -> List[Dict[str, object]]:
        """Infer coarse-grained domain and emotion from transformer activations."""
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must have shape (batch, seq, hidden)")
        if pooled_states.dim() != 2:
            raise ValueError("pooled_states must have shape (batch, hidden)")
        with torch.no_grad():
            hidden_mean = hidden_states.mean(dim=1)  # (batch, hidden)
            pooled_mean = pooled_states.mean(dim=-1)  # (batch,)
            pooled_abs = pooled_states.abs().mean(dim=-1)
        batch_size = pooled_states.size(0)
        context_batch: List[Dict[str, object]] = []
        num_domains = len(self.available_domains)
        for idx in range(batch_size):
            domain_index = int((pooled_abs[idx].item() * 1000) % num_domains)
            domain = self.available_domains[domain_index]
            # Map pooled_mean through sigmoid to get [0, 1] emotion score.
            emotion = float(torch.sigmoid(pooled_mean[idx]).item())
            context_batch.append(
                {
                    "current_step": float(current_step),
                    "domain": domain,
                    "emotion": emotion,
                    "hidden_descriptor": hidden_mean[idx].detach(),
                }
            )
        return context_batch

    def compute_context_modulated_similarity(
        self,
        similarities: torch.Tensor,
        metadata: Sequence[Dict[str, object]],
        context_batch: Optional[List[Dict[str, object]]],
    ) -> torch.Tensor:
        """Multiply base similarities by context-aware factors."""
        if context_batch is None:
            return similarities
        modulated = similarities.clone()
        memory_size = len(metadata)
        device = similarities.device
        for row, context in enumerate(context_batch):
            if row >= modulated.size(0):
                break
            factors = torch.ones(memory_size, dtype=modulated.dtype, device=device)
            current_step = float(context.get("current_step", self.step_counter))
            query_domain = context.get("domain", "general")
            query_emotion = float(context.get("emotion", 0.5))
            for col, meta in enumerate(metadata):
                created_at = float(meta.get("created_at", 0.0))
                age = max(current_step - created_at, 0.0)
                recency_factor = self.decay_base ** (age / 100.0)
                mem_domain = meta.get("domain", "general")
                domain_factor = self.domain_boost if mem_domain == query_domain else 1.0
                mem_emotion = float(meta.get("emotion", 0.5))
                emotion_alignment = 1.0 - min(abs(mem_emotion - query_emotion), 1.0)
                emotion_factor = 1.0 + self.emotion_scale * emotion_alignment
                success_rate = float(meta.get("success_rate", 0.0))
                importance_factor = 1.0 + self.importance_scale * success_rate
                factors[col] = recency_factor * domain_factor * emotion_factor * importance_factor
            modulated[row] = modulated[row] * factors
        return modulated

    def contextual_retrieve(
        self,
        similarities: torch.Tensor,
        metadata: Sequence[Dict[str, object]],
        hidden_states: torch.Tensor,
        pooled_states: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper combining extraction and modulation."""
        current_step = self.next_step()
        context = self.extract_context_signals(hidden_states, pooled_states, current_step)
        return self.compute_context_modulated_similarity(similarities, metadata, context)
