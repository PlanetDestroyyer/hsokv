"""Key-Value memory with normalized embeddings and efficient pruning."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import CONFIG

LOGGER = logging.getLogger(__name__)


class KeyValueMemory:
    def __init__(self, key_dim: int, device: torch.device) -> None:
        self.device = device
        self.key_dim = key_dim
        self.keys = torch.empty(0, key_dim, device=self.device)
        self.values: List[Dict[str, object]] = []
        self.metadata: List[Dict[str, object]] = []

    def __len__(self) -> int:
        return self.keys.size(0)

    def get_memory_stage(self, entry_id: int) -> str:
        """
        Determine which learning stage this memory is in.

        Based on human learning (e.g., learning "overwhelming" from a movie):
        - LEARNING: First 5 uses - pure recall, maximum protection
        - REINFORCEMENT: Next 15 uses - boosted confidence, high protection
        - MATURE: After 20 uses - standard retrieval, can be consolidated/forgotten

        Args:
            entry_id: Index of memory entry

        Returns:
            "LEARNING" | "REINFORCEMENT" | "MATURE"
        """
        if entry_id < 0 or entry_id >= len(self.metadata):
            return "MATURE"

        meta = self.metadata[entry_id]

        # Non-first-exposure memories are always MATURE
        if not meta.get("is_first_exposure", False):
            return "MATURE"

        retrieval_count = meta.get("retrieval_count", 0)

        # STAGE 1: LEARNING (like Day 0-1 with "overwhelming")
        # Pure recall, no averaging, maximum protection
        if retrieval_count < CONFIG.get("memory_learning_phase_duration", 5):
            return "LEARNING"

        # STAGE 2: REINFORCEMENT (like Days 2-14 with "overwhelming")
        # Boosted confidence, high protection, gradual blending
        elif retrieval_count < CONFIG.get("memory_reinforcement_phase_duration", 20):
            return "REINFORCEMENT"

        # STAGE 3: MATURE (like Week 3+ with "overwhelming")
        # Standard retrieval, can be consolidated or forgotten if unused
        else:
            return "MATURE"

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """L2 normalization with numerical stability across hardware."""
        # FIXED: Add epsilon for stability across different GPUs
        norm = torch.sqrt(torch.sum(tensor ** 2, dim=-1, keepdim=True) + 1e-12)
        return tensor / norm

    def write(self, key_embedding: torch.Tensor, value_dict: Dict[str, object], metadata: Dict[str, object]) -> int:
        key_embed = self._normalize(key_embedding.detach().to(self.device))
        if self.keys.numel() == 0:
            self.keys = key_embed.unsqueeze(0)
        else:
            self.keys = torch.cat([self.keys, key_embed.unsqueeze(0)], dim=0)
        value_vector = self._normalize(value_dict["value_vector"].detach().to(self.device))
        stored_value = {
            "word": value_dict["word"],
            "definition": value_dict["definition"],
            "usage": value_dict["usage"],
            "value_vector": value_vector,
        }
        self.values.append(stored_value)
        default_created = float(metadata.get("created_at", len(self.metadata)))
        meta = {
            "confidence": metadata.get("confidence", 0.2),
            "retrieval_count": metadata.get("retrieval_count", 0),
            "success_rate": metadata.get("success_rate", 0.0),
            "story_hash": metadata.get("story_hash"),
            "created_at": default_created,
            "domain": metadata.get("domain", "general"),
            "emotion": float(metadata.get("emotion", 0.5)),
        }
        self.metadata.append(meta)
        if len(self.values) > CONFIG.get("memory_cap", 1000):
            self._enforce_memory_cap()
        return len(self.values) - 1

    def _enforce_memory_cap(self) -> None:
        confidences = torch.tensor([m["confidence"] for m in self.metadata], device=self.device, dtype=torch.float32)
        keep = torch.topk(confidences, k=min(confidences.numel(), CONFIG.get("memory_cap", 1000)), sorted=False)
        indices = keep.indices
        self.keys = self.keys[indices]
        keep_list = indices.detach().cpu().tolist()
        self.values = [self.values[i] for i in keep_list]
        self.metadata = [self.metadata[i] for i in keep_list]

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        context_modulator=None,
        context_signals=None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if self.keys.numel() == 0:
            default = torch.zeros_like(query_embedding)
            details = {"avg_hits": 0.0, "topk_indices": [[] for _ in range(query_embedding.shape[0] if query_embedding.dim() > 1 else 1)], "avg_similarity": 0.0}
            return default, details

        # Get query device for DataParallel compatibility
        # Each GPU replica will process queries on its own device
        query_device = query_embedding.device

        single = False
        if query_embedding.dim() == 1:
            query = query_embedding.unsqueeze(0)
            single = True
        else:
            query = query_embedding

        # Normalize query on its original device
        query = self._normalize(query)

        # Move keys to query device instead of moving query to keys device
        # This allows DataParallel replicas on different GPUs to work independently
        keys = self.keys.to(query_device)

        similarities = torch.clamp(F.linear(query, keys), min=0.0)
        if context_modulator is not None:
            try:
                similarities = context_modulator.compute_context_modulated_similarity(
                    similarities, self.metadata, context_signals
                )
            except Exception as exc:
                LOGGER.warning("Context modulation failed: %s", exc)
        min_sim = similarities.min().item()
        max_sim = similarities.max().item()
        mean_sim = similarities.mean().item()
        # print(f"[KV] Sim range: min={min_sim:.4f}, max={max_sim:.4f}, mean={mean_sim:.4f}")
        k = min(top_k, keys.size(0))
        topk = similarities.topk(k, dim=-1)
        outputs = []
        hit_counts = []
        sim_scores = []
        topk_indices: List[List[int]] = []

        # Check if stage-aware retrieval is enabled
        use_stage_aware = CONFIG.get("use_stage_aware_retrieval", True)
        use_pure_recall = CONFIG.get("use_pure_recall_for_new_words", True)

        for i in range(query.size(0)):
            indices = topk.indices[i]
            sims = topk.values[i]

            # STAGE-AWARE RETRIEVAL: Check if best match is in LEARNING stage
            best_idx = indices[0].item()
            best_stage = self.get_memory_stage(best_idx) if use_stage_aware else "MATURE"

            # LEARNING STAGE: Pure recall (like recalling "overwhelming" on Day 1)
            # Return ONLY the best match, no averaging
            if best_stage == "LEARNING" and use_pure_recall:
                best_meta = self.metadata[best_idx]
                best_value = self.values[best_idx]["value_vector"]
                best_sim = float(sims[0].item())

                # Update retrieval count
                self.metadata[best_idx]["retrieval_count"] += 1

                # Check if should graduate from LEARNING stage
                if self.metadata[best_idx]["retrieval_count"] >= CONFIG.get("memory_learning_phase_duration", 5):
                    LOGGER.info(f"Memory '{self.values[best_idx]['word']}' graduated from LEARNING to REINFORCEMENT stage")

                # Return pure best match
                outputs.append(best_value)
                hit_counts.append(1)
                sim_scores.append(best_sim)
                topk_indices.append([best_idx])
                continue  # Skip averaging logic

            # REINFORCEMENT or MATURE STAGE: Use weighted averaging
            weights: List[float] = []
            vectors: List[torch.Tensor] = []
            for j, idx in enumerate(indices):
                entry_id = idx.item()
                metadata = self.metadata[entry_id]
                confidence = float(metadata["confidence"])
                similarity = float(sims[j].item())

                # STAGE-AWARE CONFIDENCE BOOSTING
                # Get current stage for this memory
                current_stage = self.get_memory_stage(entry_id)
                retrieval_count = metadata.get("retrieval_count", 0)

                if current_stage == "REINFORCEMENT":
                    # REINFORCEMENT STAGE (like Days 2-14 with "overwhelming")
                    # Strong boost that gradually decays
                    reinforcement_duration = CONFIG.get("memory_reinforcement_phase_duration", 20)
                    learning_duration = CONFIG.get("memory_learning_phase_duration", 5)
                    reinforcement_progress = retrieval_count - learning_duration
                    reinforcement_window = reinforcement_duration - learning_duration
                    confidence_boost = 1.5 - (0.025 * reinforcement_progress)
                    effective_confidence = min(confidence * confidence_boost, 1.0)
                elif current_stage == "LEARNING":
                    # LEARNING STAGE - should not reach here due to pure recall above
                    # But if it does, apply maximum boost
                    effective_confidence = min(confidence * 1.5, 1.0)
                else:
                    # MATURE STAGE - standard confidence
                    effective_confidence = confidence

                # Skip very low similarity matches
                if similarity < 0.3:
                    continue

                weight = max(effective_confidence, 1e-4) * similarity
                weights.append(weight)
                vectors.append(self.values[entry_id]["value_vector"])

                # Update retrieval count
                self.metadata[entry_id]["retrieval_count"] += 1

                # Graduate from first_exposure after completing REINFORCEMENT stage
                stage_after_update = self.get_memory_stage(entry_id)
                if stage_after_update == "MATURE" and metadata.get("is_first_exposure", False):
                    self.metadata[entry_id]["is_first_exposure"] = False
                    LOGGER.info(f"Memory '{self.values[entry_id]['word']}' graduated to MATURE stage")
            if not weights or sum(weights) == 0:
                weights = [1.0]
                vectors = [torch.zeros_like(keys[0])]
            # Use query_device for DataParallel compatibility
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=query_device)
            stacked_vectors = torch.stack(vectors)
            aggregated = (weight_tensor.unsqueeze(-1) * stacked_vectors).sum(dim=0) / (weight_tensor.sum() + 1e-8)
            outputs.append(aggregated)
            hit_counts.append(len(indices))
            sim_scores.append(float(sims.mean().item()) if sims.numel() > 0 else 0.0)
            topk_indices.append(indices.tolist())
        result = torch.stack(outputs)
        if single:
            result = result.squeeze(0)
        details = {
            "avg_hits": float(np.mean(hit_counts)) if hit_counts else 0.0,
            "topk_indices": topk_indices,
            "avg_similarity": float(np.mean(sim_scores)) if sim_scores else 0.0,
        }
        result = result.detach()
        if context_signals is not None:
            details["context"] = context_signals
        return result, details

    def update_confidence(self, entry_id: int, success_signal: float) -> None:
        if entry_id < 0 or entry_id >= len(self.metadata):
            return
        meta = self.metadata[entry_id]
        count = meta["retrieval_count"]
        meta["success_rate"] = (meta["success_rate"] * count + success_signal) / (count + 1)
        meta["confidence"] = float(np.clip(meta["confidence"] + 0.1 * (success_signal - 0.5), 0.05, 1.0))

    def prune(self, threshold: float) -> None:
        if not self.metadata:
            return
        confidences = torch.tensor([m["confidence"] for m in self.metadata], device=self.device, dtype=torch.float32)
        mask = confidences >= threshold
        if mask.all():
            return
        keep_indices = mask.nonzero(as_tuple=True)[0]
        if len(keep_indices) == 0:
            self.keys = torch.empty(0, self.key_dim, device=self.device)
            self.values = []
            self.metadata = []
        else:
            self.keys = self.keys[keep_indices]
            keep_list = keep_indices.detach().cpu().tolist()
            self.values = [self.values[i] for i in keep_list]
            self.metadata = [self.metadata[i] for i in keep_list]

    def get_state(self) -> Dict[str, object]:
        return {
            "keys": self.keys.detach().cpu(),
            "values": [
                {
                    "word": val["word"],
                    "definition": val["definition"],
                    "usage": val["usage"],
                    "value_vector": val["value_vector"].detach().cpu(),
                }
                for val in self.values
            ],
            "metadata": [dict(meta) for meta in self.metadata],
        }

    def load_state(self, state: Dict[str, object]) -> None:
        self.keys = state["keys"].to(self.device)
        self.values = [
            {
                "word": val["word"],
                "definition": val["definition"],
                "usage": val["usage"],
                "value_vector": val["value_vector"].to(self.device),
            }
            for val in state["values"]
        ]
        self.metadata = [dict(meta) for meta in state["metadata"]]

    def remove_indices(self, indices: List[int]) -> None:
        """Remove memories at the specified indices."""
        if not indices:
            return
        unique_indices = sorted(set(i for i in indices if 0 <= i < len(self.values)))
        if not unique_indices:
            return
        keep_mask = torch.ones(len(self.values), dtype=torch.bool, device=self.device)
        keep_mask[unique_indices] = False
        if keep_mask.any():
            self.keys = self.keys[keep_mask]
        else:
            self.keys = torch.empty(0, self.key_dim, device=self.device)
        keep_list = [i for i, flag in enumerate(keep_mask.detach().cpu().tolist()) if flag]
        self.values = [self.values[i] for i in keep_list]
        self.metadata = [self.metadata[i] for i in keep_list]
