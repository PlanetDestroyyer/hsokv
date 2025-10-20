"""Key-Value memory with normalized embeddings and efficient pruning."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import CONFIG


class KeyValueMemory:
    def __init__(self, key_dim: int, device: torch.device) -> None:
        self.device = device
        self.key_dim = key_dim
        self.keys = torch.empty(0, key_dim, device=self.device)
        self.values: List[Dict[str, object]] = []
        self.metadata: List[Dict[str, object]] = []

    def __len__(self) -> int:
        return self.keys.size(0)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, p=2, dim=-1)

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
        meta = {
            "confidence": metadata.get("confidence", 0.2),
            "retrieval_count": metadata.get("retrieval_count", 0),
            "success_rate": metadata.get("success_rate", 0.0),
            "story_hash": metadata.get("story_hash"),
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

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, Dict[str, object]]:
        if self.keys.numel() == 0:
            default = torch.zeros_like(query_embedding)
            details = {"avg_hits": 0.0, "topk_indices": [[] for _ in range(query_embedding.shape[0] if query_embedding.dim() > 1 else 1)], "avg_similarity": 0.0}
            return default, details
        single = False
        if query_embedding.dim() == 1:
            query = query_embedding.unsqueeze(0)
            single = True
        else:
            query = query_embedding
        query = self._normalize(query.to(self.device))
        keys = self.keys
        similarities = torch.clamp(F.linear(query, keys), min=0.0)
        min_sim = similarities.min().item()
        max_sim = similarities.max().item()
        mean_sim = similarities.mean().item()
        print(f"[KV] Sim range: min={min_sim:.4f}, max={max_sim:.4f}, mean={mean_sim:.4f}")
        k = min(top_k, keys.size(0))
        topk = similarities.topk(k, dim=-1)
        outputs = []
        hit_counts = []
        sim_scores = []
        topk_indices: List[List[int]] = []
        for i in range(query.size(0)):
            indices = topk.indices[i]
            sims = topk.values[i]
            weights: List[float] = []
            vectors: List[torch.Tensor] = []
            for j, idx in enumerate(indices):
                entry_id = idx.item()
                weight = max(float(self.metadata[entry_id]["confidence"]), 1e-4) * float(sims[j].item())
                weights.append(weight)
                vectors.append(self.values[entry_id]["value_vector"])
                self.metadata[entry_id]["retrieval_count"] += 1
            if not weights or sum(weights) == 0:
                weights = [1.0]
                vectors = [torch.zeros_like(keys[0])]
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
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
        return result.detach(), details

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
