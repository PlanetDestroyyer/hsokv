"""Model architectures with optional KV integration."""

import json
import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import CONFIG, override_config
from .memory import KeyValueMemory

# Import transformers for pre-trained model support
try:
    from transformers import AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerWithKV(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, tokenizer, config: Dict[str, object]) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device_name = config["device"]
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        # Check if using pre-trained model (GPT-2, BERT, etc.)
        pretrained_model_name = config.get("pretrained_model_name", None)
        self.use_pretrained = pretrained_model_name is not None

        if self.use_pretrained:
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers library required for pre-trained models. Install with: pip install transformers")

            print(f"[Pre-trained Model] Loading {pretrained_model_name}...")
            self.pretrained_encoder = AutoModel.from_pretrained(pretrained_model_name)

            # Get hidden size from pre-trained model
            if hasattr(self.pretrained_encoder.config, 'hidden_size'):
                hidden_dim = self.pretrained_encoder.config.hidden_size
            elif hasattr(self.pretrained_encoder.config, 'n_embd'):  # GPT-2 uses n_embd
                hidden_dim = self.pretrained_encoder.config.n_embd
            else:
                hidden_dim = config["d_model"]

            config["d_model"] = hidden_dim  # Update config to match pre-trained model
            print(f"[Pre-trained Model] Using hidden dimension: {hidden_dim}")

            # No need for custom embedding/transformer
            self.embedding = None
            self.pos_encoder = None
            self.transformer = None
        else:
            # Use custom Transformer (original behavior)
            self.pretrained_encoder = None
            self.embedding = nn.Embedding(vocab_size, config["d_model"])
            self.pos_encoder = PositionalEncoding(config["d_model"], dropout=config["dropout"], max_len=config["max_seq_length"])
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=config["d_model"],
                nhead=config["nhead"],
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=config["num_layers"])

        self.layer_norm = nn.LayerNorm(config["d_model"])
        self.gate_network = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.ReLU(),
            nn.Linear(config["d_model"], 1),
        )
        self.classifier = nn.Linear(config["d_model"], num_labels)
        self.kv_memory = KeyValueMemory(config["d_model"], torch.device(self.device_name))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, Dict[str, object]]:
        if self.use_pretrained:
            # Use pre-trained model (GPT-2, BERT, etc.)
            outputs = self.pretrained_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Get last hidden states
            hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            hidden = self.layer_norm(hidden)
            # Pool: mean of all tokens (weighted by attention mask)
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            # Use custom Transformer (original behavior)
            # FIXED: Ensure float32 precision for hardware reproducibility
            embeddings = self.embedding(input_ids) * math.sqrt(float(self.config["d_model"]))
            embeddings = embeddings.float()  # Force float32
            embeddings = self.pos_encoder(embeddings)
            hidden = self.transformer(embeddings)
            hidden = self.layer_norm(hidden)
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        retrieved, kv_details = (
            torch.zeros_like(pooled),
            {"avg_hits": 0.0, "topk_indices": [], "avg_similarity": 0.0},
        )
        if self.config.get("use_kv", True):
            # Simplified retrieval (no context modulation - removed noisy context signals)
            retrieved, kv_details = self.kv_memory.retrieve(
                pooled.detach(),
                top_k=top_k,
                context_modulator=None,
                context_signals=None,
            )
            if retrieved.dim() == 1:
                retrieved = retrieved.unsqueeze(0)
        gate = torch.sigmoid(self.gate_network(pooled))
        if not self.config.get("use_kv", True):
            gate = torch.zeros_like(gate)
        fused = gate * retrieved + (1 - gate) * pooled
        logits = self.classifier(fused)

        # For DataParallel compatibility: convert all values to tensors (no floats/lists)
        info = {
            "gate_values": gate.detach().view(-1),
            # Convert kv_details floats to tensors for DataParallel gather
            "kv_avg_hits": torch.tensor(kv_details.get("avg_hits", 0.0), device=pooled.device, dtype=torch.float32),
            "kv_avg_similarity": torch.tensor(kv_details.get("avg_similarity", 0.0), device=pooled.device, dtype=torch.float32),
            "pooled": pooled.detach(),
            "retrieved": retrieved.detach(),
        }
        return logits, info

    def encode_text(self, text: str, max_length: int) -> torch.Tensor:
        ids = self.tokenizer.encode(text, max_length)
        if self.use_pretrained:
            device = next(self.pretrained_encoder.parameters()).device
        else:
            device = self.embedding.weight.device
        tensor_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = (tensor_ids != self.tokenizer.pad_token_id).float()
        with torch.no_grad():
            if self.use_pretrained:
                outputs = self.pretrained_encoder(input_ids=tensor_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                embeddings = self.embedding(tensor_ids)
                mask = attention_mask.unsqueeze(-1)
                pooled = (embeddings * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return pooled.squeeze(0)

    def get_pretrained_config(self) -> Dict[str, object]:
        config = dict(self.config)
        config["num_labels"] = self.num_labels
        config["vocab_size"] = int(self.embedding.num_embeddings)
        return config

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(self.get_pretrained_config(), handle, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        torch.save(self.kv_memory.get_state(), os.path.join(save_directory, "kv_memory.pt"))
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        tokenizer,
        map_location: Optional[str] = "cpu",
        overrides: Optional[Dict[str, object]] = None,
    ) -> "TransformerWithKV":
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
        combined_config = override_config(CONFIG, saved_config)
        if overrides:
            combined_config = override_config(combined_config, overrides)
        num_labels = int(combined_config.pop("num_labels", saved_config.get("num_labels", combined_config.get("num_labels", len(CONFIG)))))
        vocab_size = int(combined_config.pop("vocab_size", saved_config.get("vocab_size", len(tokenizer))))
        device = torch.device(map_location or combined_config.get("device", "cpu"))
        combined_config["device"] = str(device)
        model = cls(vocab_size, num_labels, tokenizer, combined_config)
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        kv_path = os.path.join(load_directory, "kv_memory.pt")
        if os.path.exists(kv_path):
            kv_state = torch.load(kv_path, map_location=device)
            model.kv_memory.load_state(kv_state)
        model.to(device)
        return model


class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, tokenizer, config: Dict[str, object]) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(vocab_size, config["d_model"])
        self.pos_encoder = PositionalEncoding(config["d_model"], dropout=config["dropout"], max_len=config["max_seq_length"])
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=config["num_layers"])
        self.layer_norm = nn.LayerNorm(config["d_model"])
        self.classifier = nn.Linear(config["d_model"], num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids) * math.sqrt(self.config["d_model"])
        embeddings = self.pos_encoder(embeddings)
        hidden = self.transformer(embeddings)
        hidden = self.layer_norm(hidden)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        logits = self.classifier(pooled)
        return logits
