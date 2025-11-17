"""
Frozen embedders for HSOKV.

All embedders are frozen (no training) to prevent embedding drift,
which would cause catastrophic forgetting.
"""

import torch
import torch.nn as nn
from typing import Union, List
from abc import ABC, abstractmethod


class FrozenEmbedder(ABC):
    """Base class for frozen embedders."""

    @abstractmethod
    def embed(self, input: Union[str, torch.Tensor]) -> torch.Tensor:
        """Embed input to vector."""
        pass

    @abstractmethod
    def get_dim(self) -> int:
        """Get embedding dimension."""
        pass


class SentenceBERTEmbedder(FrozenEmbedder):
    """Frozen Sentence-BERT embedder for text."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize Sentence-BERT embedder.

        Args:
            model_name: Sentence-BERT model name
            device: Device to run on
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.model.eval()

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Embed text to vector."""
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                device=self.device
            )
        return embedding

    def get_dim(self) -> int:
        return self._dim


class CLIPEmbedder(FrozenEmbedder):
    """Frozen CLIP embedder for images and text."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = 'cpu'):
        """
        Initialize CLIP embedder.

        Args:
            model_name: CLIP model name
            device: Device to run on
        """
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self._dim = 512  # CLIP embedding dimension

    def embed(self, input: Union[str, torch.Tensor, List[str]]) -> torch.Tensor:
        """
        Embed image or text to vector.

        Args:
            input: Image tensor, text string, or list of text strings

        Returns:
            Embedding tensor
        """
        with torch.no_grad():
            if isinstance(input, str) or (isinstance(input, list) and isinstance(input[0], str)):
                # Text input
                inputs = self.processor(text=input, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embedding = self.model.get_text_features(**inputs)
            else:
                # Image input
                if input.dim() == 3:
                    input = input.unsqueeze(0)

                # CLIP expects [0, 1] range
                if input.max() > 1.0:
                    input = input / 255.0

                inputs = self.processor(images=input, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embedding = self.model.get_image_features(**inputs)

            # Return single vector if single input
            if embedding.size(0) == 1:
                embedding = embedding[0]

            return embedding

    def get_dim(self) -> int:
        return self._dim
