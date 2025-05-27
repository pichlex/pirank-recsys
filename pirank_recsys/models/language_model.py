
"""Language model for document embeddings."""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List


class LanguageModel:
    """Small language model for document embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, documents: List[str]) -> torch.Tensor:
        """Encode documents into embeddings."""
        inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings
