"""Inference module for PiRank model."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json

from ..models.pirank import PiRankModel
from ..models.language_model import LanguageModel


class PiRankPredictor:
    """Predictor for PiRank model."""

    def __init__(self, model_path: Path, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load PiRank model
        self.model = PiRankModel(**config.get("model", {}))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model.to(self.device)
        
        # Load language model
        self.language_model = LanguageModel(
            config.get("language_model_path", "sentence-transformers/all-MiniLM-L6-v2")
        )

    def predict(self, documents: List[str], query: str) -> List[float]:
        """Generate relevance scores for documents given a query."""
        # Combine query and documents for encoding
        texts = [f"{query} [SEP] {doc}" for doc in documents]
        
        # Get embeddings
        embeddings = self.language_model.encode(texts)
        embeddings = embeddings.to(self.device)
        
        # Get relevance scores
        with torch.no_grad():
            scores = self.model(embeddings)
        
        return scores.cpu().numpy().tolist()

    def rank_documents(self, documents: List[str], query: str) -> List[Dict[str, Any]]:
        """Rank documents by relevance to query."""
        scores = self.predict(documents, query)
        
        # Create ranked list
        ranked_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            ranked_docs.append({
                "document_id": i,
                "document": doc,
                "relevance_score": score,
            })
        
        # Sort by relevance score (descending)
        ranked_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_docs


def run_inference(model_path: Path, data_path: Path, output_path: Path, config: Dict[str, Any]) -> None:
    """Run inference on new data."""
    predictor = PiRankPredictor(model_path, config)
    
    # Load input data
    with open(data_path, "r") as f:
        input_data = json.load(f)
    
    results = []
    for item in input_data:
        query = item["query"]
        documents = item["documents"]
        
        ranked_docs = predictor.rank_documents(documents, query)
        results.append({
            "query": query,
            "ranked_documents": ranked_docs,
        })
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
