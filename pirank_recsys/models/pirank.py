"""PiRank model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np


class PiRankModel(nn.Module):
    """PiRank model for learning to rank."""

    def __init__(
        self,
        input_dim: int = 136,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Build network layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(features).squeeze(-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute PiRank loss."""
        features = batch["features"]
        labels = batch["labels"]
        qids = batch["qids"]

        scores = self.forward(features)
        
        # Group by query ID and compute pairwise loss
        unique_qids = torch.unique(qids)
        total_loss = 0.0
        num_pairs = 0

        for qid in unique_qids:
            mask = qids == qid
            query_scores = scores[mask]
            query_labels = labels[mask]

            if len(query_scores) < 2:
                continue

            # Compute pairwise differences
            score_diff = query_scores.unsqueeze(1) - query_scores.unsqueeze(0)
            label_diff = query_labels.unsqueeze(1) - query_labels.unsqueeze(0)

            # Only consider pairs where labels differ
            valid_pairs = (label_diff != 0)
            if not valid_pairs.any():
                continue

            # PiRank loss: sigmoid of score differences weighted by label differences
            pairwise_loss = torch.sigmoid(-score_diff * torch.sign(label_diff))
            pairwise_loss = pairwise_loss[valid_pairs]
            
            total_loss += pairwise_loss.sum()
            num_pairs += len(pairwise_loss)

        if num_pairs == 0:
            return torch.tensor(0.0, requires_grad=True)

        return total_loss / num_pairs

    def compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute ranking metrics."""
        from ..utils.metrics import compute_ndcg, compute_map, compute_pfound

        features = batch["features"]
        labels = batch["labels"]
        qids = batch["qids"]

        with torch.no_grad():
            scores = self.forward(features)

        # Group predictions by query
        unique_qids = torch.unique(qids)
        ndcg_scores = []
        map_scores = []
        pfound_scores = []

        for qid in unique_qids:
            mask = qids == qid
            query_scores = scores[mask].cpu().numpy()
            query_labels = labels[mask].cpu().numpy()

            if len(query_scores) < 2:
                continue

            # Sort by scores (descending)
            sorted_indices = np.argsort(-query_scores)
            sorted_labels = query_labels[sorted_indices]

            ndcg_scores.append(compute_ndcg(sorted_labels))
            map_scores.append(compute_map(sorted_labels))
            pfound_scores.append(compute_pfound(sorted_labels))

        return {
            "ndcg": np.mean(ndcg_scores) if ndcg_scores else 0.0,
            "map": np.mean(map_scores) if map_scores else 0.0,
            "pfound": np.mean(pfound_scores) if pfound_scores else 0.0,
        }
