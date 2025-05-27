"""LambdaMART baseline model."""

import numpy as np
from typing import Dict, Any
import lightgbm as lgb
from sklearn.base import BaseEstimator


class LambdaMARTModel(BaseEstimator):
    """LambdaMART model using LightGBM."""

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        verbose: int = 0,
        **kwargs
    ):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "LambdaMARTModel":
        """Train LambdaMART model."""
        train_data = lgb.Dataset(X, label=y, group=group)
        
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "verbose": self.verbose,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
        )
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def compute_metrics(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> Dict[str, float]:
        """Compute ranking metrics."""
        from ..utils.metrics import compute_ndcg, compute_map, compute_pfound

        scores = self.predict(X)
        
        ndcg_scores = []
        map_scores = []
        pfound_scores = []
        
        start_idx = 0
        for group_size in group:
            end_idx = start_idx + group_size
            group_scores = scores[start_idx:end_idx]
            group_labels = y[start_idx:end_idx]
            
            # Sort by scores (descending)
            sorted_indices = np.argsort(-group_scores)
            sorted_labels = group_labels[sorted_indices]
            
            ndcg_scores.append(compute_ndcg(sorted_labels))
            map_scores.append(compute_map(sorted_labels))
            pfound_scores.append(compute_pfound(sorted_labels))
            
            start_idx = end_idx

        return {
            "ndcg": np.mean(ndcg_scores),
            "map": np.mean(map_scores),
            "pfound": np.mean(pfound_scores),
        }
