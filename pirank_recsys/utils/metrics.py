"""Ranking metrics implementation."""

import numpy as np
from typing import List


def compute_ndcg(labels: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    def dcg(scores: np.ndarray, k: int) -> float:
        scores = scores[:k]
        return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

    if len(labels) == 0:
        return 0.0

    # Actual DCG
    actual_dcg = dcg(labels, k)
    
    # Ideal DCG
    ideal_labels = np.sort(labels)[::-1]
    ideal_dcg = dcg(ideal_labels, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def compute_map(labels: np.ndarray) -> float:
    """Compute Mean Average Precision."""
    if len(labels) == 0:
        return 0.0

    relevant_docs = labels > 0
    if not relevant_docs.any():
        return 0.0

    precision_at_k = []
    num_relevant = 0
    
    for i, is_relevant in enumerate(relevant_docs):
        if is_relevant:
            num_relevant += 1
            precision_at_k.append(num_relevant / (i + 1))
    
    if not precision_at_k:
        return 0.0
    
    return np.mean(precision_at_k)


def compute_pfound(labels: np.ndarray, p_break: float = 0.15) -> float:
    """Compute pFound metric."""
    if len(labels) == 0:
        return 0.0

    p_look = 1.0
    pfound = 0.0
    
    for i, relevance in enumerate(labels):
        p_rel = relevance / np.max(labels) if np.max(labels) > 0 else 0
        pfound += p_look * p_rel
        p_look *= (1 - p_rel) * (1 - p_break)
        
        if p_look < 1e-6:  # Early stopping
            break
    
    return pfound
