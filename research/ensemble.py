"""
Ensemble Strategy combining multiple models.
"""
from __future__ import annotations

from typing import List, Protocol, Tuple, Any

import numpy as np


class AlphaModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class EnsembleAlpha:
    def __init__(self, models: List[AlphaModel], weights: List[float] | None = None):
        """
        Args:
            models: List of trained model instances (must have .predict method)
            weights: Optional list of weights. If None, equal weighting is used.
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: [N, T, F]
        # Collect predictions from all models
        preds_list = []
        for model in self.models:
            p = model.predict(X)
            preds_list.append(p)
        
        # Weighted Average
        # preds_list: List of [N] arrays
        # stack: [M, N]
        stacked = np.stack(preds_list, axis=0) 
        
        # Weighted sum across axis 0
        w = np.array(self.weights).reshape(-1, 1) # [M, 1]
        ensemble_pred = np.sum(stacked * w, axis=0)
        return ensemble_pred
