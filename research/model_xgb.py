"""
XGBoost model wrapper for the research pipeline.
Flattens 3D time-series tensors [N, T, F] into 2D [N, T*F] for tree boosting.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import xgboost as xgb


@dataclass
class XGBConfig:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "auto"
    early_stopping_rounds: int | None = 10
    n_jobs: int = -1


class XGBAlpha:
    def __init__(self, cfg: XGBConfig):
        self.cfg = cfg
        self.model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            tree_method=cfg.tree_method,
            n_jobs=cfg.n_jobs,
            objective="reg:squarederror",
        )

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # X: [N, T, F] -> [N, T*F]
        if X.ndim == 3:
            N, T, F = X.shape
            return X.reshape(N, T * F)
        return X

    def fit(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        X_train, y_train = train_data
        X_train = self._flatten(X_train)
        
        eval_set = []
        if val_data is not None:
            X_val, y_val = val_data
            X_val = self._flatten(X_val)
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            verbose=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)


def train_xgb(
    model: XGBAlpha,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
) -> XGBAlpha:
    model.fit(train_data, val_data)
    return model
