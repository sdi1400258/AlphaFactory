"""
Walk-forward training utilities and simple adversarial robustness tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class WalkForwardConfig:
    window_train: int = 1000
    window_val: int = 250
    step: int = 250


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    cfg: WalkForwardConfig,
    train_fn: Callable[[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], object],
) -> List[dict]:
    """
    Generic walk-forward loop. For each window:
      - trains model using `train_fn`
      - collects predictions on the subsequent step window
    """
    n = len(X)
    results: List[dict] = []
    start = 0
    while start + cfg.window_train + cfg.window_val + cfg.step <= n:
        train_start = start
        train_end = start + cfg.window_train
        val_end = train_end + cfg.window_val
        test_end = val_end + cfg.step

        X_train, y_train = X[train_start:train_end], y[train_start:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:test_end], y[val_end:test_end]

        model = train_fn((X_train, y_train), (X_val, y_val))

        preds = model_predict(model, X_test)

        results.append(
            {
                "train_range": (train_start, train_end),
                "val_range": (train_end, val_end),
                "test_range": (val_end, test_end),
                "y_true": y_test.copy(),
                "y_pred": preds.copy(),
            }
        )

        start += cfg.step

    return results


def model_predict(model: object, X: np.ndarray) -> np.ndarray:
    """
    Light wrapper that supports PyTorch-like models (with .eval and callable)
    and scikit-learn / xgboost-like models (with .predict).
    """
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype="float32")

    import torch

    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(next(model.parameters()).device)
        preds = model(X_t).cpu().numpy()
    return preds.astype("float32")


def adversarial_noisy_eval(
    model: object,
    X: np.ndarray,
    noise_std: float = 0.01,
    n_trials: int = 5,
) -> np.ndarray:
    """
    Basic adversarial-style robustness check:
    - Add Gaussian noise to inputs multiple times
    - Measure sensitivity of predictions.
    Returns: per-sample std of predictions across trials.
    """
    preds_all = []
    for _ in range(n_trials):
        X_noisy = X + np.random.normal(0.0, noise_std, size=X.shape).astype("float32")
        preds_all.append(model_predict(model, X_noisy))
    preds_all = np.stack(preds_all, axis=0)
    return preds_all.std(axis=0)


