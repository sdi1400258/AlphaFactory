"""
Evaluation utilities: Sharpe, Sortino, drawdowns, and aggregation of
walk-forward results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np


@dataclass
class PerformanceStats:
    sharpe: float
    sortino: float
    max_drawdown: float
    mean: float
    vol: float


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    vol = excess.std()
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(252))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_vol = downside.std()
    if downside_vol == 0:
        return 0.0
    return float(excess.mean() / downside_vol * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / peaks
    return float(dd.min())


def equity_from_returns(returns: np.ndarray, start: float = 1.0) -> np.ndarray:
    return start * np.cumprod(1.0 + returns)


def summarize_performance(returns: np.ndarray) -> PerformanceStats:
    eq = equity_from_returns(returns)
    return PerformanceStats(
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=max_drawdown(eq),
        mean=float(returns.mean()),
        vol=float(returns.std()),
    )


def aggregate_walk_forward(results: List[Dict]) -> Dict[str, object]:
    """
    Flatten walk-forward segments into a single series.
    """
    y_true_all = np.concatenate([r["y_true"] for r in results])
    y_pred_all = np.concatenate([r["y_pred"] for r in results])

    # Simple linear strategy: proportional to prediction
    # position_t = sign(pred_t), pnl ~ position * actual
    # magnitude-sensitive positions
    positions = y_pred_all  # or np.tanh(y_pred_all), or y_pred_all / np.std(y_pred_all)

    returns = positions * y_true_all

    stats = summarize_performance(returns)
    return {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "positions": positions,
        "returns": returns,
        "stats": stats,
    }


