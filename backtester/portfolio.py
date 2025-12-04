"""
Simple portfolio and PnL aggregation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.evaluation import (
    PerformanceStats,
    summarize_performance,
    equity_from_returns,
)


@dataclass
class PortfolioResult:
    equity_curve: np.ndarray
    returns: np.ndarray
    stats: PerformanceStats


def pnl_from_positions(
    positions: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    """
    positions: [T] or [T, N]
    asset_returns: same shape, arithmetic returns
    """
    if asset_returns.ndim == 2:
        return (positions * asset_returns).sum(axis=-1)
    return positions * asset_returns


def build_portfolio(
    positions: np.ndarray,
    asset_returns: np.ndarray,
) -> PortfolioResult:
    strat_returns = pnl_from_positions(positions, asset_returns)
    equity = equity_from_returns(strat_returns)
    stats = summarize_performance(strat_returns)
    return PortfolioResult(equity_curve=equity, returns=strat_returns, stats=stats)


