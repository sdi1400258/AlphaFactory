"""
End-to-end backtest runner that takes model predictions or
walk-forward outputs and computes portfolio PnL.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .slippage_models import SlippageConfig, simple_bps_slippage
from .portfolio import build_portfolio, PortfolioResult


def backtest_from_walk_forward(
    walk_results: List[Dict],
    slippage_cfg: SlippageConfig | None = None,
) -> PortfolioResult:
    """
    Given walk-forward results (with y_true and y_pred),
    construct a simple long/short strategy and compute PnL.
    """
    y_true_all = np.concatenate([r["y_true"] for r in walk_results])
    y_pred_all = np.concatenate([r["y_pred"] for r in walk_results])
    positions = np.sign(y_pred_all)

    if slippage_cfg is not None:
        realized = simple_bps_slippage(
            mid_returns=y_true_all,
            trades=positions,
            cfg=slippage_cfg,
        )
    else:
        realized = y_true_all

    return build_portfolio(positions=positions, asset_returns=realized)


