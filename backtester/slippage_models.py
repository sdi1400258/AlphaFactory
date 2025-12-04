"""
Slippage and transaction cost models used by the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlippageConfig:
    half_spread_bps: float = 5.0
    impact_coeff_bps: float = 1.0


def simple_bps_slippage(
    mid_returns: np.ndarray,
    trades: np.ndarray,
    cfg: SlippageConfig,
) -> np.ndarray:
    """
    mid_returns: underlying mid-price returns
    trades: signed notional fraction of daily volume (0..1)
    Returns realized returns after slippage & impact.
    """
    # spread cost
    spread_cost = cfg.half_spread_bps * 1e-4 * np.abs(trades)
    # impact cost scales with trade size
    impact_cost = cfg.impact_coeff_bps * 1e-4 * np.square(trades)
    cost = spread_cost + impact_cost
    return mid_returns - cost


