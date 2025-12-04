"""
Signal export utilities.

Takes model predictions and writes timestamped signal CSVs
to `signals/signal_files/`, which can be consumed by the C++
execution engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SignalExportConfig:
    out_dir: str = "signals/signal_files"
    strategy_name: str = "alpha_ml"


def export_signals(
    timestamps: Iterable[pd.Timestamp],
    asset_ids: Iterable[str],
    scores: np.ndarray,
    cfg: SignalExportConfig | None = None,
) -> str:
    """
    scores: shape [T, N] of signal strengths (e.g. expected returns or ranks)
    """
    if cfg is None:
        cfg = SignalExportConfig()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_list = list(timestamps)
    asset_list = list(asset_ids)
    T, N = scores.shape
    if len(ts_list) != T:
        raise ValueError("timestamps length must match scores.shape[0]")
    if len(asset_list) != N:
        raise ValueError("asset_ids length must match scores.shape[1]")

    rows = []
    for t_idx, ts in enumerate(ts_list):
        for a_idx, asset in enumerate(asset_list):
            rows.append(
                {
                    "timestamp": ts,
                    "asset": asset,
                    "score": float(scores[t_idx, a_idx]),
                    "strategy": cfg.strategy_name,
                }
            )

    df = pd.DataFrame(rows)
    fname = f"{cfg.strategy_name}_signals.csv"
    path = out_dir / fname
    df.to_csv(path, index=False)
    return os.fspath(path)


