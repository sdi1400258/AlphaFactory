"""
Feature engineering for multi-asset OHLCV + microstructure-style features.

Transforms a long dataframe of bars into model-ready 3D tensors:
[samples, lookback, features].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .common import BarConfig, compute_forward_returns


@dataclass
class FeatureConfig:
    bar: BarConfig
    include_microstructure: bool = True
    include_technical: bool = True
    include_volume: bool = True


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"]).diff()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()
    df["vol_chg"] = df["vol_20"] / (df["vol_60"] + 1e-8)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_zscore_20"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        / (df["volume"].rolling(20).std() + 1e-8)
    )
    df["vol_roll_sum_20"] = df["volume"].rolling(20).sum()
    return df


def add_microstructure_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate microstructure with bar-based features:
    - high_low_spread: (high - low) / close
    - close_position: where close sits in the bar range
    """
    df = df.copy()
    rng = df["high"] - df["low"]
    df["high_low_spread"] = rng / (df["close"] + 1e-8)
    df["close_pos_in_bar"] = (df["close"] - df["low"]) / (rng + 1e-8)
    return df


def build_features(
    df: pd.DataFrame, feature_cfg: FeatureConfig
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Turn a single-asset dataframe into rolling-window tensors.
    Returns:
        X: [n_samples, lookback, n_features]
        y: [n_samples] forward returns
    """
    df = df.copy()
    df = add_basic_features(df)
    if feature_cfg.include_volume:
        df = add_volume_features(df)
    if feature_cfg.include_microstructure:
        df = add_microstructure_proxies(df)

    df["fwd_ret"] = compute_forward_returns(
        df["close"], feature_cfg.bar.prediction_horizon
    )

    # Drop initial NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "fwd_ret",
        }
    ]

    lookback = feature_cfg.bar.lookback_window
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    values = df[feature_cols].values.astype("float32")
    targets = df["fwd_ret"].values.astype("float32")

    for i in range(lookback, len(df)):
        X_list.append(values[i - lookback : i])
        y_list.append(targets[i])

    if not X_list:
        X = np.empty((0, lookback, len(feature_cols)), dtype="float32")
        y = np.empty((0,), dtype="float32")
    else:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype="float32")

    return X, y, feature_cols


