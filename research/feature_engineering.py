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
    include_momentum: bool = True  # RSI, MACD
    include_volatility: bool = True  # ATR, Bollinger
    include_stats: bool = True  # Skew, Kurtosis


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


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"] / 100.0  # Normalize to 0-1

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal
    # Normalize MACD by close price to make it asset-agnostic
    df["macd_hist_norm"] = df["macd_hist"] / (close + 1e-8)

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    # Distance from bands as feature
    df["bb_width"] = (upper - lower) / (close + 1e-8)
    df["bb_position"] = (close - lower) / (upper - lower + 1e-8)

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_rel"] = df["atr_14"] / (close + 1e-8)
    
    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    window = 20
    # Rolling Skew & Kurtosis of returns
    rets = df["ret_1"]
    df["ret_skew_20"] = rets.rolling(window).skew()
    df["ret_kurt_20"] = rets.rolling(window).kurt()
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
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Turn a single-asset dataframe into rolling-window tensors.
    Returns:
        X: [n_samples, lookback, n_features]
        y: [n_samples] forward returns
        feature_cols: list of feature names
        times: [n_samples] timestamps corresponding to the end of each input window
    """
    df = df.copy()
    df = add_basic_features(df)
    if feature_cfg.include_volume:
        df = add_volume_features(df)
    if feature_cfg.include_microstructure:
        df = add_microstructure_proxies(df)
    if feature_cfg.include_momentum or feature_cfg.include_volatility:
        df = add_technical_indicators(df)
    if feature_cfg.include_stats:
        df = add_statistical_features(df)

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
    t_list: List[np.datetime64] = []

    values = df[feature_cols].values.astype("float32")
    targets = df["fwd_ret"].values.astype("float32")
    times = df["timestamp"].values

    for i in range(lookback, len(df)):
        X_list.append(values[i - lookback : i])
        y_list.append(targets[i])
        t_list.append(times[i-1]) # Timestamp of the last bar in the window

    if not X_list:
        X = np.empty((0, lookback, len(feature_cols)), dtype="float32")
        y = np.empty((0,), dtype="float32")
        out_times = np.empty((0,), dtype="datetime64[ns]")
    else:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype="float32")
        out_times = np.array(t_list)

    return X, y, feature_cols, out_times


