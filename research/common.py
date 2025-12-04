import enum
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


class AssetClass(str, enum.Enum):
    EQUITY = "equity"
    FX = "fx"
    CRYPTO = "crypto"


@dataclass
class BarConfig:
    lookback_window: int = 128
    prediction_horizon: int = 5
    train_fraction: float = 0.7


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    Minimal loader that expects a CSV with at least:
    timestamp, open, high, low, close, volume
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    return prices.shift(-horizon) / prices - 1.0


