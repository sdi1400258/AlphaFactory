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
    
    Also supports Binance format:
    Open time, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(path)
    
    # Standardize columns
    rename_map = {
        "Open time": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Quote asset volume": "quote_volume",
        "Number of trades": "trades",
        "Taker buy base asset volume": "taker_buy_base",
        "Taker buy quote asset volume": "taker_buy_quote"
    }
    df = df.rename(columns=rename_map)
    
    # Core columns that must exist
    required_core = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Extended columns (optional, fill with 0 if missing)
    extended = ["quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    
    # Validate core
    if not all(col in df.columns for col in required_core):
        missing = [c for c in required_core if c not in df.columns]
        print(f"Warning: {path} missing core columns {missing}")
        return pd.DataFrame()

    # Fill extended
    for col in extended:
        if col not in df.columns:
            df[col] = 0.0

    # Normalize to UTC and remove timezone to ensure compatibility (Naive UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Return all standardized columns
    return df[required_core + extended]


def compute_forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    return prices.shift(-horizon) / prices - 1.0


