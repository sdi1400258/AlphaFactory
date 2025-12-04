"""
Utility script to download daily OHLCV data for equities, crypto, and FX.

Usage examples (from project root):

    python -m data.download_data --asset-class equities --symbol AAPL --start 2015-01-01
    python -m data.download_data --asset-class crypto --symbol BTC-USD --start 2018-01-01
    python -m data.download_data --asset-class fx --symbol EURUSD --start 2015-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_equity(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end or None, progress=False)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index.name = "timestamp"
    return df.reset_index()


def download_crypto(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    # Many crypto pairs are available via Yahoo (e.g. BTC-USD, ETH-USD)
    return download_equity(symbol, start=start, end=end)


def download_fx(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """
    Simple FX via Yahoo (e.g. EURUSD=X). For higher-quality FX, plug in a
    dedicated data provider API here instead.
    """
    yahoo_symbol = f"{symbol}=X" if not symbol.endswith("=X") else symbol
    return download_equity(yahoo_symbol, start=start, end=end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data to CSV.")
    parser.add_argument(
        "--asset-class",
        choices=["equities", "crypto", "fx"],
        required=True,
        help="Asset class to download.",
    )
    parser.add_argument("--symbol", required=True, help="Ticker / symbol, e.g. AAPL, BTC-USD, EURUSD.")
    parser.add_argument(
        "--start", default="2015-01-01", help="Start date (YYYY-MM-DD). Default: %(default)s"
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYY-MM-DD). Default: None (up to latest).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.asset_class == "equities":
        df = download_equity(args.symbol, args.start, args.end)
        out_dir = root / "data" / "equities"
        out_name = f"{args.symbol}.csv"
    elif args.asset_class == "crypto":
        df = download_crypto(args.symbol, args.start, args.end)
        out_dir = root / "data" / "crypto"
        out_name = f"{args.symbol}.csv"
    else:  # fx
        df = download_fx(args.symbol, args.start, args.end)
        out_dir = root / "data" / "fx"
        out_name = f"{args.symbol}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    df[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()


