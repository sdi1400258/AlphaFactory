"""
Streamlit dashboard for AlphaFactory.

Shows:
- signal summaries from CSV
- PnL / equity curve from backtests
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from backtester.portfolio import equity_from_returns


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def main() -> None:
    st.title("AlphaFactory Dashboard")

    st.sidebar.header("Inputs")
    signal_path = st.sidebar.text_input(
        "Signal CSV", value="signals/signal_files/ensemble_signals.csv"
    )
    pnl_path = st.sidebar.text_input(
        "PnL CSV (optional)", value="signals/signal_files/backtest_pnl.csv"
    )

    if Path(signal_path).exists():
        sig = load_signals(signal_path)
        st.subheader("Signals overview")
        st.write(sig.head())
        st.line_chart(
            sig.pivot(index="timestamp", columns="asset", values="signal").fillna(0.0)
        )
    else:
        st.info("Signal file not found yet. Run research/backtest pipeline first.")

    if Path(pnl_path).exists():
        pnl_df = pd.read_csv(pnl_path, parse_dates=["timestamp"])
        st.subheader("Equity curve")
        st.line_chart(pnl_df.set_index("timestamp")["equity"])
    else:
        st.info("PnL file not found yet. After running backtest, save equity to CSV.")


if __name__ == "__main__":
    main()


