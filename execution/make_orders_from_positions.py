"""
Convert a time series of desired positions into an orders CSV
consumed by the C++ execution simulator.

Input CSV format (example):

    timestamp,price,target_position
    2024-01-01 09:30:00,100.0,0
    2024-01-01 09:31:00,100.5,10
    2024-01-01 09:32:00,101.0,5

This script computes position changes (deltas) and emits one MARKET
order per change. Output CSV has **no header** and columns:

    timestamp_ns,side,price,qty,order_type,id

which is exactly what `alpha_engine_sim` expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def positions_to_orders(df: pd.DataFrame) -> pd.DataFrame:
    if not {"timestamp", "price", "target_position"} <= set(df.columns):
        raise ValueError("Input must contain columns: timestamp, price, target_position")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_position"] = df["target_position"].astype(float)

    df["delta"] = df["target_position"].diff().fillna(df["target_position"])

    orders = []
    for i, row in df.iterrows():
        delta = row["delta"]
        if delta == 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        qty = float(abs(delta))
        ts_ns = int(row["timestamp"].value)  # pandas datetime64 ns
        price = float(row["price"])
        order_type = "MARKET"
        order_id = f"ord_{i}"

        orders.append(
            {
                "timestamp_ns": ts_ns,
                "side": side,
                "price": price,
                "qty": qty,
                "order_type": order_type,
                "id": order_id,
            }
        )

    return pd.DataFrame(orders)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert target positions into an orders CSV for the C++ simulator."
    )
    parser.add_argument(
        "--positions-csv",
        required=True,
        help="Path to CSV with columns: timestamp,price,target_position",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output orders CSV path (no header, used by alpha_engine_sim).",
    )
    args = parser.parse_args()

    pos_path = Path(args.positions_csv)
    df = pd.read_csv(pos_path)
    orders_df = positions_to_orders(df)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with no header to match simulator expectations
    orders_df.to_csv(out_path, index=False, header=False)
    print(f"Wrote {len(orders_df)} orders to {out_path}")


if __name__ == "__main__":
    main()


