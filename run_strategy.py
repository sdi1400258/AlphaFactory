"""
run_strategy.py

End-to-End Orchestration Script for AlphaFactory.
1. Loads OHLCV data for multiple assets.
2. Generates features (Technical + Statistical).
3. Trains an XGBoost model on a training split.
4. Generates predictions (Signals) for the test split.
5. Exports signals for the Execution Engine.
"""

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research import common, feature_engineering
from research.model_xgb import XGBAlpha, XGBConfig, train_xgb

# --- Configuration ---
LOOKBACK = 64
HORIZON = 5
TRAIN_SPLIT_DATE = "2023-01-01"

def load_all_assets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSVs from data/equities, data/crypto, data/fx."""
    assets = {}
    for cat in ["equities", "crypto", "fx"]:
        p = data_dir / cat
        if p.exists():
            for f in p.glob("*.csv"):
                model_name = f"{cat}.{f.stem}"  # e.g. equities.AAPL
                df = common.load_ohlcv_csv(str(f))
                if not df.empty:
                    assets[model_name] = df
    return assets

def train_and_predict(
    assets: Dict[str, pd.DataFrame]
) -> Tuple[XGBAlpha, Dict[str, pd.DataFrame]]:
    
    # 1. Prepare Global Dataset
    # Train a single XGBoost model on all assets
    
    print(f"Loaded {len(assets)} assets.")
    
    # We collect all samples to train
    all_X = []
    all_y = []
    
    # We will need to store test data to run predictions later
    test_datasets = {} # asset -> (X_test, dates, close_prices)
    
    bar_cfg = common.BarConfig(
        lookback_window=LOOKBACK,
        prediction_horizon=HORIZON
    )
    feat_cfg = feature_engineering.FeatureConfig(
        bar=bar_cfg,
        include_microstructure=True,
        include_technical=True, # RSI, MACD
        include_stats=True,     # Skew, Kurtosis
        include_volatility=True
    )
    
    feature_cols = []
    
    print("Building features...")
    for name, df in assets.items():
        print(f"  Processing {name}...")
        # Build features
        X, y, cols, times = feature_engineering.build_features(df, feat_cfg)
        if len(X) == 0:
            continue
            
        feature_cols = cols
        
        # Split by date using returned timestamps
        train_mask = times < np.datetime64(TRAIN_SPLIT_DATE)
        test_mask = times >= np.datetime64(TRAIN_SPLIT_DATE)
        
        # Append to Training Set
        if train_mask.any():
            all_X.append(X[train_mask])
            all_y.append(y[train_mask])
            
        # Store Test Data
        if test_mask.any():
            test_datasets[name] = {
                "X": X[test_mask],
                "dates": times[test_mask],
                "df_close": pd.merge(
                    pd.DataFrame({"timestamp": times[test_mask]}),
                    df[["timestamp", "close"]],
                    on="timestamp",
                    how="left"
                )["close"]
            }
        
        # Free memory immediately after processing each asset
        del df, X, y, times
        gc.collect()

    if not all_X:
        raise ValueError("No training data found! Check data dir or split date.")
        
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    
    # Free intermediate lists
    del all_X, all_y
    gc.collect()
    
    print(f"Training Data: {X_train.shape} samples. Features: {len(feature_cols)}")
    
    # Validation split (random 20% of train)
    perm = np.random.permutation(len(X_train))
    val_cut = int(len(X_train) * 0.8)
    train_idx = perm[:val_cut]
    val_idx = perm[val_cut:]
    
    X_t, y_t = X_train[train_idx], y_train[train_idx]
    X_v, y_v = X_train[val_idx], y_train[val_idx]
    
    # Free full training set after split
    del X_train, y_train
    gc.collect()
    
    # --- Train XGBoost ---
    print("Training XGBoost...")
    x_cfg = XGBConfig(n_estimators=100, max_depth=6, learning_rate=0.1)
    model_x = XGBAlpha(x_cfg)
    train_xgb(model_x, (X_t, y_t), (X_v, y_v))
    
    return model_x, test_datasets

def main():
    root = Path(__file__).parent
    data_dir = root / "data"
    
    # 1. Load
    assets = load_all_assets(data_dir)
    if not assets:
        print("No data found! Run 'python -m data.download_data' first.")
        return

    # 2. Train
    model_xgb, test_data = train_and_predict(assets)
    
    # Free assets dict after training
    del assets
    gc.collect()
    
    # 3. Predict & Export
    print("Generating Signals...")
    all_signals = []
    
    for name, data in test_data.items():
        preds = model_xgb.predict(data["X"])
        
        # Create Signal DF
        sig_df = pd.DataFrame({
            "timestamp": data["dates"],
            "asset": name,
            "signal": preds,
            "close": data["df_close"].values
        })
        all_signals.append(sig_df)
    
    if all_signals:
        final_df = pd.concat(all_signals).sort_values("timestamp")
        
        # 1. Save Signals
        out_signals = root / "signals" / "signal_files" / "ensemble_signals.csv"
        out_signals.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_signals, index=False)
        print(f"Saved {len(final_df)} signals to {out_signals}")
        
        # 2. Run Vectorized Backtest (PnL)
        print("Running Backtest...")
        # Simple PnL: Position * Returns
        # Shift signal to align with next day return? 
        # Our model predicts "fwd_ret". So if we trade NOW based on prediction, we capture that return.
        # But commonly signal is generated at Close, so we trade at next Open or Close.
        # Let's assume we trade at Close (Simulated) and hold for 1 bar.
        
        # Calculate Forward Returns in the final_df
        # We need to group by asset to shift correctly
        final_df["next_ret"] = final_df.groupby("asset")["close"].shift(-1) / final_df["close"] - 1.0
        final_df["pnl"] = final_df["signal"] * final_df["next_ret"]
        
        # Aggregate daily PnL
        daily_pnl = final_df.groupby("timestamp")["pnl"].sum().fillna(0)
        equity = (1 + daily_pnl).cumprod()
        
        pnl_df = pd.DataFrame({
            "timestamp": daily_pnl.index,
            "equity": equity.values,
            "daily_pnl": daily_pnl.values
        })
        out_pnl = root / "signals" / "signal_files" / "backtest_pnl.csv"
        pnl_df.to_csv(out_pnl, index=False)
        print(f"Saved PnL to {out_pnl}")

        # 3. Export Orders for C++
        print("Exporting Orders for C++ Engine...")
        # Convert signal (-1 to 1) to target position (e.g. -100 to 100 shares)
        # We need to pivot first to handle multiple assets or process sequentially?
        # The C++ simulator takes a linear stream of orders.
        # Let's target 100 units * signal
        
        orders = []
        for asset, group in final_df.groupby("asset"):
            group = group.sort_values("timestamp")
            # Target position
            group["target_pos"] = (group["signal"] * 100).round()
            group["delta"] = group["target_pos"].diff().fillna(group["target_pos"])
            
            for i, row in group.iterrows():
                delta = row["delta"]
                if delta == 0 or np.isnan(delta):
                    continue
                    
                side = "BUY" if delta > 0 else "SELL"
                qty = abs(delta)
                # Naive order generation: Market order at Close price
                # Timestamp in nanos
                ts_ns = int(row["timestamp"].value)
                
                # LIQUIDITY INJECTION (Market Maker)
                # To ensure the C++ matching engine executes the trade, 
                # we inject a passive LIMIT order on the opposite side.
                mm_side = "SELL" if side == "BUY" else "BUY"
                mm_id = f"lp_{asset}_{i}"
                mm_price = row["close"] # Assume tight spread, fill at close
                
                orders.append({
                    "timestamp_ns": ts_ns - 1, # Arrives just before
                    "side": mm_side,
                    "price": mm_price,
                    "qty": qty, # Provide exact liquidity needed
                    "order_type": "LIMIT",
                    "id": mm_id
                })
                
                # STRATEGY ORDER
                orders.append({
                    "timestamp_ns": ts_ns,
                    "side": side,
                    "price": row["close"],
                    "qty": qty,
                    "order_type": "MARKET",
                    "id": f"ord_{asset}_{i}"
                })
        
        # Sort by timestamp to simulate realistic feed
        orders_df = pd.DataFrame(orders).sort_values("timestamp_ns")
        out_orders = root / "signals" / "signal_files" / "orders.csv"
        # C++ engine expects no header, specific columns
        orders_df.to_csv(out_orders, index=False, header=False)
        print(f"Saved {len(orders_df)} orders to {out_orders}")

    else:
        print("No test data found to predict on.")

if __name__ == "__main__":
    main()
