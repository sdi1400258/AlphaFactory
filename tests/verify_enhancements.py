
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch

from research import feature_engineering as fe
from research import common
from research.model_transformer import TransformerAlpha, TransformerConfig, train_transformer, TransformerWrapper
from research.model_xgb import XGBAlpha, XGBConfig, train_xgb
from research.ensemble import EnsembleAlpha

def test_pipeline():
    print(">>> 1. Generating synthetic data...")
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100 + np.random.randn(200).cumsum(),
        "high": 105 + np.random.randn(200).cumsum(),
        "low": 95 + np.random.randn(200).cumsum(),
        "close": 100 + np.random.randn(200).cumsum(),
        "volume": np.abs(np.random.randn(200) * 1000) + 100
    })
    
    # Fix consistency
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    
    print(">>> 2. Testing Feature Engineering...")
    bar_cfg = common.BarConfig(lookback_window=20, prediction_horizon=1)
    # Enable all new features
    feat_cfg = fe.FeatureConfig(
        bar=bar_cfg,
        include_microstructure=True,
        include_technical=True,
        include_stats=True
    )
    
    X, y, cols, _ = fe.build_features(df, feat_cfg)
    print(f"Features shape: {X.shape}")
    print(f"Features: {cols}")
    
    # Check for expected technical columns
    expected = ["rsi_14", "macd_hist", "bb_width", "ret_skew_20"]
    for e in expected:
        assert isinstance(cols, list)
        if e not in cols:
            print(f"WARNING: Feature {e} not found in columns!")
    
    assert X.shape[0] > 0
    assert X.shape[1] == 20
    assert not np.isnan(X).any(), "Found NaNs in features"
    
    # Split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    print(">>> 3. Testing Transformer Training...")
    t_cfg = TransformerConfig(input_size=len(cols), d_model=16, nhead=2, num_layers=1)
    model_t = TransformerAlpha(t_cfg)
    train_transformer(model_t, (X_train, y_train), (X_val, y_val), epochs=1)
    print("Transformer trained.")
    
    # Wrap model
    model_t_wrapper = TransformerWrapper(model_t)
    
    print(">>> 4. Testing XGBoost Training...")
    x_cfg = XGBConfig(n_estimators=10, max_depth=3)
    model_x = XGBAlpha(x_cfg)
    train_xgb(model_x, (X_train, y_train), (X_val, y_val))
    print("XGBoost trained.")
    
    print(">>> 5. Testing Ensemble...")
    # Combine wrapped transformer and xgboost
    ensemble = EnsembleAlpha([model_t_wrapper, model_x], weights=[0.5, 0.5])
    preds = ensemble.predict(X_val)
    print(f"Ensemble preds shape: {preds.shape}")
    print(f"Sample preds: {preds[:5]}")
    
    assert preds.shape == (len(X_val),)
    # Ensure not all same (meaning models are doing something)
    assert np.std(preds) > 0, "Predictions are constant zero or flat!"
    
    print(">>> Verification Complete!")

if __name__ == "__main__":
    test_pipeline()
