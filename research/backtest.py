"""
Multi-Asset Backtesting Utility for AlphaFactory v3.0

Runs a trained RL model on validation/test data and saves results for dashboard visualization.
"""

import os
import argparse
import pandas as pd
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from research.data_loader import CryptoDataLoader
from research.feature_engineering import build_rl_features, add_cross_asset_features
from research.crypto_env import MultiCryptoTradingEnv


def run_backtest(args):
    # 1. Load Configs
    with open("configs/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/rl_config.yaml", "r") as f:
        rl_cfg = yaml.safe_load(f)

    symbols = data_cfg['symbols']
    model_path = args.model_path or "models/ppo_multi_crypto_final.zip"
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # 2. Load and Preprocess Data
    loader = CryptoDataLoader()
    assets_data = {}
    
    for symbol in symbols:
        try:
            df = loader.load_local_data(symbol)
            df_feats = build_rl_features(df)
            assets_data[symbol] = df_feats
        except Exception as e:
            print(f"Error loading {symbol}: {e}")

    df_merged = loader.align_multiple_assets(assets_data)
    df_merged = add_cross_asset_features(df_merged, symbols)

    # Split into validation/test set based on training size
    train_size = int(len(df_merged) * 0.8)
    df_test = df_merged[train_size:]
    test_dates = df_test['timestamp'].values

    # 3. Initialize Environment
    def make_env():
        return MultiCryptoTradingEnv(
            df_test,
            symbols=symbols,
            initial_balance=rl_cfg['environment']['initial_balance'],
            transaction_cost=rl_cfg['environment']['transaction_cost'],
            slippage=rl_cfg['environment']['slippage'],
            reward_type=rl_cfg['reward']['type'],
            max_leverage=rl_cfg['environment'].get('max_leverage', 0.95)
        )

    env = DummyVecEnv([make_env])

    # 4. Run Simulation
    obs = env.reset()
    done = False
    
    portfolio_values = []
    weights_log = []
    returns = []
    
    print(f"Running backtest on {len(df_test)} steps...")
    
    idx = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        info = infos[0]
        portfolio_values.append(info['total_value'])
        weights_log.append(info['weights'])
        returns.append(info['return'])
        
        done = dones[0]
        idx += 1

    # 5. Save Results
    results_df = pd.DataFrame({
        'timestamp': test_dates[:len(portfolio_values)],
        'equity': portfolio_values,
        'returns': returns
    })
    
    # Add weights
    for i, symbol in enumerate(symbols):
        results_df[f'weight_{symbol}'] = [w[i] for w in weights_log]
        
    os.makedirs("results", exist_ok=True)
    out_file = "results/backtest_results.csv"
    results_df.to_csv(out_file, index=False)
    print(f"Backtest complete. Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest for Multi-Asset RL agent")
    parser.add_argument("--model_path", type=str, help="Path to saved model (.zip)")
    
    args = parser.parse_args()
    run_backtest(args)
