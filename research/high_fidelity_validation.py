"""
High-Fidelity Validation for AlphaFactory v3.0

Bridges the Python RL agent with the C++ Execution Engine to 
simulate realistic order book matching and slippage.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml
import json
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from research.data_loader import CryptoDataLoader
from research.feature_engineering import build_rl_features, add_cross_asset_features
from research.crypto_env import MultiCryptoTradingEnv


def run_high_fid_backtest(args):
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

    # Split into test set
    train_size = int(len(df_merged) * 0.8)
    df_test = df_merged[train_size:].reset_index(drop=True)
    
    # 3. Create Orders CSV for C++ Engine
    # Format: timestamp,side,price,qty,type,id
    orders_log = []
    
    print(f"Generating agent intentions on {len(df_test)} steps...")
    
    # We'll use the Python environment to get the agent's actions (intentions)
    def make_env():
        return MultiCryptoTradingEnv(df_test, symbols=symbols)
    
    env = make_env()
    obs, _ = env.reset()
    
    for i in range(len(df_test) - 1):
        action, _ = model.predict(obs, deterministic=True)
        ts_ns = int(df_test.loc[i, ('timestamp', '')].timestamp() * 1e9)
        
        # 1. Provide Market Liquidity (Simulated Book)
        # For each asset, we place limit orders around the current price
        # This allows the agent's market orders to actually execute
        for j, symbol in enumerate(symbols):
            price = df_test.loc[i, (symbol, 'close')]
            # Add some sell-side liquidity just above close
            orders_log.append({
                'timestamp': ts_ns,
                'side': 'SELL',
                'price': price * 1.0005,
                'qty': 100.0,
                'type': 'LIMIT',
                'id': f"mkt_lqd_ask_{i}_{symbol}",
                'symbol': symbol
            })
            # Add some buy-side liquidity just below close
            orders_log.append({
                'timestamp': ts_ns,
                'side': 'BUY',
                'price': price * 0.9995,
                'qty': 100.0,
                'type': 'LIMIT',
                'id': f"mkt_lqd_bid_{i}_{symbol}",
                'symbol': symbol
            })

        # 2. Agent Decisions (Rebalancing)
        target_weights = action
        # env.weights is current weights from last step
        curr_weights = env.weights
        
        for j, symbol in enumerate(symbols):
            if abs(target_weights[j] - curr_weights[j]) > 0.01:
                side = "BUY" if target_weights[j] > curr_weights[j] else "SELL"
                price = df_test.loc[i, (symbol, 'close')]
                
                # Emit Market Order for the agent
                orders_log.append({
                    'timestamp': ts_ns + 1000, 
                    'side': side,
                    'price': price,
                    'qty': 1.0, 
                    'type': 'MARKET',
                    'id': f"agent_{i}_{symbol}",
                    'symbol': symbol
                })
        
        obs, _, done, _, _ = env.step(action)
        if done: break

    # Save to CSV - Ensure 7 fields in correct order: timestamp,side,price,qty,type,id,symbol
    orders_df = pd.DataFrame(orders_log)
    orders_df = orders_df[['timestamp', 'side', 'price', 'qty', 'type', 'id', 'symbol']]
    orders_csv = "results/intentions.csv"
    os.makedirs("results", exist_ok=True)
    orders_df.to_csv(orders_csv, index=False, header=False)
    
    # 4. Run C++ Simulator
    print("Running C++ Execution Engine...")
    sim_bin = "execution_engine/alpha_simulator"
    if not os.path.exists(sim_bin):
        print(f"Error: {sim_bin} not found. Please compile it first.")
        return

    result = subprocess.run([sim_bin, orders_csv], capture_output=True, text=True)
    
    try:
        trades = json.loads(result.stdout)
        print(f"C++ Engine executed {len(trades)} trades with realistic matching.")
        
        if trades:
            print(f"Example Trade Fill: {trades[0]}")
            
        # Save high-fidelity results
        with open("results/cpp_fills.json", "w") as f:
            json.dump(trades, f, indent=2)
            
    except json.JSONDecodeError:
        print("Error parsing C++ output. Is it producing valid JSON?")
        print(result.stderr)
        print(result.stdout[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Fidelity Validation with C++ Engine")
    parser.add_argument("--model_path", type=str, help="Path to saved model")
    
    args = parser.parse_args()
    run_high_fid_backtest(args)
