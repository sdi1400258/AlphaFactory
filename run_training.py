"""
Main Training Script for AlphaFactory v3.0

Trains a Multi-Asset Reinforcement Learning agent (PPO) 
to manage a crypto portfolio simultaneously.
"""

import os
import argparse
import pandas as pd
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from research.data_loader import CryptoDataLoader
from research.feature_engineering import build_rl_features, add_cross_asset_features
from research.crypto_env import MultiCryptoTradingEnv


def train(args):
    # 1. Load Configs
    with open("configs/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/rl_config.yaml", "r") as f:
        rl_cfg = yaml.safe_load(f)

    symbols = data_cfg['symbols']
    print(f"Starting Multi-Asset training for: {symbols}")

    # 2. Load and Preprocess Data for all assets
    loader = CryptoDataLoader()
    assets_data = {}
    
    for symbol in symbols:
        try:
            print(f"Loading data for {symbol}...")
            df = loader.load_local_data(symbol)
            # Add advanced features
            df_feats = build_rl_features(df)
            assets_data[symbol] = df_feats
        except Exception as e:
            print(f"Error loading {symbol}: {e}")

    # Align all assets by timestamp
    print("Aligning multiple assets...")
    df_merged = loader.align_multiple_assets(assets_data)
    
    # Add cross-asset features (correlation, beta, relative strength)
    print("Adding cross-asset features...")
    df_merged = add_cross_asset_features(df_merged, symbols)
    print(f"Merged Data Shape with cross-asset features: {df_merged.shape}")

    # Split into train and validation (temporal split)
    train_size = int(len(df_merged) * 0.8)
    df_train = df_merged[:train_size]
    df_val = df_merged[train_size:]

    # 3. Initialize Environment
    def make_env():
        return MultiCryptoTradingEnv(
            df_train,
            symbols=symbols,
            initial_balance=rl_cfg['environment']['initial_balance'],
            transaction_cost=rl_cfg['environment']['transaction_cost'],
            slippage=rl_cfg['environment']['slippage'],
            reward_type=rl_cfg['reward']['type'],
            max_leverage=rl_cfg['environment'].get('max_leverage', 0.95),
            use_kelly=rl_cfg['position_sizing'].get('method') == 'kelly'
        )

    env = DummyVecEnv([make_env])

    # 4. Initialize Model (PPO)
    # Adjust hyperparameters for larger state/action space if needed
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=rl_cfg['training']['learning_rate'],
        n_steps=rl_cfg['training']['n_steps'],
        batch_size=rl_cfg['training']['batch_size'],
        gamma=rl_cfg['training']['gamma'],
        gae_lambda=rl_cfg['training']['gae_lambda'],
        verbose=1,
        tensorboard_log="./logs/ppo_multi_crypto/"
    )

    # 5. Training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_multi_crypto"
    )

    total_timesteps = args.timesteps or rl_cfg['training']['total_timesteps']
    print(f"Training for {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 6. Save Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_multi_crypto_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Asset RL agent")
    parser.add_argument("--timesteps", type=int, help="Total training timesteps")
    
    args = parser.parse_args()
    train(args)
