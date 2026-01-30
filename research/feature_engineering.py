"""
Advanced Feature Engineering for Crypto RL Trading.

Uses pandas_ta for robust technical indicators and statistical features.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import List, Tuple, Optional


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced technical and statistical features using pandas_ta.
    """
    df = df.copy()
    
    # 1. Momentum Indicators
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # 2. Volatility Indicators
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands.iloc[:, 2] # Use integer index to be safer
    df['bb_lower'] = bbands.iloc[:, 0]
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_norm'] = df['atr_14'] / df['close']
    
    # 3. Volume Indicators
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['obv_norm'] = df['obv'].pct_change()
    
    df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    # 4. Trend Indicators
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200)
    
    # Distance from EMAs
    df['dist_ema_20'] = (df['close'] - df['ema_20']) / df['ema_20']
    df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50']
    df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['ema_200']
    
    # 5. Statistical Features
    df['returns'] = df['close'].pct_change()
    df['roll_vol_20'] = df['returns'].rolling(20).std()
    df['roll_skew_20'] = df['returns'].rolling(20).skew()
    df['roll_kurt_20'] = df['returns'].rolling(20).kurt()
    
    # 6. Price Action
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['close']
    
    return df


def build_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for RL environment (single asset).
    """
    df = add_advanced_features(df)
    
    # Drop rows with NaNs (from indicators)
    df = df.dropna().reset_index(drop=True)
    
    return df


def add_cross_asset_features(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Add features that depend on multiple assets:
    - Correlation to BTC
    - Beta to BTC
    - Relative Return Rank
    
    Assumes df has MultiIndex columns (symbol, feature) and 'timestamp'.
    """
    df = df.copy()
    
    # Use BTC/USDT as market proxy (consistent with data_config.yaml)
    market_proxy = 'BTC/USDT'
    if market_proxy not in symbols:
        market_proxy = symbols[0] # Fallback
        
    market_rets = df[(market_proxy, 'returns')]
    
    for symbol in symbols:
        if symbol == market_proxy:
            df[(symbol, 'corr_btc')] = 1.0
            df[(symbol, 'beta_btc')] = 1.0
            continue
            
        asset_rets = df[(symbol, 'returns')]
        
        # 1. Rolling Correlation (60h ~ 2.5 days)
        df[(symbol, 'corr_btc')] = asset_rets.rolling(60).corr(market_rets)
        
        # 2. Rolling Beta
        # Beta = Cov(asset, market) / Var(market)
        covariance = asset_rets.rolling(60).cov(market_rets)
        variance = market_rets.rolling(60).var()
        df[(symbol, 'beta_btc')] = covariance / (variance + 1e-8)
        
    # 3. Relative Strength (Cross-sectional Rank)
    # Get all 'returns' columns
    returns_df = pd.DataFrame({
        s: df[(s, 'returns')] for s in symbols
    })
    
    # Rank assets by returns at each timestamp (normalized to 0-1)
    ranks = returns_df.rank(axis=1, pct=True)
    
    for symbol in symbols:
        df[(symbol, 'return_rank')] = ranks[symbol]
        
    # Drop rows with NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)
    
    return df


def get_feature_list(df: pd.DataFrame) -> List[str]:
    """
    Return list of feature columns (excluding OHLCV meta).
    """
    # For MultiIndex, we check the second level
    exclude = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'ema_20', 'ema_50', 'ema_200', 'bb_upper', 'bb_lower', 'obv', 'returns'}
    
    # If it's a MultiIndex (after cross-asset features)
    if isinstance(df.columns, pd.MultiIndex):
        # We want the unique features (level 1)
        features = set(df.columns.get_level_values(1))
        return sorted(list(features - exclude - {''}))
    else:
        return [col for col in df.columns if col not in exclude]
