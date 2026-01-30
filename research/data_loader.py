"""
Data Loader for Crypto Trading System

Fetches OHLCV data from cryptocurrency exchanges using CCXT.
Supports multiple symbols, timeframes, and data preprocessing.
"""

import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml


class CryptoDataLoader:
    """Load and preprocess cryptocurrency data from exchanges."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize exchange
        exchange_name = self.config['exchange']['name']
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': self.config['exchange']['enable_rate_limit']
        })
        
        self.data_dir = Path("data/crypto")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '4h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Save to CSV
            
        Returns:
            DataFrame with OHLCV data
        """
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
        
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                current_ts = candles[-1][0] + 1
                
                # Rate limiting
                self.exchange.sleep(self.exchange.rateLimit)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Save to CSV
        if save:
            filename = f"{symbol.replace('/', '-')}_{timeframe}.csv"
            filepath = self.data_dir / filename
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} candles to {filepath}")
        
        return df
    
    def fetch_multiple_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        symbols = symbols or self.config['symbols']
        
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe)
                data[symbol] = df
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
        
        return data
    
    def load_local_data(
        self,
        symbol: str,
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """Load previously downloaded data from CSV."""
        filename = f"{symbol.replace('/', '-')}_{timeframe}.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OHLCV data.
        
        - Fill missing values
        - Remove outliers
        - Normalize (optional)
        """
        df = df.copy()
        
        # Fill missing values
        fill_method = self.config['preprocessing']['fill_method']
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        
        # Remove outliers (optional)
        if self.config['preprocessing'].get('outlier_threshold'):
            threshold = self.config['preprocessing']['outlier_threshold']
            for col in ['open', 'high', 'low', 'close', 'volume']:
                mean = df[col].mean()
                std = df[col].std()
                df = df[
                    (df[col] >= mean - threshold * std) &
                    (col == 'volume' or df[col] <= mean + threshold * std)
                ]
        
        return df

    def align_multiple_assets(
        self,
        assets_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Align multiple asset dataframes by timestamp.
        Returns a single DataFrame with multi-indexed columns (asset, feature).
        """
        combined = []
        for symbol, df in assets_data.items():
            df = df.copy()
            # Set timestamp as index for joining
            df = df.set_index('timestamp')
            # Add symbol prefix to columns using MultiIndex
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
            combined.append(df)
        
        # Merge all dataframes on index (timestamp) using inner join to ensure alignment
        merged_df = pd.concat(combined, axis=1, join='inner')
        # Reset index to bring timestamp back as column
        merged_df = merged_df.reset_index()
        
        return merged_df


def main():
    """Example usage."""
    loader = CryptoDataLoader()
    
    # Fetch data for configured symbols
    data = loader.fetch_multiple_symbols(timeframe='1h')
    
    print(f"\nFetched data for {len(data)} symbols:")
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} candles")


if __name__ == "__main__":
    main()
