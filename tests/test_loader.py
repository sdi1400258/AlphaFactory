
import unittest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from research import common

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent.parent / "data" / "crypto"

    def test_load_existing_format(self):
        # Test loading BTC-USD.csv (known existing format)
        file_path = self.data_dir / "BTC-USD.csv"
        if not file_path.exists():
            self.skipTest(f"{file_path} not found")
        
        df = common.load_ohlcv_csv(str(file_path))
        self.assertFalse(df.empty)
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
    def test_load_new_format(self):
        # Test loading one of the new files, e.g., btc_1d_data_2018_to_2025.csv
        file_path = self.data_dir / "btc_1d_data_2018_to_2025.csv"
        if not file_path.exists():
            self.skipTest(f"{file_path} not found")

        df = common.load_ohlcv_csv(str(file_path))
        self.assertFalse(df.empty, "DataFrame should not be empty for new format")
        
        # Check Extended Columns
        extended_cols = ["timestamp", "open", "high", "low", "close", "volume", 
                         "trades", "taker_buy_base", "quote_volume"]
        for col in extended_cols:
            self.assertIn(col, df.columns, f"Missing {col}")
        
        # Verify timestamp is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))
        
        # Quick check if features run without error
        from research import feature_engineering
        bar_cfg = common.BarConfig()
        feat_cfg = feature_engineering.FeatureConfig(bar=bar_cfg)
        X, y, cols, times = feature_engineering.build_features(df, feat_cfg)
        self.assertGreater(len(cols), 10)
        self.assertIn("avg_trade_size", cols)

if __name__ == "__main__":
    unittest.main()
