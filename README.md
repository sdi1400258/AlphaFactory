# AlphaFactory
Multi-Asset ML Alpha Factory with C++ Execution Engine

## Overview

AlphaFactory is a miniature systematic trading stack that combines:

- **Python research**: feature engineering, ML alpha models (LSTM, TCN, Transformer), walk-forward training, robustness tests, and backtesting.
- **C++ execution engine**: simple limit order book, matching engine, risk checks, and CSV-driven simulation.
- **Dashboard**: Streamlit app to visualize signals and PnL.

This is designed as a portfolio-quality project that looks like what a small quant research pod would build.

## Project Structure

- `data/`
  - `equities/`, `crypto/`, `fx/`: place your raw OHLCV data here (CSV with `timestamp,open,high,low,close,volume`).
- `research/`
  - `common.py`: shared configs and helpers (e.g. `BarConfig`, CSV loader, forward returns).
  - `feature_engineering.py`: microstructure/technical/volume features and tensorization into `[samples, lookback, features]`.
  - `model_lstm.py`: LSTM sequence model for returns prediction.
  - `model_tcn.py`: Temporal Convolutional Network model.
  - `model_transformer.py`: Transformer encoder model with positional encoding.
  - `walk_forward.py`: rolling walk-forward training and adversarial noise robustness test.
  - `evaluation.py`: Sharpe, Sortino, drawdowns, and aggregation of walk-forward results.
- `signals/`
  - `signal_exporter.py`: exports timestamped signals to `signals/signal_files/*.csv` for the execution engine.
- `execution_engine/`
  - `orderbook.hpp`: basic limit order book with bid/ask queues.
  - `matching_engine.cpp`: limit/market order matching, partial fills, and book updates.
  - `simulator.cpp`: CSV-driven simulator that replays orders and produces trades.
  - `risk_checks.cpp`: simple notional and max-position risk checks.
- `backtester/`
  - `slippage_models.py`: basic spread + impact slippage model in bps.
  - `portfolio.py`: PnL aggregation, equity curve, and stats wrapper.
  - `backtest.py`: plugs walk-forward outputs into slippage + portfolio.
- `notebooks/`
  - `01_feature_engineering.ipynb`: exploratory feature analysis (skeleton).
  - `02_model_training.ipynb`: model comparison and walk-forward (skeleton).
  - `03_pnl_analysis.ipynb`: PnL and risk analytics (skeleton).
- `dashboard_app.py`: Streamlit dashboard for signals and PnL.
- `CMakeLists.txt`: build configuration for the C++ execution engine.

## Python Research Pipeline

1. **Feature engineering**
   - Load OHLCV data with `research.common.load_ohlcv_csv`.
   - Build features (returns, volatility, volume z-scores, microstructure proxies) using `research.feature_engineering.build_features`.
2. **Model training**
   - Choose model family:
     - LSTM (`LSTMAlpha` in `model_lstm.py`)
     - TCN (`TCNAlpha` in `model_tcn.py`)
     - Transformer (`TransformerAlpha` in `model_transformer.py`)
   - Each model has a `train_*` helper that trains on `(X_train, y_train)` / `(X_val, y_val)`.
3. **Walk-forward and robustness**
   - Use `research.walk_forward.walk_forward` with a training function that constructs and trains your chosen model.
   - Use `research.walk_forward.adversarial_noisy_eval` to probe sensitivity to small input perturbations (adversarial robustness style).
4. **Evaluation**
   - Aggregate walk-forward segments with `research.evaluation.aggregate_walk_forward`.
   - Inspect Sharpe, Sortino, drawdowns, and PnL distribution.
5. **Backtesting**
   - Convert predictions to positions and run through `backtester.backtest.backtest_from_walk_forward`.
   - Apply realistic slippage with `backtester.slippage_models.simple_bps_slippage`.

## C++ Execution Engine

The execution engine is deliberately small but shows the right ideas:

- **Order book**: price levels with FIFO queues, best bid/ask, mid-price.
- **Matching**: limit and market orders, partial fills, queue-based priority.
- **Risk checks**: simple notional and max-position guards.
- **Simulator**: replays a CSV of orders and outputs trades.

Build it with CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

This produces the `alpha_engine` library target that can be linked into a larger trading system or a dedicated execution service.

## Streamlit Dashboard

Run the dashboard from the project root:

```bash
streamlit run dashboard_app.py
```

It will:

- Load signal CSVs from `signals/signal_files/alpha_ml_signals.csv`.
- Load PnL / equity curve CSV (e.g. `backtest_pnl.csv`).
- Plot signal heatmaps over time and equity curves to visually inspect behaviour.

## Data Download Helper

You can quickly pull daily OHLCV data into the `data/` folders using:

```bash
python -m data.download_data --asset-class equities --symbol AAPL --start 2015-01-01
python -m data.download_data --asset-class crypto --symbol BTC-USD --start 2018-01-01
python -m data.download_data --asset-class fx --symbol EURUSD --start 2015-01-01
```

This writes CSVs like:

- `data/equities/AAPL.csv`
- `data/crypto/BTC-USD.csv`
- `data/fx/EURUSD.csv`

Each CSV is normalized to `timestamp,open,high,low,close,volume` so it plugs directly into the research pipeline.

## Example End-to-End Flow

1. **Prepare data**
   - Drop an OHLCV CSV into `data/equities/` (for example).
2. **Feature + model training**
   - Use `research` modules or `02_model_training.ipynb` to:
     - build features
     - train LSTM/TCN/Transformer with walk-forward
     - export combined predictions.
3. **Export signals**
   - Convert model outputs into scores per asset and timestamp.
   - Call `signals.signal_exporter.export_signals` to write `alpha_ml_signals.csv`.
4. **Backtest and PnL**
   - Run `backtester.backtest.backtest_from_walk_forward` to obtain PnL.
   - Optionally save equity curve as `signals/signal_files/backtest_pnl.csv` for the dashboard.
5. **Execution simulation (C++)**
   - Generate a synthetic order stream (from signals, not included here) into a CSV.
   - Use the logic in `execution_engine/simulator.cpp` (`run_simulation_csv`) to simulate fills and slippage.


