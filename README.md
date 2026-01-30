# AlphaFactory v3.0: Multi-Crypto DRL Trading System

AlphaFactory v3.0 is an industrial-grade Deep Reinforcement Learning (DRL) framework for multi-asset cryptocurrency portfolio management. It utilizes Proximal Policy Optimization (PPO) and the Kelly Criterion to manage a portfolio of top-tier crypto assets simultaneously.

![Pipeline Flowchart](/assets/pipeline_flowchart.png)

## Key Features
- **Multi-Asset RL**: A single Master Agent manages a portfolio of symbols (BTC, ETH, SOL, AVAX, MATIC).
- **Kelly Criterion**: Dynamic position sizing based on historical win rates and volatility-adjusted scaling.
- **Advanced Features**: Cross-asset correlation, beta to BTC market, and relative strength metrics.
- **C++ Execution Engine**: High-fidelity validation with an isolated multi-symbol order book matching engine.
- **Performance Dashboard**: Real-time visualization of equity curves, asset weights, and risk metrics.

---

## Quick Start

### 1. Local Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Compile the C++ Execution Engine
cd execution_engine
g++ -O3 -Wall -std=c++17 main.cpp matching_engine.cpp simulator.cpp -o alpha_simulator
cd ..
```

### 2. Training the Master Agent
```bash
# Start extended 1,000,000 timestep training
python run_training.py --timesteps 1000000
```

### 3. Backtesting & Validation
```bash
# Run standard portfolio backtest
python research/backtest.py

# Run High-Fidelity C++ Engine Validation
python research/high_fidelity_validation.py
```

### 4. Visualizing Performance
```bash
# Launch the Streamlit Dashboard
streamlit run dashboard_app.py
```

---

## Docker Deployment

Containerize everything for consistent deployment:

```bash
# 1. Build the Docker image
docker build -t alphafactory-v3 .

# 2. Run the full stack (Dashboard on port 8501)
docker run -p 8501:8501 alphafactory-v3
```

---

## Performance Targets
| Metric | Goal |
|--------|------|
| **Annual Return** | 30% - 50%+ |
| **Sharpe Ratio** | > 1.5 |
| **Max Drawdown** | < 15% |
| **Asset Universe** | BTC, ETH, SOL, AVAX, MATIC |

---

## Methodology
AlphaFactory v3.0 transitions from single-asset models to a holistic portfolio approach. By training a single RL agent to understand inter-asset relationships (correlations and relative strength), it achieves superior risk-adjusted returns compared to individual asset strategies.

---
**Disclaimer**: This software is for research purposes only. Cryptocurrency trading involves substantial risk.
