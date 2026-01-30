"""
Multi-Asset Cryptocurrency Portfolio Environment for FinRL

Implements a Gymnasium environment for training RL agents to manage 
a portfolio of multiple cryptocurrencies simultaneously.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import yaml


class MultiCryptoTradingEnv(gym.Env):
    """
    Simultaneous Portfolio Management Environment.
    
    State Space:
        - Features for all assets (normalized)
        - Current portfolio weights
        - Portfolio value metrics
        
    Action Space:
        - Continuous: [w1, w2, ..., wn] where wi is the target weight for asset i.
        - Weights are normalized such that sum(abs(wi)) <= max_leverage.
        
    Reward:
        - Portfolio returns
        - Risk-adjusted returns (Sharpe/Sortino)
    """
    
    metadata = {'render_modes': ['human']}
    def __init__(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        reward_type: str = 'sharpe',
        max_leverage: float = 0.95,
        lookback: int = 1,
        use_kelly: bool = True
    ):
        """
        Initialize multi-asset trading environment.
        
        Args:
            df: Merged DataFrame with MultiIndex columns (symbol, feature)
            symbols: List of symbols in the portfolio
            initial_balance: Starting capital
            transaction_cost: Trading fee
            slippage: Slippage per trade
            reward_type: 'sharpe', 'pnl', 'sortino'
            max_leverage: Max fraction of capital to allocate
            lookback: Number of past steps to include in observation (optional)
            use_kelly: Whether to scale actions by Kelly Criterion
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.reward_type = reward_type
        self.max_leverage = max_leverage
        self.use_kelly = use_kelly
        
        # Identify feature columns (common across all assets)
        # Assuming df has MultiIndex (symbol, feature)
        self.asset_features = [col for col in df[symbols[0]].columns if col not in ['timestamp']]
        num_assets = len(symbols)
        num_features_per_asset = len(self.asset_features)
        
        # State space: [Assets * Features] + [Assets * CurrentWeights] + [PortfolioMetrics]
        total_features = (num_assets * num_features_per_asset) + num_assets + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )
        
        # Action space: Target weights for each asset
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_assets,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # Portfolio state
        self.cash = initial_balance
        self.holdings = {symbol: 0.0 for symbol in symbols}
        self.weights = np.zeros(num_assets)
        self.total_value = initial_balance
        
        # Performance tracking
        self.portfolio_returns = []
        self.trades_log = []
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_balance
        self.holdings = {symbol: 0.0 for symbol in self.symbols}
        self.weights = np.zeros(len(self.symbols))
        self.total_value = self.initial_balance
        self.portfolio_returns = []
        self.trades_log = []
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step."""
        # 1. Determine target weights based on action and sizing method
        # If action is from RL, it's a preferred allocation. 
        # We can further scale it by Kelly fraction if enabled.
        
        target_weights = action
        
        # Apply Kelly Scaling if configured (optional addition to the raw RL action)
        # In a Multi-Asset RL context, the agent usually LEARNS the weights, 
        # but we can provide a 'Kelly-adjusted' reward or observation.
        # Here we scale the raw action by a calculated Kelly fraction.
        
        if hasattr(self, 'use_kelly') and self.use_kelly:
            f_kelly = self._calculate_kelly_fraction()
            target_weights = target_weights * f_kelly

        # Normalize actions to target weights
        # Ensure weights sum to <= max_leverage
        total_action = np.sum(target_weights)
        if total_action > self.max_leverage:
            target_weights = (target_weights / total_action) * self.max_leverage
            
        # 2. Get current prices for all assets
        current_prices = {
            symbol: self.df.loc[self.current_step, (symbol, 'close')]
            for symbol in self.symbols
        }
        
        # 3. Calculate current portfolio value before trades
        current_holdings_value = sum(self.holdings[s] * current_prices[s] for s in self.symbols)
        self.total_value = self.cash + current_holdings_value
        
        # 4. Execute trades to match target weights
        # Calculate target dollar amount for each asset
        target_values = target_weights * self.total_value
        
        # Rebalance: Sell first, then buy
        # This is a simplified rebalancing logic
        
        # a. Sells
        for i, symbol in enumerate(self.symbols):
            curr_val = self.holdings[symbol] * current_prices[symbol]
            if curr_val > target_values[i]:
                diff = curr_val - target_values[i]
                shares_to_sell = diff / current_prices[symbol]
                proceeds = diff * (1 - self.transaction_cost - self.slippage)
                self.holdings[symbol] -= shares_to_sell
                self.cash += proceeds
                
        # b. Buys
        for i, symbol in enumerate(self.symbols):
            curr_val = self.holdings[symbol] * current_prices[symbol]
            if curr_val < target_values[i]:
                diff = target_values[i] - curr_val
                # Check if we have enough cash
                buy_amount = min(diff, self.cash * 0.999) # Keep tiny buffer
                if buy_amount > 0:
                    cost = buy_amount * (1 + self.transaction_cost + self.slippage)
                    if cost <= self.cash:
                        shares_to_buy = buy_amount / current_prices[symbol]
                        self.holdings[symbol] += shares_to_buy
                        self.cash -= cost
        
        # 5. Update total value and weights after trades
        old_value = self.total_value
        current_holdings_value = sum(self.holdings[s] * current_prices[s] for s in self.symbols)
        self.total_value = self.cash + current_holdings_value
        
        if self.total_value > 0:
            self.weights = np.array([
                (self.holdings[s] * current_prices[s]) / self.total_value 
                for s in self.symbols
            ])
        
        # 6. Calculate return and reward
        ret = (self.total_value - old_value) / old_value if old_value > 0 else 0
        self.portfolio_returns.append(ret)
        
        reward = self._calculate_reward()
        
        # 7. Move step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 8. Info
        info = {
            'total_value': self.total_value,
            'cash': self.cash,
            'weights': self.weights,
            'return': ret,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Concatenate all asset features and portfolio state."""
        obs_parts = []
        
        # 1. Asset features
        for symbol in self.symbols:
            # All features for this asset at current step
            feat_values = self.df.loc[self.current_step, symbol][self.asset_features].values
            obs_parts.append(feat_values.astype(np.float32))
            
        # 2. Portfolio weights
        obs_parts.append(self.weights.astype(np.float32))
        
        # 3. Portfolio metrics (normalized)
        metrics = np.array([
            self.cash / self.initial_balance,
            self.total_value / self.initial_balance,
            len(self.portfolio_returns) / self.max_steps # progression
        ], dtype=np.float32)
        obs_parts.append(metrics)
        
        return np.concatenate(obs_parts)
    
    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly fraction based on historical performance.
        f = (win_prob * ratio - loss_prob) / ratio
        Or simple volatility-based: f = mu / sigma^2
        """
        if len(self.portfolio_returns) < 50:
            return 1.0 # Default full sizing early on
            
        rets = np.array(self.portfolio_returns[-100:]) # Last 100 steps
        mu = np.mean(rets)
        var = np.var(rets) + 1e-8
        
        # Kelly fraction (f*)
        # We cap it at 1.0 to avoid excessive leverage
        f_star = mu / var
        
        # Half-Kelly for safety
        f_star = 0.5 * f_star
        
        return np.clip(f_star, 0.1, 1.0)

    def _calculate_reward(self) -> float:
        """Calculate portfolio-level reward."""
        if len(self.portfolio_returns) < 2:
            return 0.0
        
        rets = np.array(self.portfolio_returns)
        
        if self.reward_type == 'pnl':
            return rets[-1]
        
        elif self.reward_type == 'sharpe':
            mean = np.mean(rets)
            std = np.std(rets) + 1e-8
            # Annualized Sharpe (Hourly)
            return (mean / std) * np.sqrt(252 * 24)
            
        elif self.reward_type == 'sortino':
            mean = np.mean(rets)
            downside = rets[rets < 0]
            std = np.std(downside) + 1e-8 if len(downside) > 0 else 1e-8
            return (mean / std) * np.sqrt(252 * 24)
            
        return rets[-1]

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Total Value: ${self.total_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Weights: {dict(zip(self.symbols, self.weights.round(4)))}")


def test_multi_env():
    """Dummy test for MultiCryptoTradingEnv."""
    num_assets = 3
    num_steps = 100
    symbols = [f'ASSET_{i}' for i in range(num_assets)]
    
    # Create mock MultiIndex DataFrame
    cols = pd.MultiIndex.from_product([symbols, ['open', 'high', 'low', 'close', 'volume', 'rsi']])
    data = np.random.randn(num_steps, len(cols))
    df = pd.DataFrame(data, columns=cols)
    # Ensure prices are positive
    for s in symbols:
        df[(s, 'close')] = 100 + df[(s, 'close')].cumsum()
        
    env = MultiCryptoTradingEnv(df, symbols)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"Reward: {reward:.4f}, Value: {info['total_value']:.2f}")
        if done: break

if __name__ == "__main__":
    test_multi_env()
