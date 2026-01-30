"""
Streamlit Dashboard for AlphaFactory v3.0

Visualizes backtest results, portfolio performance, and asset allocations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os


def calculate_metrics(df):
    """Calculate key performance metrics."""
    rets = df['returns']
    total_ret = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
    
    # Annualized Return (Hourly data)
    ann_ret = (1 + total_ret) ** ( (252*24) / len(df) ) - 1
    
    # Volatility
    vol = rets.std() * np.sqrt(252*24)
    
    # Sharpe
    sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252*24)
    
    # Max Drawdown
    equity = df['equity']
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    
    return {
        "Total Return": f"{total_ret*100:.2f}%",
        "Annualized Return": f"{ann_ret*100:.2f}%",
        "Volatility (Ann)": f"{vol*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd*100:.2f}%"
    }


def main():
    st.set_page_config(page_title="AlphaFactory v3.0 Dashboard", layout="wide")
    
    st.title("🚀 AlphaFactory v3.0: Multi-Crypto Portfolio")
    st.markdown("### Deep RL Strategy Performance Analytics")
    
    # Sidebar for data selection
    results_file = "results/backtest_results.csv"
    if not os.path.exists(results_file):
        st.error(f"Results file not found at {results_file}. Please run a backtest first.")
        st.info("Run `python research/backtest.py` to generate results.")
        return
    
    df = pd.read_csv(results_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Metrics Overview
    metrics = calculate_metrics(df)
    cols = st.columns(len(metrics))
    for i, (label, value) in enumerate(metrics.items()):
        cols[i].metric(label, value)
        
    st.divider()
    
    # Main Plots
    row1_col1, row1_col2 = st.columns([2, 1])
    
    with row1_col1:
        st.subheader("Cumulative Equity Curve")
        fig_equity = px.line(df, x='timestamp', y='equity', title="Portfolio Value ($)")
        st.plotly_chart(fig_equity, use_container_width=True)
        
    with row1_col2:
        st.subheader("Returns Distribution")
        fig_hist = px.histogram(df, x='returns', nbins=50, title="Hourly Returns Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    st.divider()
    
    # Portfolio Weights
    st.subheader("Dynamic Portfolio Allocation")
    weight_cols = [c for c in df.columns if c.startswith('weight_')]
    
    # Melt for Plotly area chart
    df_weights = df.melt(id_vars=['timestamp'], value_vars=weight_cols, 
                         var_name='Asset', value_name='Weight')
    df_weights['Asset'] = df_weights['Asset'].str.replace('weight_', '')
    
    fig_weights = px.area(df_weights, x='timestamp', y='Weight', color='Asset',
                          title="Asset Weights Over Time")
    st.plotly_chart(fig_weights, use_container_width=True)
    
    # Final Weights Pie Chart
    st.divider()
    col_final1, col_final2 = st.columns(2)
    
    with col_final2:
        st.subheader("Final Portfolio Composition")
        final_weights = {}
        for c in weight_cols:
            final_weights[c.replace('weight_', '')] = df[c].iloc[-1]
        
        fig_pie = px.pie(names=list(final_weights.keys()), values=list(final_weights.values()),
                         title="Current Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_final1:
        st.subheader("Strategy Description")
        st.write("""
        **System**: AlphaFactory v3.0 (Multi-Crypto RL)  
        **Model**: PPO (Proximal Policy Optimization)  
        **Observation Space**: OHLCV + Indicators + Cross-Asset Correlation + BTC Beta  
        **Action Space**: Continuous Portfolio Weights (Simultaneous Management)  
        **Targets**: 30-50% Annual Return, Sharpe > 1.5, Max DD < 15%
        """)
        st.info("The agent learns to shift capital between assets based on relative strength and market regimes.")


if __name__ == "__main__":
    main()
