import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# Custom imports
from src.fetch_data import fetch_stock_data, load_prices, load_returns
from src.optimize import (
    optimize_max_sharpe, optimize_min_volatility,
    efficient_frontier, random_portfolios, RISK_FREE_RATE
)
from src.backtest import backtest_portfolio, performance_metrics
from generate_data import generate_synthetic_data, MARKET_CONDITIONS

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Portfolio Optimizer")
st.caption("Modern Portfolio Theory: Efficient Frontier & Asset Allocation")

# ============================================================================
# SIDEBAR: Data Source Selection & Configuration
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    data_source = st.radio(
        "**Data Source**",
        options=["Synthesized Data", "Live Data (Yahoo Finance)"],
        help="Choose between generated data or real market data"
    )
    
    if data_source == "Synthesized Data":
        market_condition = st.selectbox(
            "**Market Condition**",
            options=list(MARKET_CONDITIONS.keys()),
            format_func=lambda x: MARKET_CONDITIONS[x]["name"],
            help="Select market scenario"
        )
        market_desc = MARKET_CONDITIONS[market_condition]["description"]
        st.info(f"📊 {market_desc}")
        
        n_days = st.slider(
            "**Number of Trading Days**",
            min_value=252,
            max_value=2520,
            value=1566,
            step=252,
            help="Historical period for analysis"
        )
        
        # Generate data
        @st.cache_data
        def get_synthetic_data(condition, days, seed=42):
            df = generate_synthetic_data(condition=condition, n_days=days, seed=seed)
            prices = df["Close"].unstack()
            returns = prices.pct_change().dropna()
            return prices, returns
        
        prices, returns = get_synthetic_data(market_condition, n_days)
        data_status = f"✓ {market_condition.title()} Market ({len(prices)} days, {len(prices.columns)} stocks)"
        
    else:
        st.subheader("Live Data Configuration")
        tickers_input = st.text_input(
            "**Stock Tickers**",
            value="AAPL,MSFT,GOOG,AMZN,TSLA",
            help="Comma-separated ticker symbols"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        
        periods = st.selectbox(
            "**Historical Period**",
            options=["1y", "2y", "3y", "5y"],
            help="How far back to fetch data"
        )
        
        try:
            with st.spinner("📡 Fetching live data..."):
                end_date = datetime.now()
                if periods == "1y":
                    start_date = end_date - timedelta(days=365)
                elif periods == "2y":
                    start_date = end_date - timedelta(days=730)
                elif periods == "3y":
                    start_date = end_date - timedelta(days=1095)
                else:
                    start_date = end_date - timedelta(days=1825)
                
                import yfinance as yf
                data = yf.download(tickers, start=start_date, end=end_date, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    level0 = data.columns.get_level_values(0)
                    if "Adj Close" in level0:
                        prices = data.loc[:, level0 == "Adj Close"]
                    elif "Close" in level0:
                        prices = data.loc[:, level0 == "Close"]
                    else:
                        raise ValueError("Yahoo Finance data missing Adjusted Close or Close prices")
                    prices.columns = prices.columns.get_level_values(1)
                else:
                    if "Adj Close" in data.columns:
                        prices = data[["Adj Close"]].copy()
                        prices.columns = tickers if len(tickers) > 1 else [tickers[0]]
                    elif "Close" in data.columns:
                        prices = data[["Close"]].copy()
                        prices.columns = tickers if len(tickers) > 1 else [tickers[0]]
                    else:
                        raise ValueError("Yahoo Finance data missing Adjusted Close or Close prices")

                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=tickers[0])

                returns = prices.pct_change().dropna()
                
                data_status = f"✓ Live Data ({len(prices)} days, {len(prices.columns)} stocks)"
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            st.stop()
    
    st.success(data_status)
    
    st.divider()
    
    # Optimization parameters
    st.subheader("🎯 Optimization")
    n_random_portfolios = st.slider(
        "**Random Portfolios for Visualization**",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000,
        help="More portfolios = better frontier visualization"
    )
    
    n_frontier_points = st.slider(
        "**Efficient Frontier Points**",
        min_value=50,
        max_value=200,
        value=100,
        step=10
    )

# ============================================================================
# MAIN CONTENT: Tabs
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "🎯 Portfolio Optimization", "💰 Allocation", "📈 Backtest"]
)

# ============================================================================
# TAB 1: Data Overview
# ============================================================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Trading Days", len(prices))
        st.metric("Number of Stocks", len(prices.columns))
    
    with col2:
        st.metric("Date Range", f"{prices.index[0].date()} to {prices.index[-1].date()}")
        st.metric("Risk-Free Rate", f"{RISK_FREE_RATE:.1%}")
    
    st.subheader("Normalized Price Performance")
    
    # Normalize prices to 100
    normalized = prices / prices.iloc[0] * 100
    
    fig = px.line(
        normalized.reset_index(),
        x="Date",
        y=normalized.columns,
        title="Normalized Stock Prices (Base=100)",
        labels={"value": "Indexed Price", "variable": "Stock"}
    )
    fig.update_layout(hovermode="x unified", height=400)
    st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Annual Statistics")
        summary = pd.DataFrame({
            "Annual Return": returns.mean() * 252,
            "Annual Volatility": returns.std() * np.sqrt(252),
            "Sharpe Ratio": (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252)),
        })
        st.dataframe(summary.style.format("{:.4f}"), width='stretch')
    
    with col2:
        st.subheader("Correlation Matrix")
        corr = returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))
        fig.update_layout(height=400, width=400)
        st.plotly_chart(fig, width='stretch')

# ============================================================================
# TAB 2: Portfolio Optimization
# ============================================================================
with tab2:
    st.subheader("Efficient Frontier Analysis")
    
    # Compute optimizations
    with st.spinner("Optimizing portfolios..."):
        ms_weights, ms_stats = optimize_max_sharpe(returns)
        mv_weights, mv_stats = optimize_min_volatility(returns)
        rand_results, _ = random_portfolios(returns, n_portfolios=n_random_portfolios)
        ef_rets, ef_vols = efficient_frontier(returns, n_points=n_frontier_points)
    
    # Plot efficient frontier
    fig = go.Figure()
    
    # Random portfolios
    fig.add_trace(go.Scatter(
        x=rand_results[:, 1],
        y=rand_results[:, 0],
        mode="markers",
        marker=dict(
            size=4,
            color=rand_results[:, 2],
            colorscale="Viridis",
            colorbar=dict(title="Sharpe Ratio"),
            opacity=0.6
        ),
        name="Random Portfolios",
        hovertemplate="<b>Volatility:</b> %{x:.3f}<br><b>Return:</b> %{y:.3f}<extra></extra>"
    ))
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=ef_vols,
        y=ef_rets,
        mode="lines",
        line=dict(color="red", width=3),
        name="Efficient Frontier",
        hovertemplate="<b>Volatility:</b> %{x:.3f}<br><b>Return:</b> %{y:.3f}<extra></extra>"
    ))
    
    # Max Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[ms_stats[1]],
        y=[ms_stats[0]],
        mode="markers",
        marker=dict(size=15, color="red", symbol="star"),
        name=f"Max Sharpe ({ms_stats[2]:.3f})",
        hovertemplate=f"<b>Max Sharpe</b><br>Volatility: %{{x:.3f}}<br>Return: %{{y:.3f}}<br>Sharpe: {ms_stats[2]:.3f}<extra></extra>"
    ))
    
    # Min Volatility portfolio
    fig.add_trace(go.Scatter(
        x=[mv_stats[1]],
        y=[mv_stats[0]],
        mode="markers",
        marker=dict(size=15, color="blue", symbol="star"),
        name=f"Min Volatility ({mv_stats[2]:.3f})",
        hovertemplate=f"<b>Min Volatility</b><br>Volatility: %{{x:.3f}}<br>Return: %{{y:.3f}}<br>Sharpe: {mv_stats[2]:.3f}<extra></extra>"
    ))
    
    # Individual stocks
    for ticker in returns.columns:
        ann_ret = returns[ticker].mean() * 252
        ann_vol = returns[ticker].std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=[ann_vol],
            y=[ann_ret],
            mode="markers+text",
            marker=dict(size=10, symbol="diamond"),
            text=[ticker],
            textposition="top center",
            name=ticker,
            hovertemplate=f"<b>{ticker}</b><br>Volatility: %{{x:.3f}}<br>Return: %{{y:.3f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Efficient Frontier & Portfolio Optimization",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        hovermode="closest",
        height=600,
        template="plotly_white"
    )
    st.plotly_chart(fig, width='stretch')
    
    # Display optimal weights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Maximum Sharpe Ratio Portfolio")
        st.metric("Expected Return", f"{ms_stats[0]:.2%}")
        st.metric("Volatility", f"{ms_stats[1]:.2%}")
        st.metric("Sharpe Ratio", f"{ms_stats[2]:.4f}")
        
        weights_df = pd.DataFrame({
            "Stock": returns.columns,
            "Weight": ms_weights,
            "% Allocation": ms_weights * 100
        })
        weights_df = weights_df[weights_df["Weight"] > 0.001].sort_values("Weight", ascending=False)
        st.dataframe(weights_df, width='stretch', hide_index=True)
    
    with col2:
        st.subheader("Minimum Volatility Portfolio")
        st.metric("Expected Return", f"{mv_stats[0]:.2%}")
        st.metric("Volatility", f"{mv_stats[1]:.2%}")
        st.metric("Sharpe Ratio", f"{mv_stats[2]:.4f}")
        
        weights_df = pd.DataFrame({
            "Stock": returns.columns,
            "Weight": mv_weights,
            "% Allocation": mv_weights * 100
        })
        weights_df = weights_df[weights_df["Weight"] > 0.001].sort_values("Weight", ascending=False)
        st.dataframe(weights_df, width='stretch', hide_index=True)

# ============================================================================
# TAB 3: Allocation
# ============================================================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Max Sharpe Allocation")
        nonzero_ms = [(t, w) for t, w in zip(returns.columns, ms_weights) if w > 0.01]
        
        fig = go.Figure(data=[go.Pie(
            labels=[t for t, _ in nonzero_ms],
            values=[w for _, w in nonzero_ms],
            textinfo="label+percent",
            textposition="auto",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.4f}<br>Allocation: %{percent}<extra></extra>"
        )])
        fig.update_layout(
            title=f"Max Sharpe (Sharpe={ms_stats[2]:.3f})",
            height=500
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Min Volatility Allocation")
        nonzero_mv = [(t, w) for t, w in zip(returns.columns, mv_weights) if w > 0.01]
        
        fig = go.Figure(data=[go.Pie(
            labels=[t for t, _ in nonzero_mv],
            values=[w for _, w in nonzero_mv],
            textinfo="label+percent",
            textposition="auto",
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.4f}<br>Allocation: %{percent}<extra></extra>"
        )])
        fig.update_layout(
            title=f"Min Volatility (Sharpe={mv_stats[2]:.3f})",
            height=500
        )
        st.plotly_chart(fig, width='stretch')

# ============================================================================
# TAB 4: Backtest
# ============================================================================
with tab4:
    st.subheader("Historical Performance: Optimized vs Equal-Weight")
    
    with st.spinner("Running backtest..."):
        cumulative = backtest_portfolio(prices, ms_weights)
        metrics = performance_metrics(cumulative)
    
    # Cumulative returns chart
    fig = go.Figure()
    
    for col in cumulative.columns:
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col],
            mode="lines",
            name=col,
            hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>"
        ))
    
    fig.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Initial Investment", annotation_position="right")
    
    fig.update_layout(
        title="Portfolio Backtest: Optimized vs Equal-Weight ($10,000 Initial)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, width='stretch')
    
    # Metrics table
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        st.dataframe(metrics, width='stretch')
    
    with col2:
        st.subheader("Key Insights")
        
        optimized_final = cumulative.iloc[-1, 0]
        equal_weight_final = cumulative.iloc[-1, 1]
        
        st.info(f"""
        📊 **Optimized Portfolio**: ${optimized_final:,.2f}  
        📊 **Equal-Weight Portfolio**: ${equal_weight_final:,.2f}  
        📈 **Difference**: ${optimized_final - equal_weight_final:,.2f}
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("📌 Portfolio Optimizer | Modern Portfolio Theory Analysis | Risk-Free Rate: 5.0%")
