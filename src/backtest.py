import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import numpy as np
import pandas as pd


def backtest_portfolio(prices, weights, initial_capital=10000):
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)

    cumulative = (1 + portfolio_returns).cumprod() * initial_capital
    cumulative.name = "Portfolio"

    equal_w = np.ones(len(weights)) / len(weights)
    equal_returns = returns.dot(equal_w)
    equal_cumulative = (1 + equal_returns).cumprod() * initial_capital
    equal_cumulative.name = "Equal Weight"

    return pd.DataFrame({
        "Optimized": cumulative,
        "Equal Weight": equal_cumulative,
    })


def performance_metrics(cumulative_df, risk_free_rate=0.05):
    metrics = {}
    for col in cumulative_df.columns:
        series = cumulative_df[col]
        daily_returns = series.pct_change().dropna()
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        n_years = len(daily_returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / ann_vol

        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        metrics[col] = {
            "Total Return": f"{total_return:.2%}",
            "Annualized Return": f"{ann_return:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
        }
    return pd.DataFrame(metrics)
