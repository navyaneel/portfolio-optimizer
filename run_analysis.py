import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from src.fetch_data import load_prices, load_returns
from src.optimize import (
    optimize_max_sharpe, optimize_min_volatility,
    efficient_frontier, random_portfolios, RISK_FREE_RATE,
)
from src.backtest import backtest_portfolio, performance_metrics

plt.style.use("seaborn-whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 100

prices = load_prices()
returns = load_returns()
print(f"Price data: {prices.shape[0]} trading days, {prices.shape[1]} stocks")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# --- Exploratory ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
normalized = prices / prices.iloc[0] * 100
normalized.plot(ax=axes[0], linewidth=1.5)
axes[0].set_title("Normalized Stock Prices (Base=100)", fontsize=14)
axes[0].set_ylabel("Indexed Price")
axes[0].legend(fontsize=10)

corr = returns.corr()
im = axes[1].imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
axes[1].set_xticks(range(len(corr.columns)))
axes[1].set_yticks(range(len(corr.columns)))
axes[1].set_xticklabels(corr.columns, fontsize=11)
axes[1].set_yticklabels(corr.columns, fontsize=11)
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
axes[1].set_title("Return Correlation Matrix", fontsize=14)
plt.colorbar(im, ax=axes[1], shrink=0.8)
plt.tight_layout()
plt.savefig("data/exploratory.png", bbox_inches="tight")
plt.close()
print("Saved data/exploratory.png")

# --- Individual stock summary ---
summary = pd.DataFrame({
    "Ann. Return": returns.mean() * 252,
    "Ann. Volatility": returns.std() * np.sqrt(252),
    "Sharpe Ratio": (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252)),
})
print("\n=== Individual Stock Summary ===")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

# --- Optimize ---
ms_weights, ms_stats = optimize_max_sharpe(returns)
print("\n=== Maximum Sharpe Ratio Portfolio ===")
print(f"Expected Return:  {ms_stats[0]:.2%}")
print(f"Volatility:       {ms_stats[1]:.2%}")
print(f"Sharpe Ratio:     {ms_stats[2]:.4f}")
print("Weights:")
for t, w in zip(returns.columns, ms_weights):
    print(f"  {t}: {w:.2%}")

mv_weights, mv_stats = optimize_min_volatility(returns)
print("\n=== Minimum Volatility Portfolio ===")
print(f"Expected Return:  {mv_stats[0]:.2%}")
print(f"Volatility:       {mv_stats[1]:.2%}")
print(f"Sharpe Ratio:     {mv_stats[2]:.4f}")
print("Weights:")
for t, w in zip(returns.columns, mv_weights):
    print(f"  {t}: {w:.2%}")

# --- Efficient Frontier ---
np.random.seed(42)
rand_results, _ = random_portfolios(returns, n_portfolios=5000)
ef_rets, ef_vols = efficient_frontier(returns, n_points=100)

fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(rand_results[:, 1], rand_results[:, 0],
                     c=rand_results[:, 2], cmap="viridis", marker="o", s=10, alpha=0.5)
plt.colorbar(scatter, label="Sharpe Ratio", shrink=0.8)
ax.plot(ef_vols, ef_rets, "r-", linewidth=3, label="Efficient Frontier")
ax.scatter(ms_stats[1], ms_stats[0], marker="*", color="red",
           s=500, zorder=5, label=f"Max Sharpe ({ms_stats[2]:.3f})")
ax.scatter(mv_stats[1], mv_stats[0], marker="*", color="blue",
           s=500, zorder=5, label=f"Min Volatility ({mv_stats[2]:.3f})")
for ticker in returns.columns:
    ann_ret = returns[ticker].mean() * 252
    ann_vol = returns[ticker].std() * np.sqrt(252)
    ax.scatter(ann_vol, ann_ret, marker="D", s=100, zorder=5)
    ax.annotate(ticker, (ann_vol, ann_ret), fontsize=11, fontweight="bold",
                xytext=(8, 8), textcoords="offset points")
ax.set_title("Efficient Frontier & Portfolio Optimization", fontsize=16, fontweight="bold")
ax.set_xlabel("Annualized Volatility", fontsize=13)
ax.set_ylabel("Annualized Return", fontsize=13)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/efficient_frontier.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved data/efficient_frontier.png")

# --- Allocation pie ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.Set2(np.linspace(0, 1, len(returns.columns)))
nonzero_ms = [(t, w) for t, w in zip(returns.columns, ms_weights) if w > 0.01]
axes[0].pie([w for _, w in nonzero_ms], labels=[t for t, _ in nonzero_ms],
            autopct="%1.1f%%", colors=colors, startangle=90)
axes[0].set_title(f"Max Sharpe Portfolio\n(Sharpe={ms_stats[2]:.3f})", fontsize=13)
nonzero_mv = [(t, w) for t, w in zip(returns.columns, mv_weights) if w > 0.01]
axes[1].pie([w for _, w in nonzero_mv], labels=[t for t, _ in nonzero_mv],
            autopct="%1.1f%%", colors=colors, startangle=90)
axes[1].set_title(f"Min Volatility Portfolio\n(Sharpe={mv_stats[2]:.3f})", fontsize=13)
plt.tight_layout()
plt.savefig("data/allocation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data/allocation.png")

# --- Backtest ---
cumulative = backtest_portfolio(prices, ms_weights)
metrics = performance_metrics(cumulative)
print("\n=== Backtest Performance Metrics ===")
print(metrics.to_string())

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
cumulative.plot(ax=axes[0], linewidth=2)
axes[0].set_title("Portfolio Backtest: Optimized vs Equal-Weight ($10,000 initial)", fontsize=14)
axes[0].set_ylabel("Portfolio Value ($)", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].axhline(y=10000, color="gray", linestyle="--", alpha=0.5)
for col in cumulative.columns:
    rolling_max = cumulative[col].cummax()
    drawdown = (cumulative[col] - rolling_max) / rolling_max
    axes[1].fill_between(drawdown.index, drawdown, alpha=0.4, label=col)
axes[1].set_title("Drawdown", fontsize=13)
axes[1].set_ylabel("Drawdown", fontsize=12)
axes[1].legend(fontsize=10)
plt.tight_layout()
plt.savefig("data/backtest.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data/backtest.png")

# --- Save weights ---
results = pd.DataFrame({
    "Ticker": returns.columns,
    "Max Sharpe Weight": ms_weights,
    "Min Vol Weight": mv_weights,
})
results.to_csv("data/optimal_weights.csv", index=False)
print("\nSaved data/optimal_weights.csv")

# --- Save results for README ---
readme_data = {
    "ms_sharpe": round(ms_stats[2], 4),
    "ms_return": round(ms_stats[0] * 100, 2),
    "ms_vol": round(ms_stats[1] * 100, 2),
    "mv_sharpe": round(mv_stats[2], 4),
    "mv_return": round(mv_stats[0] * 100, 2),
    "mv_vol": round(mv_stats[1] * 100, 2),
    "ms_weights": {t: round(w * 100, 2) for t, w in zip(returns.columns, ms_weights)},
    "mv_weights": {t: round(w * 100, 2) for t, w in zip(returns.columns, mv_weights)},
    "metrics": metrics.to_dict(),
}
with open("data/results.json", "w") as f:
    json.dump(readme_data, f, indent=2)
print("Saved data/results.json")
print("\nDone!")
