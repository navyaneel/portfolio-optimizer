import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05


def portfolio_stats(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns) * TRADING_DAYS
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * TRADING_DAYS, weights)))
    sharpe = (ret - RISK_FREE_RATE) / vol
    return ret, vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]


def optimize_max_sharpe(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    w0 = np.array([1.0 / n] * n)

    result = minimize(neg_sharpe, w0, args=args, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x, portfolio_stats(result.x, mean_returns, cov_matrix)


def optimize_min_volatility(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    w0 = np.array([1.0 / n] * n)

    result = minimize(portfolio_volatility, w0, args=args, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x, portfolio_stats(result.x, mean_returns, cov_matrix)


def efficient_frontier(returns, n_points=100):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)

    _, min_stats = optimize_min_volatility(returns)
    _, max_stats = optimize_max_sharpe(returns)

    target_returns = np.linspace(min_stats[0] - 0.02, max_stats[0] + 0.10, n_points)
    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) * TRADING_DAYS - target},
        ]
        bounds = tuple((0, 1) for _ in range(n))
        w0 = np.array([1.0 / n] * n)
        result = minimize(portfolio_volatility, w0,
                          args=(mean_returns, cov_matrix),
                          method="SLSQP", bounds=bounds,
                          constraints=constraints)
        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)

    return np.array(frontier_rets), np.array(frontier_vols)


def random_portfolios(returns, n_portfolios=5000):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    results = np.zeros((n_portfolios, 3))
    weights_record = []

    for i in range(n_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix)
        results[i] = [ret, vol, sharpe]
        weights_record.append(w)

    return results, weights_record
