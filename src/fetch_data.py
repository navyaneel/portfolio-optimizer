import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import numpy as np
import pandas as pd
import os

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

STOCK_PARAMS = {
    "AAPL": {"ann_return": 0.35, "ann_vol": 0.32, "start_price": 39.48},
    "MSFT": {"ann_return": 0.32, "ann_vol": 0.30, "start_price": 101.12},
    "GOOG": {"ann_return": 0.25, "ann_vol": 0.31, "start_price": 52.58},
    "AMZN": {"ann_return": 0.22, "ann_vol": 0.35, "start_price": 80.22},
    "TSLA": {"ann_return": 0.55, "ann_vol": 0.60, "start_price": 24.54},
}

CORRELATION_MATRIX = np.array([
    [1.00, 0.72, 0.65, 0.58, 0.42],
    [0.72, 1.00, 0.70, 0.62, 0.38],
    [0.65, 0.70, 1.00, 0.68, 0.40],
    [0.58, 0.62, 0.68, 1.00, 0.35],
    [0.42, 0.38, 0.40, 0.35, 1.00],
])


def fetch_stock_data(tickers=None, start=None, end=None, save=True):
    tickers = tickers or TICKERS
    start = start or START_DATE
    end = end or END_DATE

    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    np.random.seed(42)

    daily_vols = np.array([STOCK_PARAMS[t]["ann_vol"] / np.sqrt(252) for t in tickers])
    daily_rets = np.array([STOCK_PARAMS[t]["ann_return"] / 252 for t in tickers])

    L = np.linalg.cholesky(CORRELATION_MATRIX)
    uncorrelated = np.random.normal(0, 1, size=(n_days, len(tickers)))
    correlated = uncorrelated @ L.T

    daily_returns = daily_rets + correlated * daily_vols

    start_prices = np.array([STOCK_PARAMS[t]["start_price"] for t in tickers])
    cum_returns = np.cumprod(1 + daily_returns, axis=0)
    price_data = start_prices * cum_returns

    prices = pd.DataFrame(price_data, index=dates, columns=tickers)
    prices.index.name = "Date"

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
        returns = prices.pct_change().dropna()
        returns.to_csv(os.path.join(DATA_DIR, "returns.csv"))
        print(f"Saved {len(prices)} rows -> data/prices.csv, data/returns.csv")

    return prices


def load_prices(data_path=None):
    """Load prices from CSV file."""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "prices.csv")
    return pd.read_csv(data_path, index_col=0, parse_dates=True)


def load_returns(data_path=None):
    """Load returns from CSV file."""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, "returns.csv")
    return pd.read_csv(data_path, index_col=0, parse_dates=True)


def fetch_live_data(tickers, start, end):
    """
    Fetch live data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with adjusted close prices
    """
    try:
        import yfinance as yf
        data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
        if len(tickers) == 1:
            data = pd.DataFrame(data)
            data.columns = tickers
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to fetch live data: {e}")


if __name__ == "__main__":
    df = fetch_stock_data()
    print(df.tail())
