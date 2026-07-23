import sys
sys.path.insert(0, "/Users/navyaneelamegam/fun_gitrepo/navya/.venv/lib/python3.13/site-packages")

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Market conditions and their stock characteristics
MARKET_CONDITIONS = {
    "bull": {
        "name": "Bull Market",
        "description": "Strong uptrend with low volatility",
        "stocks": {
            "TECH_GROWTH": {"ann_return": 0.45, "ann_vol": 0.28, "start_price": 100},
            "TECH_LARGE": {"ann_return": 0.32, "ann_vol": 0.25, "start_price": 120},
            "CLOUD_SVC": {"ann_return": 0.55, "ann_vol": 0.35, "start_price": 85},
            "EV_MAKER": {"ann_return": 0.50, "ann_vol": 0.40, "start_price": 60},
            "E_COMMERCE": {"ann_return": 0.38, "ann_vol": 0.32, "start_price": 95},
        },
        "correlations": np.array([
            [1.00, 0.75, 0.68, 0.50, 0.62],
            [0.75, 1.00, 0.72, 0.48, 0.70],
            [0.68, 0.72, 1.00, 0.52, 0.65],
            [0.50, 0.48, 0.52, 1.00, 0.45],
            [0.62, 0.70, 0.65, 0.45, 1.00],
        ])
    },
    "bear": {
        "name": "Bear Market",
        "description": "Strong downtrend with high volatility",
        "stocks": {
            "TECH_GROWTH": {"ann_return": -0.25, "ann_vol": 0.45, "start_price": 100},
            "TECH_LARGE": {"ann_return": -0.12, "ann_vol": 0.40, "start_price": 120},
            "CLOUD_SVC": {"ann_return": -0.35, "ann_vol": 0.55, "start_price": 85},
            "EV_MAKER": {"ann_return": -0.40, "ann_vol": 0.60, "start_price": 60},
            "E_COMMERCE": {"ann_return": -0.20, "ann_vol": 0.48, "start_price": 95},
        },
        "correlations": np.array([
            [1.00, 0.82, 0.75, 0.68, 0.72],
            [0.82, 1.00, 0.78, 0.70, 0.75],
            [0.75, 0.78, 1.00, 0.72, 0.78],
            [0.68, 0.70, 0.72, 1.00, 0.65],
            [0.72, 0.75, 0.78, 0.65, 1.00],
        ])
    },
    "sideways": {
        "name": "Sideways Market",
        "description": "Ranging market with moderate volatility",
        "stocks": {
            "TECH_GROWTH": {"ann_return": 0.05, "ann_vol": 0.22, "start_price": 100},
            "TECH_LARGE": {"ann_return": 0.08, "ann_vol": 0.18, "start_price": 120},
            "CLOUD_SVC": {"ann_return": 0.10, "ann_vol": 0.28, "start_price": 85},
            "EV_MAKER": {"ann_return": 0.02, "ann_vol": 0.32, "start_price": 60},
            "E_COMMERCE": {"ann_return": 0.06, "ann_vol": 0.25, "start_price": 95},
        },
        "correlations": np.array([
            [1.00, 0.65, 0.55, 0.35, 0.50],
            [0.65, 1.00, 0.60, 0.40, 0.58],
            [0.55, 0.60, 1.00, 0.45, 0.52],
            [0.35, 0.40, 0.45, 1.00, 0.30],
            [0.50, 0.58, 0.52, 0.30, 1.00],
        ])
    }
}


def generate_synthetic_data(condition="bull", n_days=1566, start_date="2019-01-01", seed=42):
    """
    Generate synthetic OHLCV stock data with realistic characteristics.
    
    Args:
        condition: 'bull', 'bear', or 'sideways' market condition
        n_days: Number of trading days to generate
        start_date: Start date as string (YYYY-MM-DD)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV data indexed by date with MultiIndex (Date, Ticker)
    """
    np.random.seed(seed)
    
    config = MARKET_CONDITIONS[condition]
    tickers = list(config["stocks"].keys())
    
    dates = pd.bdate_range(start=start_date, periods=n_days)
    
    # Parameters for each stock
    daily_vols = np.array([config["stocks"][t]["ann_vol"] / np.sqrt(252) for t in tickers])
    daily_rets = np.array([config["stocks"][t]["ann_return"] / 252 for t in tickers])
    start_prices = np.array([config["stocks"][t]["start_price"] for t in tickers])
    
    # Generate correlated returns
    L = np.linalg.cholesky(config["correlations"])
    uncorrelated = np.random.normal(0, 1, size=(n_days, len(tickers)))
    correlated = uncorrelated @ L.T
    daily_returns = daily_rets + correlated * daily_vols
    
    # Generate OHLCV data
    ohlcv_data = []
    
    for stock_idx, ticker in enumerate(tickers):
        prices = start_prices[stock_idx] * np.cumprod(1 + daily_returns[:, stock_idx])
        
        for day_idx, (date, close_price) in enumerate(zip(dates, prices)):
            # Intraday volatility
            intraday_vol = daily_vols[stock_idx] * 0.8
            
            # Open: slight gap from previous close
            if day_idx == 0:
                open_price = close_price * (1 + np.random.normal(0, intraday_vol * 0.3))
            else:
                prev_close = prices[day_idx - 1]
                open_price = prev_close * (1 + np.random.normal(0, intraday_vol * 0.2))
            
            # High: above both open and close
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_vol * 0.5)))
            
            # Low: below both open and close
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_vol * 0.5)))
            
            # Volume: random with slight trend correlation
            volume = int(np.random.uniform(5_000_000, 80_000_000))
            
            ohlcv_data.append({
                "Date": date,
                "Ticker": ticker,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
                "Adj Close": close_price
            })
    
    df = pd.DataFrame(ohlcv_data)
    df.set_index(["Date", "Ticker"], inplace=True)
    df.sort_index(inplace=True)
    
    return df


def save_synthetic_data(condition="bull", output_dir="data"):
    """Save synthetic data to CSV files (Close prices and Returns)."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = generate_synthetic_data(condition=condition)
    
    # Extract close prices
    prices_df = df["Close"].unstack()
    prices_df.to_csv(os.path.join(output_dir, "prices.csv"))
    
    # Calculate returns
    returns_df = prices_df.pct_change().dropna()
    returns_df.to_csv(os.path.join(output_dir, "returns.csv"))
    
    print(f"✓ Saved synthetic {condition} market data:")
    print(f"  - {len(prices_df)} trading days")
    print(f"  - {len(prices_df.columns)} stocks")
    print(f"  - data/prices.csv")
    print(f"  - data/returns.csv")
    
    return prices_df, returns_df


if __name__ == "__main__":
    # Example usage
    for condition in ["bull", "bear", "sideways"]:
        print(f"\nGenerating {condition} market data...")
        prices, returns = save_synthetic_data(condition=condition)
