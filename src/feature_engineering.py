import pandas as pd
import numpy as np

def compute_RSI(series, period=14):
    """Compute the Relative Strength Index (RSI) for a pandas Series."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df):
    # Ensure the 'Close' column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    # Drop rows where conversion failed
    df.dropna(subset=['Close'], inplace=True)

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_RSI(df['Close'], period=14)
    # Create a return column for the next day and then a binary signal
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Signal'] = (df['Return'] > 0).astype(int)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Read the CSV file, parsing dates and setting the index column
    df = pd.read_csv("data/gold_data.csv", index_col=0, parse_dates=True)
    df = add_indicators(df)
    df.to_csv("data/gold_data_processed.csv")
    print("Gold data processed with technical indicators.")
