import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date):
    """Download historical data from Yahoo Finance."""
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

if __name__ == "__main__":
    # Example: Gold futures (ticker may vary; 'GC=F' is commonly used)
    gold_data = download_data("GC=F", "2010-01-01", "2023-01-01")
    # Save to CSV if needed
    gold_data.to_csv("data/gold_data.csv")
    print("Gold data downloaded and saved.")