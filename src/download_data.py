import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

#nicer format than yfinance(...) for every one, if you want to add more tickers
for ticker in TICKERS:

    df = yf.download(
        ticker,
        start="2005-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=True,
    )

    #dtype fix
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(RAW_DATA_DIR / f"{ticker}.csv")
    print(f"{ticker}: {df.shape}")
