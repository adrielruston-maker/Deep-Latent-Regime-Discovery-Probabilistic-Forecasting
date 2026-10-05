from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import StandardScaler
from tensorboard.summary.v1 import scalar

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

def load_raw_data():
    data = {}

    for ticker in TICKERS:
        path = RAW_DATA_DIR / f"{ticker}.csv"

        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
        )
        data[ticker] = df
    return data

if __name__ == "__main__":
    data = load_raw_data()

    for ticker, df in data.items():
        print(f"\n -- {ticker} --")
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        print(df.isna().sum())

#returning dtypes as objects, need to fix in download_data

#MAKE SURE DATES LINE UP sanity check

def check_date_alingment(data):
    reference = data["SPY"].index

    for ticker, df in data.items():
        print(
            f"{ticker}: "
            f"{len(df)} observations: "
            f"{df.index.min()} > {df.index.max()}"
            f"matches with SPY: {df.index.equals(reference)}"
        )
if __name__ == "__main__":
    data = load_raw_data()
    check_date_alingment(data)

#price df

def create_close_prices(data):
    close_prices = pd.concat(
    {
        ticker: df["Close"]
        for ticker, df in data.items()
    },
    axis=1,
    join="inner"
    )
    return close_prices
close_prices = create_close_prices(data)

print(close_prices.head())
print(close_prices.shape)
print(close_prices.isna().sum())

#at time t state Pt = [(P^SPY)^t, ... , (P^GLD)_t]
#normalize with log returns, fits stochastically by nature

def calculate_log_returns(close_prices):
    log_returns = np.log(
        close_prices / close_prices.shift(1)
    )
    return log_returns.dropna()
log_returns = calculate_log_returns(close_prices)

print(log_returns.head())
#(r^(j))_t = log((P^(j))_t / (P^(j))_t-1)
print("\nStats Summary")
print(log_returns.describe())

print("\nCorrelation Matrix")
print(log_returns.corr())

print("\nSkewness")
print(log_returns.skew())

print("\nKurtosis")
print(log_returns.kurt())
print(log_returns.isna().sum()) #sanity check since close prices had no missing vals

#Corr indicates SPY, QQQ, IWM contain substanial common information, however Corr(SPY,TLT) is more interesting
#kurt() evinces heavy kurtosis


def autocorrelation_analysis(log_returns, max_lag=20):
    for ticker in log_returns.columns:
        returns = log_returns[ticker]

        print(f"\n -- {ticker} --")
        print("Return autocorrelation: ")
        for lag in [1, 5, 10 ,20]:
            print(
                f"Lag {lag}: "
                f"{returns.autocorr(lag=lag):.4f}"
            )
        print("Squared Return autocorrelation: ")
        squared_returns = returns**2

        for lag in [1, 5, 10 ,20]:
            print(
                f"Lag {lag}: "
                f"{squared_returns.autocorr(lag=lag):.4f}"
            )
        print("Absolute Return autocorrelation: ")
        absolute_returns = returns.abs()

        for lag in [1, 5, 10 ,20]:
            print(
                f"Lag {lag}: "
                f"{absolute_returns.autocorr(lag=lag):.4f}"
            )
autocorrelation_analysis(log_returns)

#autocorrelation analysis suggests volatility clustering

#realized vol

def calculate_rolling_volatility(
        log_returns,
        window=20,
        annualize=True
):
    rolling_vol = log_returns.rolling(
        window=window,
    ).std()
    if annualize:
        rolling_vol *= np.sqrt(252)
    return rolling_vol

rolling_vol = calculate_rolling_volatility(
    log_returns,
    window=20,
)
print(rolling_vol.head(25))
print(rolling_vol.describe())

plt.figure(figsize=(12, 6))

plt.plot(
    rolling_vol.index,
    rolling_vol["SPY"]
)
plt.title("SPY 20 day annualized rolling volatility")
plt.xlabel("Date")
plt.ylabel("Rolling Volatility")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

#need a feature matrix

def create_feature_matrix(log_returns, rolling_vol):
    returns = log_returns.copy()
    volatility = rolling_vol.copy()

    returns.columns = [
        f"{ticker}_return"
        for ticker in returns.columns
    ]

    volatility.columns = [
        f"{ticker}_vol_20day"
        for ticker in volatility.columns
    ]

    features = pd.concat([returns, volatility], axis=1)
    features = features.dropna()

    return features
features = create_feature_matrix(log_returns, rolling_vol)

print(features.head())
print(features.shape)
print(features.isna().sum())
#temporal split

def temporal_split(features):
    train = features.loc[
        features.index < "2019-01-01"
    ].copy()

    validation = features.loc[
        (features.index >= "2019-01-01") &
        (features.index < "2022-01-01")
    ].copy()

    test = features.loc[
        (features.index >= "2022-01-01")
    ].copy()
    return train, validation, test
train, validation, test = temporal_split(features)
print("Train: ")
print(train.index.min(), train.index.max(), train.shape)

print("\nValidation: ")
print(validation.index.min(), validation.index.max(), validation.shape)

print("\nTest: ")
print(test.index.min(), test.index.max(), test.shape)

def scale_features(train, validation, test):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)
    test_scaled = scaler.transform(test)

    train_scaled = pd.DataFrame(
        train_scaled,
        index=train.index,
        columns=train.columns,
    )
    validation_scaled = pd.DataFrame(
        validation_scaled,
        index=validation.index,
        columns=validation.columns,
    )
    test_scaled = pd.DataFrame(
        test_scaled,
        index=test.index,
        columns=test.columns,
    )
    return train_scaled, validation_scaled, test_scaled, scalar
train_scaled, validation_scaled, test_scaled, scaler = scale_features(train, validation, test)

print("TRAIN")
print(train_scaled.mean())
print(train_scaled.std())

print("\nVALIDATION")
print(validation_scaled.mean())
print(validation_scaled.std())

print("\nTEST")
print(test_scaled.mean())
print(test_scaled.std())

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

features.to_csv(
    PROCESSED_DATA_DIR / "market_features.csv"
)

if __name__ == "__main__":
    data = load_raw_data()

    close_prices = create_close_prices(data)
    log_returns = calculate_log_returns(close_prices)
    rolling_vol = calculate_rolling_volatility(log_returns)

    features = create_feature_matrix(log_returns, rolling_vol)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        PROCESSED_DATA_DIR / "market_features.csv"
    )
    print(
        f"Saved {features.shape[0]} observations "
        f"with {features.shape[1]} features"
    )
