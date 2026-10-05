from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""
dataset.py 
move/load the market_features csv
fit for training inputs
construct a SPY target
construct Xt (epsilon R^60x10)
PyTorch dataset
"""

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

FEATURE_COLUMNS = [
    "SPY_return",
    "QQQ_return",
    "IWM_return",
    "TLT_return",
    "GLD_return",
    "SPY_vol_20day",
    "QQQ_vol_20day",
    "IWM_vol_20day",
    "TLT_vol_20day",
    "GLD_vol_20day"
]
def load_features():
    features = pd.read_csv(
        PROCESSED_DATA_DIR / 'market_features.csv',
        index_col="Date",
        parse_dates=["Date"],
    )
    return features
features = load_features()

#5 day forward target
def create_forward_target(features, horizon=5):
    target = sum(
        features["SPY_return"].shift(-i)
        for i in range(1, horizon + 1)
    )
    #prevent leakage

    target_end_date = pd.Series(
        features.index,
        index=features.index
    ).shift(-horizon)
    return (target.rename(f"SPY_forward_{horizon}d_return"),
target_end_date.rename("target_end_date"))

target, target_end_date = create_forward_target(
features, horizon=5)

dataset = features.copy()
dataset["target"] = target
dataset["target_end_date"] = target_end_date
print(dataset.tail(10))

def temporal_split(dataset):
    train = dataset.loc[
        (dataset.index < "2019-01-01") &
        (dataset["target_end_date"] < "2019-01-01")
    ].copy()

    validation = dataset.loc[
        (dataset.index >= "2019-01-01") &
        (dataset.index < "2022-01-01") &
        (dataset["target_end_date"] < "2022-01-01")
    ].copy()

    test = dataset.loc[
        (dataset.index >= "2022-01-01") &
        dataset["target"].notna()
    ].copy()
    return train, validation, test

train, validation, test = temporal_split(dataset)

print("Train")
print(train.index.min(), train.index.max())
print(train["target_end_date"].max())
print(train.shape)

print("\nValidation")
print(validation.index.min(), validation.index.max())
print(validation["target_end_date"].max())
print(validation.shape)

print("\nTest")
print(test.index.min(), test.index.max())
print(test["target_end_date"].max())
print(test.shape)

#scale the features with sklearn
def scale_features(features, train):
    scaler = StandardScaler()
    # learn stats refactor
    scaler.fit(train[FEATURE_COLUMNS])
    X_scaled = scaler.transform(features[FEATURE_COLUMNS])
    X_scaled = pd.DataFrame(
        X_scaled,
        index=features.index,
        columns=FEATURE_COLUMNS
    )
    return X_scaled, scaler

#60 day seq

def create_sequences(
        X_scaled,
        dataset,
        target_dates,
        lookback=60
):
    X_sequences = []
    y_targets = []
    sequence_dates = []

    for target_date in target_dates:
        i = X_scaled.index.get_loc(target_date)
        #get 60 obs
        start_i = i - lookback + 1

        if start_i < 0:
            continue
        X_sequence = X_scaled.iloc[
            start_i:i + 1
        ].to_numpy()

        y = dataset.loc[
            target_date,
            "target"
        ]

        X_sequences.append(X_sequence)
        y_targets.append(y)
        sequence_dates.append(target_date)
    return (
        np.array(X_sequences),
        np.array(y_targets),
        np.array(sequence_dates)
    )

def prepare_datasets(lookback=60, horizon=5):
    features = load_features()

    target, target_end_date = create_forward_target(features, horizon=5)
    dataset = features.copy()
    dataset["target"] = target
    dataset["target_end_date"] = target_end_date

    train, validation, test = temporal_split(dataset)

    X_scaled, scaler = scale_features(features, train)

    X_train, y_train, train_dates = create_sequences(
        X_scaled,
        dataset,
        train.index,
        lookback=lookback
    )


#shape is correct

    X_val, y_val, val_dates = create_sequences(
        X_scaled,
        dataset,
        validation.index,
        lookback=lookback
    )
    X_test, y_test, test_dates = create_sequences(
        X_scaled,
        dataset,
        test.index,
        lookback=lookback
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_dates,
        val_dates,
        test_dates,
        scaler
    )
(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    train_dates, val_dates,
    test_dates, scaler
) = prepare_datasets()



if __name__ == "__main__":
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_dates, val_dates,
        test_dates, scaler
    ) = prepare_datasets()



#important for regime detection, need to refactor, refactor done
#we shouldnt deprive the latent variable z_t of the recent dynamics that it should learn
#hence, splitting the data should not determine which observations are available
# natural filtration F(X_t-59 ... Xt,) -> GRU -> latent variable


