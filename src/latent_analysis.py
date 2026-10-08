from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset import prepare_datasets

from models.gru_forecaster import GRULatentForecaster
from torch_dataset import MarketSequenceDataset
from torch.utils.data import DataLoader
from dataset import load_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'best_gru_model.pt'

INPUT_SIZE = 10
HIDDEN_SIZE = 32
LATENT_SIZE = 3
BATCH_SIZE = 64

#duplicate from evaluate.py
def collect_latent_states(model, data_loader, device):
    model.eval()
    all_targets, all_mu, all_sigma, all_z = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mu, sigma, z = model(X_batch)
            all_targets.append(y_batch.cpu())
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_z.append(z.cpu())

    return (
        torch.cat(all_targets),
        torch.cat(all_mu),
        torch.cat(all_sigma),
        torch.cat(all_z),
    )

def extract_latent_factor(train_z, test_z):
    Z_train = train_z.numpy()
    Z_test = test_z.numpy()

    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    Z_test_scaled = scaler.transform(Z_test)

    pca = PCA()
    Z_train_pca = pca.fit_transform(Z_train_scaled)
    Z_test_pca = pca.transform(Z_test_scaled)

    train_latent_factor = Z_train_pca[:, 0]
    test_latent_factor = Z_test_pca[:, 0]
    explained_variance = pca.explained_variance_ratio_

    return (
        train_latent_factor,
        test_latent_factor,
        explained_variance,
        scaler,
        pca,
    )

def create_temporal_features(features):
    temporal = features.copy()

    tickers = [
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD"
    ]
    for ticker in tickers:
        return_col = f"{ticker}_return"
        vol_col = f"{ticker}_vol_20day"

        #recent returns
        temporal[f"{ticker}_return_5d"] = temporal[return_col].rolling(5).sum()
        temporal[f"{ticker}_return_20d"] = temporal[return_col].rolling(20).sum()

        #recent vol
        temporal[f"{ticker}_vol_change_5d"] = temporal[vol_col] - temporal[vol_col].shift(5)
        temporal[f"{ticker}_vol_change_20d"] = temporal[vol_col] - temporal[vol_col].shift(20)

    return temporal

def compare_static_temporal_reconstruction(
        train_df,
        test_df,
        static_features,
        combined_features,
        max_depth=6
):
    y_train = train_df["latent_factor"]
    y_test = test_df["latent_factor"]

    #static

    X_train_static = train_df[static_features]
    X_test_static = test_df[static_features]

    static_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    static_model.fit(X_train_static, y_train)

    static_train_pred = static_model.predict(
        X_train_static
    )
    static_test_pred = static_model.predict(
        X_test_static
    )

    #static + temporal

    X_train_temporal = train_df[combined_features]
    X_test_temporal = test_df[combined_features]

    temporal_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    temporal_model.fit(X_train_temporal, y_train)

    temporal_train_pred = temporal_model.predict(
        X_train_temporal
    )
    temporal_test_pred = temporal_model.predict(
        X_test_temporal
    )

    #results
    results = {
        "static_train_r2": r2_score(
            y_train,
            static_train_pred,
        ),
        "static_test_r2": r2_score(
            y_test,
            static_test_pred
        ),
        "temporal_train_r2": r2_score(
            y_train,
            temporal_train_pred
        ),
        "temporal_test_r2": r2_score(
            y_test,
            temporal_test_pred
        )
    }
    return results, static_model, temporal_model

def evaluate_feature_sets(
        train_df,
        test_df,
        feature_sets,
        max_depth=6
):
    y_train = train_df["latent_factor"]
    y_test = test_df["latent_factor"]

    results = []

    for name, features in feature_sets.items():

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            train_df[features],
            y_train
        )

        train_pred = model.predict(
            train_df[features]
        )

        test_pred = model.predict(
            test_df[features]
        )

        results.append({
            "model": name,
            "n_features": len(features),
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            )
        })

    return pd.DataFrame(results)

def plot_latent_factor(analysis_df, figures_dir):

    fig, axes = plt.subplots(
        2, 1, figsize = (12, 7), sharex=True)

    #PC1 latent factor

    axes[0].plot(
        analysis_df.index,
        analysis_df["latent_factor"],
        linewidth=1.2
    )
    axes[0].axhline(
        0, linestyle = "--", linewidth=0.8, alpha=0.6
    )

    axes[0].set_ylabel("Latent factor")
    axes[0].set_title("Learned Latent Market State - Test Period")

    axes[0].grid(
        alpha=0.2
    )
    #SPY realized vol

    axes[1].plot(
        analysis_df.index,
        analysis_df["SPY_vol_20day"],
        linewidth=1.2
    )
    axes[1].set_ylabel("SPY 20 day Annualized Volatility")
    axes[1].set_label("Date")
    axes[1].grid(
        alpha=0.2
    )
    fig.tight_layout()

    output_path = Path(figures_dir) / "latent_factor_vs_spy_vol.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    return output_path







if __name__ == "__main__":
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Using {device}')

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_dates, val_dates,
        test_dates, scaler
    ) = prepare_datasets()

    print(f"test X shape: {X_test.shape}")
    print(f"test y shape: {y_test.shape}")

    test_dataset = MarketSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset = MarketSequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False, time to extract latent states

    model = GRULatentForecaster(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    #latent state extraction
    train_targets, train_mu, train_sigma, train_z = collect_latent_states(
        model=model, data_loader=train_loader, device=device
    )
    test_targets, test_mu, test_sigma, test_z = collect_latent_states(
        model=model, data_loader=test_loader, device=device
    )
    print(f"Train Z shape: {train_z.shape}")
    print(f"Test Z shape: {test_z.shape}")

    (
        train_latent_factor, test_latent_factor,
        explained_variance, z_scaler, pca
    ) = extract_latent_factor(train_z=train_z, test_z=test_z)

    print("\nLatent PCA Explained Variance:")
    for i, variance in enumerate(explained_variance, start=1):
        print(f"{i}: {variance:.4f}")

    #chronological DF
    analysis_df = pd.DataFrame({
        "date": test_dates,
        "latent_factor": test_latent_factor,
        "target": test_targets.numpy(),
        "mu": test_mu.numpy(),
        "sigma": test_sigma.numpy(),
    })
    analysis_df = analysis_df.set_index("date")



    train_analysis_df = pd.DataFrame({
        "date": train_dates,
        "latent_factor": train_latent_factor,
        "target": train_targets.numpy(),
        "mu": train_mu.numpy(),
        "sigma": train_sigma.numpy(),
    }).set_index("date")

    print("\nTrain analysis shape:")
    print(train_analysis_df.shape)

    print("\nTest Analysis shape:")
    print(analysis_df.shape)

    print("\nTest Latent Analysis:")
    print(analysis_df.head())
    print(f"\nShape: {analysis_df.shape}")

    #in formalization stage we want latent factor to be learned w/o observing test period

    features = load_features()

    temporal_features_df = create_temporal_features(features)
    train_analysis_df = train_analysis_df.join(temporal_features_df, how="left")
    analysis_df = analysis_df.join(temporal_features_df, how="left")

    static_features = [
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
    tickers = [
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD"
    ]

    temporal_summary_features = []
    for ticker in tickers:
        temporal_summary_features.extend([
            f"{ticker}_return_5d",
            f"{ticker}_return_20d",
            f"{ticker}_vol_change_5d",
            f"{ticker}_vol_change_20d",
        ])

    static_plus_temporal_features = (static_features + temporal_summary_features)

    results, static_model, temporal_model = compare_static_temporal_reconstruction(
        train_df=train_analysis_df,
        test_df=analysis_df,
        static_features=static_features,
        combined_features=static_plus_temporal_features,
        max_depth=6
    )

    print("\nStatic vs Temporal Reconstruction")

    print(f"Static Train r2: {results['static_train_r2']:.4f}")
    print(f"Static Test r2: {results['static_test_r2']:.4f}")
    print(f"Temporal Train r2: {results['temporal_train_r2']:.4f}")
    print(f"Temporal Test r2: {results['temporal_test_r2']:.4f}")
    #explanatory power rises -> temporal test r2 = 0.7706
    #investigate why improvement occured

    temporal_importance = pd.Series(
        temporal_model.feature_importances_,
        index=static_plus_temporal_features
    ).sort_values(ascending=False)

    print("\nTemporal Model Feature Importances:")
    print(temporal_importance)

    return_history_features = []
    vol_change_features = []

    for ticker in tickers:
        return_history_features.extend([
            f"{ticker}_return_5d",
            f"{ticker}_return_20d",
        ])
        vol_change_features.extend([
            f"{ticker}_vol_change_5d",
            f"{ticker}_vol_change_20d",
        ])

    feature_sets = {
        "Static": static_features,
        "Static + Returns":
        static_features + return_history_features,
        "Static + Vol Changes":
        static_features + vol_change_features,
        "Static + All Temporal":
        static_features + return_history_features + vol_change_features
    }

    FIGURES_DIR = PROJECT_ROOT / "figures"
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    figure_path = plot_latent_factor(
        analysis_df=analysis_df,
        figures_dir=FIGURES_DIR
    )
    print(f"\nSaved latent figure to: {figure_path}")

    evaluate_results = evaluate_feature_sets(
        train_df=train_analysis_df,
        test_df=analysis_df,
        feature_sets=feature_sets,
        max_depth=6,
    )
    print("\nEvaluation results")
    print(evaluate_results.to_string(index=False))

    #test PC1 on recent paths

    return_correlations = (
        analysis_df[
            [
                "latent_factor",
                "SPY_return_5d",
                "QQQ_return_5d",
                "IWM_return_5d",
                "SPY_return_20d",
                "QQQ_return_20d",
                "IWM_return_20d",
            ]
        ].corr()["latent_factor"].drop("latent_factor").sort_values()
    )

    print("\nLatent Factor vs Recent Returns:")
    print(return_correlations)

