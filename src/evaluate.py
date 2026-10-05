import torch
import numpy as np
from pathlib import Path
from models.gru_forecaster import GRULatentForecaster
from torch_dataset import create_dataloaders
from loss import gaussian_nll
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataset import load_features
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

"""
remember that train_loader -> shuffle = True, Valid/Test -> Shuffle = False
create a collection and also metrics(NLL, MAE, RMSE)
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_gru_model.pt"
def collect_predictions(
        model,
        data_loader,
        device
):
    model.eval()
    #setup batch collection
    all_targets = []
    all_mu = []
    all_sigma = []
    all_z = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            mu, sigma, z = model(X_batch)
            #mps good for model because of GPU but bring back to CPU for collection
            all_targets.append(y_batch.cpu())
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_z.append(z.cpu())

        targets = torch.cat(all_targets)
        mu = torch.cat(all_mu)
        sigma = torch.cat(all_sigma)
        z = torch.cat(all_z)

        return targets, mu, sigma, z

if __name__ == '__main__':
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = GRULatentForecaster(
        input_size=10,
        hidden_size=32,
        latent_size=3
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    (
        train_loader, val_loader,
        test_loader, train_dates,
        val_dates, test_dates, scaler
    ) = create_dataloaders(
        batch_size=64,
        lookback=60,
        horizon=5
    )

    targets, mu, sigma, z = collect_predictions(
        model,
        test_loader,
        device
    )
# sanity check -> came back consistent
    #print("\nEval Shapes:")
    #print("Targets:", targets.shape)
    #print("Mu:", mu.shape)
    #print("Sigma:", sigma.shape)
    #print("Z:", z.shape)
    #print("Dates:", len(test_dates))

    #print("\nSigma Range:")
    #print("Min:", sigma.min().item())
    #print("Max:", sigma.max().item())

    #print("\nFinite Check:")
    #print("Targets:", torch.isfinite(targets).all().item())
    #print("Mu:", torch.isfinite(mu).all().item())
    #print("Sigma:", torch.isfinite(sigma).all().item())
    #print("Z:", torch.isfinite(z).all().item())

def calculate_metrics(
        targets,
        mu,
        sigma
):
    nll = gaussian_nll(
        mu,
        sigma,
        targets
    ).item()

    errors = targets - mu

    mae = torch.mean(
        torch.abs(errors)
    ).item()

    rmse = torch.sqrt(
        torch.mean(errors**2)
    ).item()

    return {
        "nll": nll,
        "mae": mae,
        "rmse": rmse
    }

metrics = calculate_metrics(
    targets,
    mu,
    sigma
)

print("\nTest Metrics:")
print(f"nll: {metrics['nll']:.6f}")
print(f"mae: {metrics['mae']:.6f}")
print(f"rmse: {metrics['rmse']:.6f}")

#recall standardized residual = (obs - expected) / sqrt(expected)

def standardized_residuals(
        targets,
        mu,
        sigma
):
    return (targets - mu) / sigma
std_residuals = standardized_residuals(
    targets,
    mu,
    sigma
)

print("\nStandardized Residuals:")
print(f"Mean: {std_residuals.mean().item():.4f}")
print(f" Std: {std_residuals.std(unbiased=False).item():.4f}")

#if predictions are gaussian => expect E[e_t] around 0 and Std(e_t) around 1
#dont expect 0 and 1 respectively, financial returns tend to have excess kurtosis

def interval_convergence(
        targets,
        mu,
        sigma,
        z_score
):
    upper_ci = mu + (sigma * z_score)
    lower_ci = mu - (sigma * z_score)

    covered = (
        (targets >= lower_ci) & (targets <= upper_ci)
    )
    return covered.float().mean().item()

coverage_68 = interval_convergence(targets, mu, sigma, 1.0)
coverage_90 = interval_convergence(targets, mu, sigma, 1.645)
coverage_95 = interval_convergence(targets, mu, sigma, 1.96)

print("\nInterval Convergence:")
print(f"68%: , {coverage_68:.3f}")
print(f"90%: , {coverage_90:.3f}")
print(f"95%: , {coverage_95:.3f}")

# 68%, and 95% are consistent, 90% is too high -> model may be too conservative

print("Sigma min", sigma.min().item())

def calculate_bias(
        targets,
        mu
):
    errors = targets - mu
    return {
        "mean_error": errors.mean().item(),
        "mean_target": targets.mean().item(),
        "mean_prediction": mu.mean().item(),
    }

bias = calculate_bias(targets, mu)

print("\nMean Forecast Diagnostics:")
print( f"Mean target: {bias['mean_target']:.6f}""]}"
)
print(f" Mean Prediction: {bias['mean_prediction']:.6f}")
print(f" Mean error: {bias['mean_error']:.6f}")

def plot_standardized_residuals(
        std_residuals
):
    residuals = std_residuals.numpy()

    x = np.linspace(-4, 4, 500)

    standard_normal_pdf = (
        1 / np.sqrt(2*np.pi)
        * np.exp(-0.5 * x**2)
    )
    plt.figure(figsize=(8, 5))

    plt.hist(
        residuals,
        bins=40,
        density=True,
        alpha=0.6,
        label="Standardized Residuals"
    )
    plt.plot(
        x,
        standard_normal_pdf,
        label="Standard Normal"
    )

    plt.xlabel("Standardized Residuals")
    plt.ylabel("Density")
    plt.title(
        "Distribution of Standardized Residuals"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

resid_plot = plot_standardized_residuals(std_residuals)
#plot indicates skew, test for skewness and kurt

def residual_shape_diagnostics(std_residuals):
    residuals = std_residuals.numpy()

    return {
        "skewness": skew(residuals, bias=False), "excess_kurtosis": kurtosis(residuals, fisher=True, bias=False),
    }

shape = residual_shape_diagnostics(std_residuals)

print("\nStandard Residual Shape")
print(
    f"Skewness: {shape['skewness']:.4f}"
)
print(
    f"Kurtosis: {shape['excess_kurtosis']:.4f}"
)
#skew result evinces negative skew, slightly leptokurtic.

def calculate_calibration_curve(
        targets,
        mu,
        simga
):
    nominal_levels = np.arange(
        0.10,
        1.00,
        0.05
    )
    empirical_coverage = []
    # -> q = phi^-1( 1 + level / 2)
    for level in nominal_levels:
        z_score = norm.ppf(
            ((1 + level) / 2)
        )

        coverage = interval_convergence(
            targets,
            mu,
            sigma,
            z_score
        )
        empirical_coverage.append(coverage)

    return (
        nominal_levels,
        np.array(empirical_coverage)
    )

def plot_calibration(
        *,
        nominal,
        empirical
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        nominal,
        empirical,
        marker="o",
        label="GRU Forecast"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.xlabel("Nominal Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title("Prediction Interval Calibration")

    plt.legend()
    plt.tight_layout()
    plt.show()

nominal, empirical = calculate_calibration_curve(
    targets,
    mu,
    sigma
)

plot_calibration(
    nominal=nominal, empirical=empirical
)
"""
up to around 68% the empirical curve stays very close to ideal diagonal,
hence predicted sigma is doing a good job describing the uncertainy

after around 68% it is evident that P(Yt in I) > P_nominal => conservative
no evidence of severe miscalibration
"""



latent_df = pd.DataFrame(
    {
        "date": test_dates,
        "z1": z[:, 0].numpy(),
        "z2": z[:, 1].numpy(),
        "z3": z[:, 2].numpy(),
        "target": targets.numpy(),
        "mu": mu.numpy(),
        "sigma": sigma.numpy(),
    }
)

latent_df = latent_df.set_index("date")
#sanity check
print("\nLatent DF: ")
print(latent_df.head())
print(latent_df.shape)
print(latent_df.describe())

def plot_latent_states(latent_df):
    latent_columns = ["z1", "z2", "z3"]

    for column in latent_columns:
        plt.figure(figsize=(10,5))

        plt.plot(
            latent_df.index,
            latent_df[column]
        )

        plt.xlabel("Date")
        plt.ylabel(column)
        plt.title(
            f"Latent State {column.upper()} Over Time"
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

plot_latent_states(latent_df=latent_df)

#evaluate the graphs to analyze latent representation, but also quanitify it

def latent_autocorrelation(
        latent_df,
        lags=(1, 5 ,10, 20)
):
    for column in ["z1", "z2", "z3"]:
        print(f"\n{column.upper()} Autocorrelation: ")

        for lag in lags:
            autocorr = latent_df[column].autocorr(
                lag=lag
            )
            print(
                f"Lag {lag:2d}:  {autocorr:.4f}"
            )
latent_autocorrelation(latent_df=latent_df)

latent_corr = latent_df[["z1", "z2", "z3"]].corr()
print("\nLatent Correlation Matrix:")
print(latent_corr)
#results from looking at the plots and the correlation matrix is consistent, z1 looks like a negative to z2,z3 and z2,z3 are highly corr

def plot_latent_relationship(
        latent_df,
        x_col,
        y_col
):
    plt.figure(figsize=(6,6))

    plt.scatter(
        latent_df[x_col],
        latent_df[y_col],
        alpha=0.5,
        s=15
    )

    plt.xlabel(x_col.upper())
    plt.ylabel(y_col.upper())
    plt.title(f"{x_col.upper()} vs {y_col.upper()}")\

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
plot_latent_relationship(
    latent_df=latent_df,
    x_col="z1",
    y_col="z2"
)
plot_latent_relationship(
    latent_df=latent_df,
    x_col="z1",
    y_col="z3"
)
plot_latent_relationship(
    latent_df=latent_df,
    x_col="z2",
    y_col="z3"
)


#PCA to create a bottlenck so evaluate how many directions of variance the 3d repres. has

def latent_pca(latent_df):
    Z = latent_df[["z1", "z2", "z3"]].to_numpy()

    scaler = StandardScaler()
    Z_scaled = StandardScaler().fit_transform(Z)

    pca = PCA()
    Z_pca = pca.fit_transform(Z_scaled)

    print("\nLatent PCA Explained Variance: ")

    for i, ratio in enumerate(
        pca.explained_variance_ratio_,
        start=1
    ):
        print(f"{i}: {ratio:.2f}")
    return pca, scaler, Z_pca

pca, latent_scaler, Z_pca = latent_pca(
    latent_df=latent_df
)
latent_df["latent_factor"] = Z_pca[:, 0]

#PCA shows that the 3D directional variance strongly collapses onto a 1d manifold

#after PCA its more likely that X_t-59:t -> s epsilon R -> z epsilon R3 -> (mu, sigma)
#claim for model hence is that it found a 1-D representation for predicting 5-day SPY distribution

latent_sigma_corr = latent_df[
    ["latent_factor", "sigma"]
].corr()

print("\nLatent Factor vs Predicted Sigma")
print(latent_sigma_corr)

def plot_latent_vs_sigma(latent_df):
    plt.figure(figsize=(8,6))

    plt.scatter(
        latent_df["latent_factor"],
        latent_df["sigma"],
        alpha=0.5,
        s=15
    )

    plt.xlabel("Latent Factor (PC1)")
    plt.ylabel("Predicted Sigma")
    plt.title(
        "Latent Factor vs Predicted Uncertainty"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_latent_vs_sigma(latent_df=latent_df)

features = load_features()

latent_df["SPY_vol_20day"] = features.loc[
    latent_df.index,
    "SPY_vol_20day"
]

print("\nLatent Factor vs SPY Realized Vol")
print(latent_df[["latent_factor", "SPY_vol_20day"]].corr())

def plot_latent_vs_realized_vol(latent_df):
    plt.figure(figsize=(8,6))

    plt.scatter(
        latent_df["latent_factor"],
        latent_df["SPY_vol_20day"],
        alpha=0.5,
        s=15
    )
    plt.xlabel("Latent Factor (PC1)")
    plt.ylabel("SPY Realized Vol")
    plt.title(
        "Latent Factor vs Observed SPY Vol"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_latent_vs_realized_vol(latent_df=latent_df)
#Now we look at the whole feature state

analysis_df = latent_df.join(
    features.loc[
        latent_df.index,
        [
            "SPY_return",
            "QQQ_return",
            "IWM_return",
            "TLT_return",
            "GLD_return",
            "QQQ_vol_20day",
            "IWM_vol_20day",
            "TLT_vol_20day",
            "GLD_vol_20day",
        ]
    ]
)

feature_correlations = analysis_df.corr()["latent_factor"].drop(
    [
        "latent_factor",
        "target",
        "mu",
        "sigma",
        "z1",
        "z2",
        "z3"
    ],
    errors="ignore"
).sort_values(
    key=abs,
    ascending=False
)

print("\nObservable Feature Correlations with Latent Factor: ")
print(feature_correlations)

#corr alludes to volatility dominated latent factor
#keep in mind PC1 solely is not a strong enough comparison with SPY 20 day vol
#Move to find R2 to check explanatory power

def explain_latent_factor(analysis_df):
    feature_columns = [
        "SPY_return",
        "QQQ_return",
        "IWM_return",
        "TLT_return",
        "GLD_return",
        "SPY_vol_20day",
        "QQQ_vol_20day",
        "IWM_vol_20day",
        "TLT_vol_20day",
        "GLD_vol_20day",

    ]
    X = analysis_df[feature_columns]
    y = analysis_df["latent_factor"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)

    return r2

latent_r2 = explain_latent_factor(
    analysis_df=analysis_df
)
print(f"\nLatent Factor Linear R2 {latent_r2:.4f}")
#the 10 features explain about 77.35% of the variance in the learned scalar latent factor
#next step is to make R2 out of sample
#change the regularization with Ridge Regression as a comparison
# recall from Ridge (sklearn) min[||y - Xw||^2_2 + alpha * ||w||^2_2

def explain_latent_factor_oos(analysis_df, feature_columns):
    data = analysis_df[
        feature_columns + ["latent_factor"]
    ].dropna()

    split_index = int(0.70 * len(data))

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["latent_factor"]

    X_test = test[feature_columns]
    y_test = test["latent_factor"]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train

    )
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2 = r2_score(
        y_train,
        train_predictions
    )
    test_r2 = r2_score(
        y_test,
        test_predictions
    )
    return train_r2, test_r2

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
    "GLD_vol_20day",
]

train_r2, test_r2 = explain_latent_factor_oos(
    analysis_df=analysis_df,
    feature_columns=static_features
)

print(f"Train R2: {train_r2:.4f}")
print(f"Test_OOS R2: {test_r2:.4f}")

def explain_latent_factor_ridge_oos(
        analysis_df, feature_columns, alpha=0.1
):
    data = analysis_df[
        feature_columns + ["latent_factor"]
    ].dropna()

    split_index = int(0.70 * len(data))

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["latent_factor"]

    X_test = test[feature_columns]
    y_test = test["latent_factor"]

    model = Pipeline([
        (
            "scaler", StandardScaler(),
        ),
        (
            "ridge", Ridge(alpha=alpha)
        )
    ])

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_ridge_r2 = r2_score(
        y_train,
        train_predictions
    )
    test_ridge_r2 = r2_score(
        y_test,
        test_predictions
    )
    return train_ridge_r2, test_ridge_r2

ridge_train_r2, ridge_test_r2 = explain_latent_factor_ridge_oos(
    analysis_df=analysis_df,
    feature_columns=static_features,
    alpha=1.0
)

print("\nRidge Regression:")
print(f"Train R2: {ridge_train_r2:.4f}")
print(f"Test R2: {ridge_test_r2:.4f}")

#both give an explanatory power of 0.67, now dive into a non linear model -> Random Forest

def explain_latent_factor_rf_oos(
        analysis_df,
        feature_columns,
        depths=(3, 4, 5, 6, 7, 8),
):
    data = analysis_df[
        feature_columns + ["latent_factor"]
    ].dropna()

    # Chronological 70/30 split
    split_index = int(0.70 * len(data))

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["latent_factor"]

    X_test = test[feature_columns]
    y_test = test["latent_factor"]

    results = []

    for depth in depths:

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_r2 = r2_score(
            y_train,
            train_predictions
        )

        test_r2 = r2_score(
            y_test,
            test_predictions
        )

        results.append({
            "depth": depth,
            "train_r2": train_r2,
            "test_r2": test_r2,
        })

        print(
            f"Depth {depth:2d} | "
            f"Train R2: {train_r2:.4f} | "
            f"OOS R2: {test_r2:.4f}"
        )

    return pd.DataFrame(results)

rf_results = explain_latent_factor_rf_oos(
    analysis_df=analysis_df,
    feature_columns=static_features
)
print("\nRandom Forest Anaysis")
print(rf_results)
#Evidence after RF alludes to nonlinear structure

print("\nStatic Feature Analysis")
print(f"OLS OOS R2 {test_r2:.4f}")
print(f"Ridge OOS R2 {ridge_test_r2:.4f}")

print("\nRandom Forest:")
for _, row in rf_results.iterrows():
    print(
        f"Depth {int(row['depth'])}: "
        f"Train={row['train_r2']:.4f}, "
        f"OOS={row['test_r2']:.4f}"
    )




