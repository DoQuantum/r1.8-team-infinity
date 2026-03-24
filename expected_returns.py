"""
Stage 3: Expected Return Estimation

Layer A: Equilibrium prior (Black-Litterman) with lambda from data.
Layer B: Elastic-Net cross-sectional regression (only — no LightGBM/CatBoost).
Layer C: Black-Litterman posterior with correct He-Litterman Omega.

Key corrections from revised methodology:
- tau = 0.05 (not 0.01)
- Omega = diag(P @ (tau * Sigma) @ P') with P=I => Omega = tau * diag(Sigma)
- Returns Sigma_BL = Sigma + M for downstream use
- Clip at 2.5 sigma (not 3.0)
- Signal scaling to target cross-sectional std
"""
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit

from config import (
    TAU,
    RISK_AVERSION_FALLBACK,
    CLIP_SIGMA,
    ELASTIC_NET_L1_RATIO,
    SAMPLE_WEIGHT_HALFLIFE_MONTHS,
    CV_FOLDS,
    EMBARGO_DAYS,
)


# =====================================================================
# Layer A: Equilibrium Prior
# =====================================================================

def compute_risk_aversion(
    benchmark_excess_return_annual: float,
    benchmark_variance_annual: float,
) -> float:
    """
    Empirical risk aversion: lambda = E[r_m - r_f] / Var(r_m).
    Typical range: 2.5-3.5 for equity portfolios.
    """
    if benchmark_variance_annual <= 0 or np.isnan(benchmark_variance_annual):
        return RISK_AVERSION_FALLBACK
    lam = benchmark_excess_return_annual / benchmark_variance_annual
    if lam <= 0 or lam > 10 or np.isnan(lam):
        return RISK_AVERSION_FALLBACK
    return lam


def equilibrium_prior(
    market_cap: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Equilibrium implied returns: Pi = lambda * Sigma @ w_mkt.
    market_cap: relative to the 30-stock universe.
    Sigma: in monthly units.
    Returns Pi in monthly units.
    """
    w_mkt = market_cap / (market_cap.sum() + 1e-10)
    Pi = lam * Sigma @ w_mkt
    return Pi


# =====================================================================
# Layer B: Elastic-Net ML Alpha Model
# =====================================================================

def train_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    l1_ratio: float = ELASTIC_NET_L1_RATIO,
    n_cv_folds: int = CV_FOLDS,
) -> tuple:
    """
    Train Elastic-Net with time-series cross-validation.
    Returns (predictions, best_alpha, cv_rmse).

    X_train: (n_obs, n_features) pooled panel observations.
    y_train: (n_obs,) forward monthly excess returns.
    X_predict: (n_stocks, n_features) current-period features.
    sample_weights: (n_obs,) exponential decay weights.
    """
    valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    X_v = X_train[valid]
    y_v = y_train[valid]
    sw = sample_weights[valid] if sample_weights is not None else None

    if len(X_v) < 10:
        return np.zeros(X_predict.shape[0]), 1.0, np.nan

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    n_folds = min(n_cv_folds, max(2, len(X_v) // 30))
    tscv = TimeSeriesSplit(n_splits=n_folds, gap=1)

    best_alpha = alphas[0]
    best_score = np.inf

    for alpha in alphas:
        fold_errors = []
        for train_idx, val_idx in tscv.split(X_v):
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               fit_intercept=True, max_iter=5000)
            w_fold = sw[train_idx] if sw is not None else None
            model.fit(X_v[train_idx], y_v[train_idx],
                      sample_weight=w_fold)
            pred = model.predict(X_v[val_idx])
            mse = np.mean((pred - y_v[val_idx]) ** 2)
            fold_errors.append(mse)
        mean_mse = np.mean(fold_errors)
        if mean_mse < best_score:
            best_score = mean_mse
            best_alpha = alpha

    final_model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio,
                             fit_intercept=True, max_iter=5000)
    final_model.fit(X_v, y_v, sample_weight=sw)

    X_pred_clean = np.nan_to_num(X_predict, nan=0.0)
    predictions = final_model.predict(X_pred_clean)
    cv_rmse = np.sqrt(best_score)

    return predictions, best_alpha, cv_rmse


def build_training_data(
    feature_dict: dict,
    monthly_excess_returns: pd.DataFrame,
    rebalance_idx: int,
    feature_dates: list,
    return_dates: list,
    tickers: list,
    halflife_months: int = SAMPLE_WEIGHT_HALFLIFE_MONTHS,
) -> tuple:
    """
    Build pooled panel training data for Elastic-Net.

    feature_dict: {date -> DataFrame(tickers x features)} keyed by daily trading dates.
    monthly_excess_returns: DataFrame indexed by month-end calendar dates.
    rebalance_idx: current rebalance period index.
    feature_dates: daily trading dates for feature lookup (aligned with return_dates).
    return_dates: month-end calendar dates for return lookup.

    Returns (X_train, y_train, sample_weights, X_current).
    Embargo: exclude last month before current rebalance.
    """
    train_end_idx = rebalance_idx - 1
    if train_end_idx < 1:
        return None, None, None, None

    X_rows = []
    y_rows = []
    time_indices = []

    for t_idx in range(0, train_end_idx):
        feat_date = feature_dates[t_idx]
        ret_date = return_dates[t_idx + 1]

        if feat_date not in feature_dict:
            continue

        X_t = feature_dict[feat_date]
        if ret_date not in monthly_excess_returns.index:
            continue

        for ticker in tickers:
            if ticker not in X_t.index:
                continue
            x_row = X_t.loc[ticker].values
            y_val = monthly_excess_returns.loc[ret_date, ticker] \
                if ticker in monthly_excess_returns.columns else np.nan
            X_rows.append(x_row)
            y_rows.append(y_val)
            time_indices.append(t_idx)

    if not X_rows:
        return None, None, None, None

    X_train = np.array(X_rows)
    y_train = np.array(y_rows)
    time_arr = np.array(time_indices)

    decay_rate = np.log(2) / halflife_months
    weights = np.exp(-decay_rate * (train_end_idx - time_arr))
    weights = weights / weights.sum() * len(weights)

    current_feat_date = feature_dates[rebalance_idx]
    X_current = None
    if current_feat_date in feature_dict:
        X_current = feature_dict[current_feat_date].reindex(tickers).values

    return X_train, y_train, weights, X_current


# =====================================================================
# Signal Scaling
# =====================================================================

def scale_signal(
    predictions: np.ndarray,
    target_cross_sectional_std: float = 0.005,
) -> np.ndarray:
    """
    Scale ML predictions to target cross-sectional standard deviation.
    Preserves ranking, controls magnitude for BL integration.
    target_cross_sectional_std: 0.5% per month.
    """
    pred_mean = np.nanmean(predictions)
    pred_std = np.nanstd(predictions)
    if pred_std < 1e-10:
        return predictions
    scaled = pred_mean + (predictions - pred_mean) * (
        target_cross_sectional_std / pred_std)
    return scaled


# =====================================================================
# Layer C: Black-Litterman Posterior
# =====================================================================

def black_litterman_posterior(
    Pi: np.ndarray,
    q: np.ndarray,
    Sigma: np.ndarray,
    tau: float = TAU,
    Omega: Optional[np.ndarray] = None,
) -> tuple:
    """
    Black-Litterman posterior with stock-level views (P = I).

    mu_BL = [(tau*Sigma)^-1 + Omega^-1]^-1 @ [(tau*Sigma)^-1 @ Pi + Omega^-1 @ q]
    Sigma_BL = Sigma + M  where M = [(tau*Sigma)^-1 + Omega^-1]^-1

    If Omega is None, uses He-Litterman proportional method:
        Omega = diag(tau * Sigma)  (since P = I)

    Returns (mu_BL, Sigma_BL).
    """
    N = len(Pi)
    reg = np.eye(N) * 1e-8

    if Omega is None:
        Omega = np.diag(np.diag(tau * Sigma))

    tau_Sigma = tau * Sigma
    tau_Sigma_inv = np.linalg.inv(tau_Sigma + reg)
    Omega_inv = np.linalg.inv(Omega + reg)

    M = np.linalg.inv(tau_Sigma_inv + Omega_inv)
    mu_BL = M @ (tau_Sigma_inv @ Pi + Omega_inv @ q)

    Sigma_BL = Sigma + M

    return mu_BL, Sigma_BL


# =====================================================================
# Post-Processing
# =====================================================================

def post_process_mu(
    mu: np.ndarray,
    clip_sigma: float = CLIP_SIGMA,
) -> np.ndarray:
    """Outlier clipping at ±clip_sigma standard deviations."""
    mu_bar = np.nanmean(mu)
    sigma_mu = np.nanstd(mu) + 1e-10
    lo = mu_bar - clip_sigma * sigma_mu
    hi = mu_bar + clip_sigma * sigma_mu
    return np.clip(mu, lo, hi)


# =====================================================================
# Information Coefficient
# =====================================================================

def compute_ic(
    predicted: np.ndarray,
    realized: np.ndarray,
) -> float:
    """
    Spearman rank IC between predicted and realized returns.
    Returns correlation coefficient (IC).
    """
    valid = ~(np.isnan(predicted) | np.isnan(realized))
    if valid.sum() < 5:
        return np.nan
    corr, _ = spearmanr(predicted[valid], realized[valid])
    return corr


def rolling_ic(
    predicted_series: dict,
    realized_series: dict,
) -> pd.Series:
    """
    Compute IC at each rebalancing date.
    predicted_series: {date: array of predictions}
    realized_series: {date: array of realized returns}
    Returns Series of IC values indexed by date.
    """
    ics = {}
    for date in predicted_series:
        if date in realized_series:
            ic = compute_ic(predicted_series[date], realized_series[date])
            ics[date] = ic
    return pd.Series(ics)
