"""
Stage 4: Covariance Matrix Estimation

Primary: Ledoit-Wolf shrinkage on non-overlapping monthly returns.
Validation: PCA factor structure check.
Advanced: Regime-conditional blend with VIX > 25 threshold.
Baseline: Sample covariance for comparison.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from config import VIX_STRESS_THRESHOLD


# =====================================================================
# Monthly Return Computation
# =====================================================================

def compute_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute non-overlapping monthly returns from daily prices.
    Uses last trading day of each month.
    """
    monthly_prices = daily_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


# =====================================================================
# Primary: Ledoit-Wolf on Monthly Returns
# =====================================================================

def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple:
    """
    Ledoit-Wolf shrinkage with constant-correlation target.
    Input: monthly returns DataFrame.
    Returns (Sigma, shrinkage_coef).
    Sigma is in monthly return units.
    """
    R = returns.dropna().values
    if R.shape[0] < 2:
        n = R.shape[1]
        return np.eye(n) * 0.001, 1.0

    lw = LedoitWolf()
    lw.fit(R)
    return lw.covariance_, lw.shrinkage_


# =====================================================================
# Baseline: Sample Covariance
# =====================================================================

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance matrix (baseline for comparison)."""
    R = returns.dropna().values
    if R.shape[0] < 2:
        return np.eye(R.shape[1]) * 0.001
    return np.cov(R, rowvar=False)


# =====================================================================
# PCA Validation
# =====================================================================

def pca_validation(returns: pd.DataFrame, n_components: int = 5) -> dict:
    """
    PCA decomposition to validate covariance structure.
    Checks whether first 3-5 components explain >50% of variance.
    Returns explained variance ratios and diagnostics.
    """
    R = returns.dropna().values
    n_obs, n_assets = R.shape
    n_comp = min(n_components, n_assets - 1, n_obs - 1)

    pca = PCA(n_components=n_comp)
    pca.fit(R)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    has_factor_structure = cumulative[min(4, n_comp - 1)] > 0.50

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumulative,
        "n_components_for_80pct": int(np.searchsorted(cumulative, 0.80) + 1),
        "has_factor_structure": has_factor_structure,
        "recommendation": (
            "Ledoit-Wolf captures real structure"
            if has_factor_structure
            else "Consider explicit factor model (Fama-French 5)"
        ),
    }


# =====================================================================
# Factor-Based Covariance (for comparison/validation)
# =====================================================================

def factor_covariance(
    returns: pd.DataFrame,
    n_factors: int = 5,
) -> tuple:
    """
    Factor-based decomposition: Σ = B F B' + D.
    Returns (Sigma, loadings_B, idio_D).
    """
    R = returns.dropna().values
    n_obs, n_assets = R.shape
    n_factors = min(n_factors, n_assets - 1, n_obs - 1)

    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(R)
    B = pca.components_.T  # (n_assets, n_factors)
    F_cov = np.cov(factors, rowvar=False)
    if F_cov.ndim == 0:
        F_cov = np.array([[F_cov]])
    resid = R - factors @ B.T
    D = np.diag(np.var(resid, axis=0) + 1e-8)
    Sigma = B @ F_cov @ B.T + D
    return Sigma, B, D


# =====================================================================
# Regime-Conditional Covariance
# =====================================================================

def regime_conditional_covariance(
    monthly_returns: pd.DataFrame,
    vix_series: pd.Series,
    stress_threshold: float = VIX_STRESS_THRESHOLD,
) -> tuple:
    """
    Simple regime-based covariance blending:
    - Normal regime: VIX <= threshold
    - Stress regime: VIX > threshold
    - Blend based on current VIX level

    vix_series: daily VIX levels (will be resampled to monthly).
    Returns (Sigma_blended, p_stress, Sigma_normal, Sigma_stress).
    """
    vix_monthly = vix_series.resample("ME").last().reindex(
        monthly_returns.index, method="ffill")

    R = monthly_returns.dropna()
    vix_aligned = vix_monthly.reindex(R.index, method="ffill").fillna(
        vix_monthly.median() if len(vix_monthly) > 0 else 20.0)

    stress_mask = vix_aligned > stress_threshold
    normal_mask = ~stress_mask

    lw_normal = LedoitWolf()
    lw_stress = LedoitWolf()

    R_vals = R.values
    n = R_vals.shape[1]

    if normal_mask.sum() >= max(n + 1, 5):
        lw_normal.fit(R_vals[normal_mask.values])
        Sigma_normal = lw_normal.covariance_
    else:
        Sigma_normal, _ = ledoit_wolf_covariance(R)

    if stress_mask.sum() >= max(n + 1, 5):
        lw_stress.fit(R_vals[stress_mask.values])
        Sigma_stress = lw_stress.covariance_
    else:
        Sigma_stress = Sigma_normal * 2.0

    current_vix = float(vix_aligned.iloc[-1]) if len(vix_aligned) > 0 else 20.0
    p_stress = np.clip((current_vix - (stress_threshold - 5)) / 20.0, 0.0, 1.0)

    Sigma_blended = (1 - p_stress) * Sigma_normal + p_stress * Sigma_stress

    return Sigma_blended, p_stress, Sigma_normal, Sigma_stress


# =====================================================================
# Convenience: Full Covariance Pipeline
# =====================================================================

def estimate_covariance_full(
    daily_prices: pd.DataFrame,
    vix_series: pd.Series = None,
    use_regime: bool = True,
) -> dict:
    """
    Full covariance estimation pipeline.
    Returns dict with Sigma (monthly units), diagnostics, etc.
    """
    monthly_returns = compute_monthly_returns(daily_prices)

    Sigma_lw, shrinkage = ledoit_wolf_covariance(monthly_returns)

    pca_report = pca_validation(monthly_returns)

    result = {
        "Sigma": Sigma_lw,
        "shrinkage_coef": shrinkage,
        "pca_report": pca_report,
        "monthly_returns": monthly_returns,
        "method": "ledoit_wolf",
    }

    if use_regime and vix_series is not None and len(vix_series) > 0:
        Sigma_regime, p_stress, Sigma_n, Sigma_s = regime_conditional_covariance(
            monthly_returns, vix_series)
        result["Sigma"] = Sigma_regime
        result["p_stress"] = p_stress
        result["Sigma_normal"] = Sigma_n
        result["Sigma_stress"] = Sigma_s
        result["method"] = "ledoit_wolf_regime"

    return result
