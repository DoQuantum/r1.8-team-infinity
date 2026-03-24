"""
Stage 2: Feature Engineering and Factor Construction

Revised methodology — streamlined for 30-stock universe.

Returns: 1d, 21d (monthly), 63d (quarterly), 252d (annual). Dropped 5d.
Momentum: 12-2 month, 1-month, 52-week high ratio. Dropped 1d reversal.
Risk: Realized vol 21d, Beta 60d, Idiosyncratic vol 60d, Skewness 120d.
      Dropped kurtosis (too noisy) and max drawdown (redundant with vol).
Liquidity: Amihud 21d, ADV 21d. Dropped Corwin-Schultz (unnecessary for large-cap).
Composites: Value, Quality, Momentum, Growth — equal-weighted.
"""
import numpy as np
import pandas as pd
from typing import Optional


# =====================================================================
# Return Computation
# =====================================================================

def compute_returns(
    prices: pd.DataFrame,
    rf_series: Optional[pd.Series] = None,
) -> dict:
    """
    Compute arithmetic, excess, and log returns at multiple horizons.
    All backward-looking (what has happened up to date t).
    """
    r_1d = prices.pct_change()
    log_r_1d = np.log(prices / prices.shift(1))

    rf = rf_series if rf_series is not None else pd.Series(
        0.0, index=prices.index)
    rf = rf.reindex(prices.index, method="ffill").fillna(0)
    excess_1d = r_1d.sub(rf, axis=0)

    return {
        "arithmetic_1d": r_1d,
        "excess_1d": excess_1d,
        "log_1d": log_r_1d,
        "arithmetic_21d": prices / prices.shift(21) - 1,
        "arithmetic_63d": prices / prices.shift(63) - 1,
        "arithmetic_252d": prices / prices.shift(252) - 1,
    }


# =====================================================================
# Momentum Features
# =====================================================================

def momentum_features(prices: pd.DataFrame) -> dict:
    """
    Revised momentum features:
    - 12-2 month cumulative return (skip last month)
    - 1-month (21-day) recent momentum
    - 52-week high ratio: P_t / max(P_{t-252:t})
    """
    features = {}

    ret_252 = prices / prices.shift(252) - 1
    ret_21 = prices / prices.shift(21) - 1
    features["momentum_12_2"] = ret_252 - ret_21

    features["momentum_1m"] = ret_21

    high_52w = prices.rolling(252, min_periods=21).max()
    features["high_52w_ratio"] = prices / high_52w

    return features


# =====================================================================
# Risk Features
# =====================================================================

def risk_features(
    returns_1d: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
) -> dict:
    """
    Revised risk features:
    - Realized volatility (21d, annualized)
    - Beta (60d OLS to benchmark)
    - Idiosyncratic volatility (60d CAPM residuals, annualized)
    - Realized skewness (120d window — more stable than 60d for 3rd moment)
    """
    r = returns_1d
    features = {}

    features["volatility_21d"] = r.rolling(
        21, min_periods=5).std() * np.sqrt(252)

    features["skewness_120d"] = r.rolling(120, min_periods=30).apply(
        lambda x: pd.Series(x).skew(), raw=True)

    if market_returns is not None:
        mkt = market_returns.reindex(r.index, method="ffill").fillna(0)
        mkt_var = mkt.rolling(60, min_periods=21).var()
        r_mean = r.rolling(60, min_periods=21).mean()
        mkt_mean = mkt.rolling(60, min_periods=21).mean()
        cov_im = (r.mul(mkt, axis=0)).rolling(
            60, min_periods=21).mean() - r_mean.mul(mkt_mean, axis=0)
        beta_df = cov_im.div(mkt_var + 1e-10, axis=0)
        features["beta_60d"] = beta_df

        resid = r.sub(beta_df.mul(mkt, axis=0), axis=0)
        features["idiosyncratic_vol"] = resid.rolling(
            60, min_periods=21).std() * np.sqrt(252)
    else:
        features["beta_60d"] = pd.DataFrame(
            np.nan, index=r.index, columns=r.columns)
        features["idiosyncratic_vol"] = r.rolling(
            60, min_periods=21).std() * np.sqrt(252)

    return features


# =====================================================================
# Liquidity Features
# =====================================================================

def liquidity_features(
    returns_1d: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict:
    """
    Revised liquidity features (Amihud + ADV only):
    - Amihud illiquidity: |r| / Volume, 21d average
    - ADV: 21-day average daily volume
    """
    features = {}

    amihud = returns_1d.abs().div(volume.replace(0, np.nan))
    features["amihud_21d"] = amihud.rolling(21, min_periods=5).mean()
    features["adv_21d"] = volume.rolling(21, min_periods=5).mean()

    return features


# =====================================================================
# Cross-Sectional Z-Scoring
# =====================================================================

def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score cross-sectionally at each date (across stocks, not time)."""
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1) + 1e-10
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


# =====================================================================
# Composite Signal Construction
# =====================================================================

def build_value_composite(
    ebit_ev_z: pd.DataFrame,
    inv_pb_z: pd.DataFrame,
    inv_pe_z: pd.DataFrame,
) -> pd.DataFrame:
    """
    Value = avg(Z(EBIT/EV), Z(1/P/B), Z(1/trailing P/E))
    Higher = cheaper = more attractive.
    """
    return (ebit_ev_z.fillna(0) + inv_pb_z.fillna(0) + inv_pe_z.fillna(0)) / 3


def build_quality_composite(
    roe_z: pd.DataFrame,
    gross_margin_z: pd.DataFrame,
    neg_de_z: pd.DataFrame,
    roa_z: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quality = avg(Z(ROE), Z(Gross Margin), Z(-D/E), Z(ROA))
    Negative D/E so lower leverage = higher quality.
    """
    return (roe_z.fillna(0) + gross_margin_z.fillna(0)
            + neg_de_z.fillna(0) + roa_z.fillna(0)) / 4


def build_momentum_composite(
    mom_12_2_z: pd.DataFrame,
    mom_1m_z: pd.DataFrame,
    high_52w_z: pd.DataFrame,
) -> pd.DataFrame:
    """
    Momentum = avg(Z(12-2 month return), Z(1-month return), Z(52-week high ratio))
    """
    return (mom_12_2_z.fillna(0) + mom_1m_z.fillna(0)
            + high_52w_z.fillna(0)) / 3


def build_growth_composite(
    revenue_growth_z: pd.DataFrame,
    delta_roe_z: pd.DataFrame,
    eps_surprise_z: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Growth = avg(Z(Revenue Growth YoY), Z(ΔROE), Z(EPS Surprise %))
    If EPS Surprise unavailable, use 2-component average.
    """
    total = revenue_growth_z.fillna(0) + delta_roe_z.fillna(0)
    n_components = 2
    if eps_surprise_z is not None:
        total = total + eps_surprise_z.fillna(0)
        n_components = 3
    return total / n_components


def build_final_alpha(
    value_comp: pd.DataFrame,
    quality_comp: pd.DataFrame,
    momentum_comp: pd.DataFrame,
    growth_comp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final alpha = equal-weighted average of 4 composites.
    Equal weights are defensible for N=30 where IC estimates are noisy.
    """
    return (value_comp + quality_comp + momentum_comp + growth_comp) / 4


# =====================================================================
# Feature Matrix Builder (for ML)
# =====================================================================

def build_feature_matrix(
    prices: pd.DataFrame,
    returns_1d: pd.DataFrame,
    volume: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    fundamental_features: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a panel DataFrame: MultiIndex (date, ticker) with all features as columns.
    Used as input for the Elastic-Net cross-sectional regression.
    """
    mom = momentum_features(prices)
    risk = risk_features(returns_1d, market_returns)
    liq = liquidity_features(returns_1d, volume)

    all_features = {}
    for name, df in mom.items():
        all_features[name] = zscore_cross_sectional(df)
    for name, df in risk.items():
        all_features[name] = zscore_cross_sectional(df)
    all_features["amihud_21d"] = zscore_cross_sectional(liq["amihud_21d"])

    if fundamental_features is not None:
        for name, df in fundamental_features.items():
            df_aligned = df.reindex(index=prices.index, columns=prices.columns)
            df_aligned = df_aligned.ffill()
            all_features[f"fund_{name}"] = zscore_cross_sectional(df_aligned)

    panels = []
    feature_names = sorted(all_features.keys())
    for date in prices.index:
        for ticker in prices.columns:
            row = {"date": date, "ticker": ticker}
            for feat_name in feature_names:
                feat_df = all_features[feat_name]
                if date in feat_df.index and ticker in feat_df.columns:
                    row[feat_name] = feat_df.loc[date, ticker]
                else:
                    row[feat_name] = np.nan
            panels.append(row)

    panel_df = pd.DataFrame(panels)
    panel_df = panel_df.set_index(["date", "ticker"])
    return panel_df


def build_feature_matrix_fast(
    prices: pd.DataFrame,
    returns_1d: pd.DataFrame,
    volume: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    fundamental_features: Optional[dict] = None,
    dates_subset: Optional[pd.DatetimeIndex] = None,
) -> tuple:
    """
    Faster feature matrix builder that returns (X_dict, feature_names).
    X_dict[date] = DataFrame(index=tickers, columns=features) for each date.
    Only computes for dates in dates_subset if provided.
    """
    mom = momentum_features(prices)
    risk = risk_features(returns_1d, market_returns)
    liq = liquidity_features(returns_1d, volume)

    all_features = {}
    for name, df in mom.items():
        all_features[name] = zscore_cross_sectional(df)
    for name, df in risk.items():
        all_features[name] = zscore_cross_sectional(df)
    all_features["amihud_21d"] = zscore_cross_sectional(liq["amihud_21d"])

    if fundamental_features is not None:
        for name, df in fundamental_features.items():
            df_aligned = df.reindex(index=prices.index, columns=prices.columns)
            df_aligned = df_aligned.ffill()
            all_features[f"fund_{name}"] = zscore_cross_sectional(df_aligned)

    feature_names = sorted(all_features.keys())
    dates = dates_subset if dates_subset is not None else prices.index
    tickers = prices.columns.tolist()

    X_dict = {}
    for date in dates:
        rows = {}
        for feat_name in feature_names:
            feat_df = all_features[feat_name]
            if date in feat_df.index:
                rows[feat_name] = feat_df.loc[date]
            else:
                rows[feat_name] = pd.Series(np.nan, index=tickers)
        X_dict[date] = pd.DataFrame(rows, index=tickers)

    return X_dict, feature_names
