"""
Main Portfolio Optimization Pipeline
Orchestrates all stages with rolling monthly backtest.

Architecture:
  Stage 1: Data collection (Bloomberg + yfinance)
  Stage 2: Feature engineering (momentum, risk, fundamentals, composites)
  Stage 3: Expected returns (equilibrium prior → Elastic-Net → BL posterior)
  Stage 4: Covariance estimation (Ledoit-Wolf on monthly returns + regime)
  Stage 5: Optimization (MV, CVaR, Robust CVaR, ERC) with fallback
  Stage 6: Constraints (sector, turnover, liquidity)
  Stage 7: Risk measurement (metrics, attribution, stress tests)
"""
import os
import warnings
import numpy as np
import pandas as pd

from config import (
    BLOOMBERG_CSV_PATH, START_DATE, END_DATE, N_STOCKS,
    DATA_OUTPUT_DIR, CACHE_DIR,
    TAU, RISK_AVERSION_FALLBACK, CVAR_BETA, CLIP_SIGMA,
    MAX_WEIGHT, MIN_WEIGHT, MAX_TURNOVER, MAX_SECTOR_WEIGHT,
    MIN_TRAIN_MONTHS, EMBARGO_DAYS,
    ELASTIC_NET_L1_RATIO, SAMPLE_WEIGHT_HALFLIFE_MONTHS, CV_FOLDS,
    VIX_STRESS_THRESHOLD,
)
from data_loader import (
    load_bloomberg_price_data, get_selected_tickers,
    fetch_market_cap, fetch_macro_data, run_data_quality_checks,
)
from utils_ticker_map import get_ticker_mapping
from fundamentals_loader import fetch_fundamentals, build_fundamental_features
from feature_engineering import (
    compute_returns, momentum_features, risk_features, liquidity_features,
    zscore_cross_sectional,
    build_value_composite, build_quality_composite,
    build_momentum_composite, build_growth_composite,
    build_final_alpha, build_feature_matrix_fast,
)
from covariance_estimation import (
    compute_monthly_returns, ledoit_wolf_covariance,
    pca_validation, estimate_covariance_full,
)
from expected_returns import (
    compute_risk_aversion, equilibrium_prior,
    train_elastic_net, build_training_data, scale_signal,
    black_litterman_posterior, post_process_mu,
    compute_ic,
)
from optimization_engines import (
    mean_variance_optimizer, mean_cvar_optimizer,
    robust_cvar_optimizer, erc_optimizer,
    compute_kappa, solve_with_fallback,
)
from constraints import (
    get_sector_mapping, sector_constraint_matrix,
    compute_turnover,
)
from risk_metrics import (
    compute_risk_metrics, component_cvar, stress_test,
    risk_attribution_summary, compute_max_drawdown,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_pipeline():
    """Execute full portfolio optimization pipeline with rolling backtest."""
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==================================================================
    # STAGE 1: DATA COLLECTION
    # ==================================================================
    print("=" * 70)
    print("STAGE 1: Data Collection and Preparation")
    print("=" * 70)

    adj_close_all, open_all, high_all, low_all, volume_all, _ = \
        load_bloomberg_price_data()
    print(f"  Loaded: {adj_close_all.shape[1]} stocks, "
          f"{adj_close_all.shape[0]} trading days")

    selected = get_selected_tickers()
    print(f"  Selected {len(selected)} diversified tickers via K-Medoids")

    missing_in_data = [t for t in selected if t not in adj_close_all.columns]
    if missing_in_data:
        print(f"  WARNING: {len(missing_in_data)} tickers not in price data, "
              f"removing: {missing_in_data}")
        selected = [t for t in selected if t in adj_close_all.columns]

    adj_close = adj_close_all[selected].copy()
    volume = volume_all[selected].copy()
    tickers = list(adj_close.columns)
    N = len(tickers)
    print(f"  Universe: {N} stocks")

    quality_report = run_data_quality_checks(adj_close, volume)
    print(f"  Data quality: {quality_report['status']} "
          f"({len(quality_report['return_outliers'])} outlier events)")

    print("\n  Fetching market caps from yfinance...")
    mcap_series = fetch_market_cap(tickers)
    mcap_arr = np.array([mcap_series.get(t, np.nan) for t in tickers])
    median_mcap = np.nanmedian(mcap_arr[mcap_arr > 0]) \
        if np.any(mcap_arr > 0) else 1e9
    mcap_arr = np.where(np.isnan(mcap_arr) | (mcap_arr <= 0),
                        median_mcap, mcap_arr)
    print(f"  Market caps: min={mcap_arr.min()/1e9:.1f}B, "
          f"max={mcap_arr.max()/1e9:.1f}B")

    print("\n  Fetching macro data (VIX, yields, benchmark)...")
    macro = fetch_macro_data(START_DATE, END_DATE)
    vix_series = macro.get("vix", pd.Series(dtype=float))
    spy_returns = macro.get("spy_returns", pd.Series(dtype=float))
    rf_daily = macro.get("rf_daily", pd.Series(0.0, index=adj_close.index))
    print(f"  VIX: {len(vix_series)} obs, "
          f"SPY returns: {len(spy_returns)} obs")

    print("\n  Fetching fundamental data from yfinance...")
    fundamentals_raw = fetch_fundamentals(tickers)
    print(f"  Fundamentals fetched for {len(fundamentals_raw)} tickers")

    print("\n  Fetching sector mapping...")
    sector_map = get_sector_mapping(tickers)
    sector_B, sector_names = sector_constraint_matrix(tickers, sector_map)
    print(f"  Sectors: {len(sector_names)} "
          f"({', '.join(sector_names[:5])}{'...' if len(sector_names)>5 else ''})")

    # ==================================================================
    # STAGE 2: FEATURE ENGINEERING
    # ==================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Feature Engineering and Factor Construction")
    print("=" * 70)

    rf_aligned = rf_daily.reindex(adj_close.index, method="ffill").fillna(0)
    returns_dict = compute_returns(adj_close, rf_aligned)
    r_1d = returns_dict["arithmetic_1d"]
    excess_1d = returns_dict["excess_1d"]

    benchmark_returns = spy_returns.reindex(
        adj_close.index, method="ffill").fillna(0)

    monthly_returns = compute_monthly_returns(adj_close)
    monthly_excess_returns = monthly_returns.sub(
        rf_aligned.resample("ME").mean(), axis=0).fillna(monthly_returns)
    rebalance_dates = list(monthly_returns.index)
    print(f"  Monthly returns: {len(rebalance_dates)} months")

    rebalance_daily_dates = []
    for rdate in rebalance_dates:
        mask = adj_close.index <= rdate
        if mask.any():
            rebalance_daily_dates.append(adj_close.index[mask][-1])
        else:
            rebalance_daily_dates.append(rdate)

    fund_features = build_fundamental_features(
        fundamentals_raw, tickers,
        pd.DatetimeIndex(rebalance_daily_dates))
    print(f"  Fundamental features: {len(fund_features)} variables")

    print("  Building feature matrices for all rebalance dates...")
    X_dict, feature_names = build_feature_matrix_fast(
        adj_close, r_1d, volume,
        market_returns=benchmark_returns,
        fundamental_features=fund_features,
        dates_subset=pd.DatetimeIndex(rebalance_daily_dates),
    )
    print(f"  Features: {len(feature_names)} per stock "
          f"({', '.join(feature_names[:5])}...)")

    # ==================================================================
    # STAGES 3-4: EXPECTED RETURNS & COVARIANCE (computed per period)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STAGE 3-4: Expected Returns & Covariance (rolling)")
    print("=" * 70)

    benchmark_mean_annual = float(benchmark_returns.mean() * 252)
    benchmark_var_annual = float(benchmark_returns.var() * 252)
    lambda_risk = compute_risk_aversion(benchmark_mean_annual,
                                        benchmark_var_annual)
    print(f"  Risk aversion lambda: {lambda_risk:.2f}")

    kappa_base = compute_kappa(N, max(MIN_TRAIN_MONTHS, 24))
    print(f"  Robust CVaR kappa (base): {kappa_base:.3f}")

    # ==================================================================
    # STAGE 5-7: ROLLING MONTHLY BACKTEST
    # ==================================================================
    print("\n" + "=" * 70)
    print("STAGE 5-7: Rolling Monthly Backtest")
    print("=" * 70)
    print(f"  Train period: first {MIN_TRAIN_MONTHS} months")
    print(f"  Out-of-sample: months {MIN_TRAIN_MONTHS+1} to "
          f"{len(rebalance_dates)}")

    results = {
        "date": [],
        "w_mv": [], "w_cvar": [], "w_robust": [], "w_erc": [],
        "method_mv": [], "method_cvar": [],
        "method_robust": [], "method_erc": [],
        "mu_bl": [],
        "Sigma": [],
        "ic": [],
        "turnover_mv": [], "turnover_cvar": [],
        "turnover_robust": [], "turnover_erc": [],
        "metrics_mv": [], "metrics_cvar": [],
        "metrics_robust": [], "metrics_erc": [],
    }

    w_prev = {
        "mv": None, "cvar": None, "robust": None, "erc": None
    }

    oos_returns = {
        "mv": [], "cvar": [], "robust": [], "erc": [],
        "equal": [], "market_cap": [],
    }
    oos_dates = []

    for i in range(MIN_TRAIN_MONTHS, len(rebalance_dates)):
        rdate = rebalance_dates[i]
        rdate_daily = rebalance_daily_dates[i]
        print(f"\n  --- Rebalance {i+1}/{len(rebalance_dates)}: "
              f"{rdate.strftime('%Y-%m')} ---")

        # ---- Covariance (expanding window of monthly returns) ----
        train_monthly = monthly_returns.iloc[:i]
        if len(train_monthly) < 5:
            print("    Skipping: insufficient monthly data")
            continue

        vix_for_cov = vix_series
        cov_result = estimate_covariance_full(
            adj_close.loc[:rdate],
            vix_series=vix_for_cov,
            use_regime=len(vix_series) > 0,
        )
        Sigma = cov_result["Sigma"]
        print(f"    Covariance: {cov_result['method']}, "
              f"shrinkage={cov_result.get('shrinkage_coef', 'N/A')}")

        # ---- Equilibrium Prior ----
        Pi = equilibrium_prior(mcap_arr, Sigma, lambda_risk)

        # ---- ML Alpha (Elastic-Net) ----
        X_train, y_train, sample_weights, X_current = build_training_data(
            X_dict, monthly_excess_returns, i,
            feature_dates=rebalance_daily_dates,
            return_dates=rebalance_dates,
            tickers=tickers,
            halflife_months=SAMPLE_WEIGHT_HALFLIFE_MONTHS,
        )

        if X_train is not None and X_current is not None and len(X_train) > 20:
            y_hat, best_alpha, cv_rmse = train_elastic_net(
                X_train, y_train, X_current,
                sample_weights=sample_weights,
                l1_ratio=ELASTIC_NET_L1_RATIO,
                n_cv_folds=CV_FOLDS,
            )
            q = scale_signal(y_hat, target_cross_sectional_std=0.005)
            print(f"    Elastic-Net: alpha={best_alpha:.4f}, "
                  f"RMSE={cv_rmse:.5f}")
        else:
            q = np.zeros(N)
            print("    Elastic-Net: skipped (insufficient training data)")

        # ---- Black-Litterman Posterior ----
        mu_BL, Sigma_BL = black_litterman_posterior(
            Pi, q, Sigma, tau=TAU)
        mu_BL = post_process_mu(mu_BL, clip_sigma=CLIP_SIGMA)

        # ---- IC Check ----
        ic_val = np.nan
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
            if next_date in monthly_excess_returns.index:
                realized = monthly_excess_returns.loc[next_date, tickers].values
                ic_val = compute_ic(mu_BL, realized)

        # ---- Scenario Matrix (monthly returns for CVaR) ----
        R_scenarios = train_monthly.values
        T_scenarios = R_scenarios.shape[0]

        r_target = float(np.median(mu_BL))

        # ---- Optimization ----
        kappa = compute_kappa(N, T_scenarios)

        common_kwargs = dict(
            gamma_turnover=0.005,
            max_weight=MAX_WEIGHT,
            min_weight=MIN_WEIGHT,
            max_turnover=MAX_TURNOVER,
            sector_B=sector_B,
            max_sector_weight=MAX_SECTOR_WEIGHT,
        )

        # MV
        w_mv, method_mv = solve_with_fallback(
            mu_BL, Sigma, R_scenarios, "mean_variance",
            w_prev=w_prev["mv"],
            lambda_risk=lambda_risk,
            **common_kwargs,
        )

        # CVaR
        w_cvar, method_cvar = solve_with_fallback(
            mu_BL, Sigma, R_scenarios, "mean_cvar",
            r_target=r_target,
            w_prev=w_prev["cvar"],
            **common_kwargs,
        )

        # Robust CVaR
        w_robust, method_robust = solve_with_fallback(
            mu_BL, Sigma, R_scenarios, "robust_cvar",
            r_target=r_target,
            kappa=kappa,
            w_prev=w_prev["robust"],
            **common_kwargs,
        )

        # ERC
        w_erc, method_erc = solve_with_fallback(
            mu_BL, Sigma, R_scenarios, "erc",
            max_weight=MAX_WEIGHT,
            min_weight=MIN_WEIGHT,
        )

        print(f"    Optimizers: MV={method_mv}, CVaR={method_cvar}, "
              f"Robust={method_robust}, ERC={method_erc}")

        # ---- Turnover ----
        to_mv = compute_turnover(w_mv, w_prev["mv"]) \
            if w_prev["mv"] is not None else 0.0
        to_cvar = compute_turnover(w_cvar, w_prev["cvar"]) \
            if w_prev["cvar"] is not None else 0.0
        to_robust = compute_turnover(w_robust, w_prev["robust"]) \
            if w_prev["robust"] is not None else 0.0
        to_erc = compute_turnover(w_erc, w_prev["erc"]) \
            if w_prev["erc"] is not None else 0.0
        print(f"    Turnover: MV={to_mv:.3f}, CVaR={to_cvar:.3f}, "
              f"Robust={to_robust:.3f}, ERC={to_erc:.3f}")

        # ---- Risk Metrics ----
        metrics_mv = compute_risk_metrics(w_mv, mu_BL, Sigma, R_scenarios)
        metrics_cvar = compute_risk_metrics(w_cvar, mu_BL, Sigma, R_scenarios)
        metrics_robust = compute_risk_metrics(w_robust, mu_BL, Sigma,
                                              R_scenarios)
        metrics_erc = compute_risk_metrics(w_erc, mu_BL, Sigma, R_scenarios)

        print(f"    Ex-ante Sharpe: MV={metrics_mv['sharpe_annual']:.2f}, "
              f"CVaR={metrics_cvar['sharpe_annual']:.2f}, "
              f"Robust={metrics_robust['sharpe_annual']:.2f}, "
              f"ERC={metrics_erc['sharpe_annual']:.2f}")

        # ---- Store Results ----
        results["date"].append(rdate)
        results["w_mv"].append(w_mv.copy())
        results["w_cvar"].append(w_cvar.copy())
        results["w_robust"].append(w_robust.copy())
        results["w_erc"].append(w_erc.copy())
        results["method_mv"].append(method_mv)
        results["method_cvar"].append(method_cvar)
        results["method_robust"].append(method_robust)
        results["method_erc"].append(method_erc)
        results["mu_bl"].append(mu_BL.copy())
        results["Sigma"].append(Sigma.copy())
        results["ic"].append(ic_val)
        results["turnover_mv"].append(to_mv)
        results["turnover_cvar"].append(to_cvar)
        results["turnover_robust"].append(to_robust)
        results["turnover_erc"].append(to_erc)
        results["metrics_mv"].append(metrics_mv)
        results["metrics_cvar"].append(metrics_cvar)
        results["metrics_robust"].append(metrics_robust)
        results["metrics_erc"].append(metrics_erc)

        # ---- Out-of-Sample Returns ----
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
            if next_date in monthly_returns.index:
                realized_monthly = monthly_returns.loc[next_date, tickers].values
                oos_dates.append(next_date)
                oos_returns["mv"].append(float(w_mv @ realized_monthly))
                oos_returns["cvar"].append(float(w_cvar @ realized_monthly))
                oos_returns["robust"].append(float(w_robust @ realized_monthly))
                oos_returns["erc"].append(float(w_erc @ realized_monthly))
                oos_returns["equal"].append(float(np.mean(realized_monthly)))
                w_mcap = mcap_arr / mcap_arr.sum()
                oos_returns["market_cap"].append(
                    float(w_mcap @ realized_monthly))

        # ---- Update Previous Weights ----
        w_prev["mv"] = w_mv.copy()
        w_prev["cvar"] = w_cvar.copy()
        w_prev["robust"] = w_robust.copy()
        w_prev["erc"] = w_erc.copy()

    # ==================================================================
    # POST-BACKTEST ANALYSIS
    # ==================================================================
    print("\n" + "=" * 70)
    print("POST-BACKTEST PERFORMANCE ANALYSIS")
    print("=" * 70)

    if not oos_dates:
        print("  No out-of-sample periods available.")
        return results

    oos_df = pd.DataFrame({
        "date": oos_dates,
        "MV": oos_returns["mv"],
        "CVaR": oos_returns["cvar"],
        "Robust_CVaR": oos_returns["robust"],
        "ERC": oos_returns["erc"],
        "Equal_Weight": oos_returns["equal"],
        "Market_Cap": oos_returns["market_cap"],
    }).set_index("date")

    oos_df.to_csv(os.path.join(DATA_OUTPUT_DIR, "oos_returns.csv"))

    print("\n  Out-of-Sample Monthly Returns Summary:")
    print("  " + "-" * 65)
    summary_rows = []
    for strategy in ["MV", "CVaR", "Robust_CVaR", "ERC",
                     "Equal_Weight", "Market_Cap"]:
        rets = oos_df[strategy].values
        ann_ret = float(np.mean(rets)) * 12
        ann_vol = float(np.std(rets)) * np.sqrt(12)
        sharpe = ann_ret / (ann_vol + 1e-10)
        cum_ret = float(np.prod(1 + rets) - 1)
        max_dd = compute_max_drawdown(rets)
        skew = float(pd.Series(rets).skew())

        summary_rows.append({
            "Strategy": strategy,
            "Ann_Return": ann_ret,
            "Ann_Vol": ann_vol,
            "Sharpe": sharpe,
            "Cum_Return": cum_ret,
            "Max_DD": max_dd,
            "Skewness": skew,
            "N_Months": len(rets),
        })

        print(f"  {strategy:15s}: Return={ann_ret:+.2%}, "
              f"Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}, "
              f"CumRet={cum_ret:+.2%}, MaxDD={max_dd:.2%}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(DATA_OUTPUT_DIR, "performance_summary.csv"),
                      index=False)

    # ---- Turnover Summary ----
    print("\n  Average Turnover:")
    for opt_name, key in [("MV", "turnover_mv"), ("CVaR", "turnover_cvar"),
                          ("Robust_CVaR", "turnover_robust"),
                          ("ERC", "turnover_erc")]:
        avg_to = np.mean(results[key])
        print(f"    {opt_name:15s}: {avg_to:.3f}")

    # ---- IC Summary ----
    ic_values = [v for v in results["ic"] if not np.isnan(v)]
    if ic_values:
        avg_ic = np.mean(ic_values)
        print(f"\n  Average Information Coefficient: {avg_ic:.4f}")
        if avg_ic < 0.03:
            print("  WARNING: IC < 0.03 — signal has weak predictive power")

    # ---- Stress Testing ----
    print("\n  Stress Test (last portfolio weights):")
    stress_periods = {
        "COVID_Crash": ("2020-02-20", "2020-03-23"),
        "Rate_Hike_2022": ("2022-01-01", "2022-12-31"),
        "Q4_2018_Selloff": ("2018-10-01", "2018-12-31"),
    }

    daily_returns_full = adj_close.pct_change().dropna()
    for opt_name, w_final in [("MV", w_prev["mv"]),
                               ("CVaR", w_prev["cvar"]),
                               ("Robust_CVaR", w_prev["robust"]),
                               ("ERC", w_prev["erc"])]:
        if w_final is None:
            continue
        stress_results = stress_test(w_final, daily_returns_full,
                                     stress_periods)
        print(f"\n    {opt_name}:")
        for scenario, metrics in stress_results.items():
            if metrics["n_days"] > 0:
                print(f"      {scenario}: Return={metrics['total_return']:+.2%}, "
                      f"MaxDD={metrics['max_drawdown']:.2%}, "
                      f"CVaR95={metrics['cvar_95']:.4f}")

    # ---- Save Final Weights ----
    if results["date"]:
        last_idx = -1
        weights_df = pd.DataFrame({
            "ticker": tickers,
            "sector": [sector_map.get(t, "Unknown") for t in tickers],
            "market_cap": mcap_arr,
            "w_mv": results["w_mv"][last_idx],
            "w_cvar": results["w_cvar"][last_idx],
            "w_robust": results["w_robust"][last_idx],
            "w_erc": results["w_erc"][last_idx],
            "mu_bl": results["mu_bl"][last_idx],
        })
        weights_df.to_csv(
            os.path.join(DATA_OUTPUT_DIR, "final_weights.csv"), index=False)

        # Risk Attribution for Robust CVaR (main contribution)
        last_Sigma = results["Sigma"][last_idx]
        last_mu = results["mu_bl"][last_idx]
        last_R = monthly_returns.values
        w_robust_final = results["w_robust"][last_idx]

        attrib = risk_attribution_summary(
            w_robust_final, last_mu, last_Sigma, last_R, tickers)
        attrib.to_csv(
            os.path.join(DATA_OUTPUT_DIR, "risk_attribution_robust.csv"),
            index=False)

        print(f"\n  Results saved to {DATA_OUTPUT_DIR}/")
        print(f"    - final_weights.csv")
        print(f"    - performance_summary.csv")
        print(f"    - oos_returns.csv")
        print(f"    - risk_attribution_robust.csv")

    # ---- PCA Validation ----
    pca_report = pca_validation(monthly_returns)
    print(f"\n  PCA Validation:")
    print(f"    First 5 components explain: "
          f"{pca_report['cumulative_variance'][:5]}")
    print(f"    Has factor structure: {pca_report['has_factor_structure']}")
    print(f"    Recommendation: {pca_report['recommendation']}")

    # ---- Sensitivity Analysis Notes ----
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS (recommended in research paper)")
    print("=" * 70)
    print("  1. tau in {0.01, 0.025, 0.05, 0.1} — show weights stability")
    print("  2. kappa in {0.1, 0.5, 1.0, 2.0, 5.0} — robustness-performance curve")
    print("  3. IC-weighted vs equal-weighted composites — robustness check")
    print("  4. Sample covariance vs Ledoit-Wolf — out-of-sample variance")
    print("  5. With/without regime adjustment — stress period performance")

    return results


# =====================================================================
# Sensitivity Analysis Runner
# =====================================================================

def run_tau_sensitivity(
    Pi: np.ndarray,
    q: np.ndarray,
    Sigma: np.ndarray,
    taus: list = None,
) -> pd.DataFrame:
    """Test BL posterior sensitivity to tau."""
    if taus is None:
        taus = [0.01, 0.025, 0.05, 0.1]

    rows = []
    for tau in taus:
        mu_bl, _ = black_litterman_posterior(Pi, q, Sigma, tau=tau)
        rows.append({
            "tau": tau,
            "mu_mean": np.mean(mu_bl),
            "mu_std": np.std(mu_bl),
            "mu_range": np.max(mu_bl) - np.min(mu_bl),
        })
    return pd.DataFrame(rows)


def run_kappa_sensitivity(
    mu: np.ndarray,
    Sigma: np.ndarray,
    R_scenarios: np.ndarray,
    kappas: list = None,
) -> pd.DataFrame:
    """Test Robust CVaR sensitivity to kappa."""
    if kappas is None:
        kappas = [0.1, 0.5, 1.0, 2.0, 5.0]

    r_target = float(np.median(mu))
    rows = []
    for kappa in kappas:
        w = robust_cvar_optimizer(
            mu, Sigma, R_scenarios, r_target, kappa=kappa)
        if w is not None:
            metrics = compute_risk_metrics(w, mu, Sigma, R_scenarios)
            rows.append({
                "kappa": kappa,
                "sharpe": metrics["sharpe_annual"],
                "cvar": metrics["CVaR_recommended"],
                "volatility": metrics["volatility_annual"],
                "max_weight": np.max(w),
                "n_active": np.sum(w > 0.015),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_pipeline()
