"""
Stage 7: Risk Measurement and Attribution

Ex-ante risk metrics:
- Portfolio volatility (annualized)
- Sharpe ratio
- VaR (parametric Gaussian, historical, t-distribution)
- CVaR (parametric, historical) with normality test for selection
- Component CVaR (Euler decomposition)
- Stress testing on historical episodes
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t, jarque_bera
from config import CVAR_BETA


# =====================================================================
# Ex-Ante Risk Metrics
# =====================================================================

def compute_risk_metrics(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    R_scenarios: np.ndarray,
    beta: float = CVAR_BETA,
) -> dict:
    """
    Comprehensive ex-ante risk metrics (monthly units, annualized for reporting).

    w: (N,) portfolio weights.
    mu: (N,) expected monthly returns.
    Sigma: (N,N) monthly covariance.
    R_scenarios: (T,N) monthly return scenarios.
    """
    portfolio_return_monthly = float(mu @ w)
    portfolio_var_monthly = float(w @ Sigma @ w)
    portfolio_vol_monthly = np.sqrt(portfolio_var_monthly)
    portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)

    sharpe_annual = (portfolio_return_monthly * 12) / (
        portfolio_vol_annual + 1e-10)

    z_alpha = norm.ppf(beta)
    VaR_parametric = -(portfolio_return_monthly
                       - z_alpha * portfolio_vol_monthly)
    CVaR_parametric = -(portfolio_return_monthly
                        - portfolio_vol_monthly * norm.pdf(z_alpha)
                        / (1 - beta))

    scenario_returns = R_scenarios @ w
    scenario_losses = -scenario_returns

    VaR_historical = float(np.quantile(scenario_losses, beta))
    tail_mask = scenario_losses >= VaR_historical
    CVaR_historical = float(scenario_losses[tail_mask].mean()) \
        if tail_mask.any() else VaR_historical

    try:
        nu, loc, scale = t.fit(scenario_returns)
        VaR_t = -float(t.ppf(1 - beta, df=nu, loc=loc, scale=scale))
    except Exception:
        VaR_t = VaR_parametric

    jb_stat, jb_pval = jarque_bera(scenario_returns)
    normality_rejected = jb_pval < 0.05

    CVaR_recommended = CVaR_historical if normality_rejected else CVaR_parametric

    return {
        "expected_return_monthly": portfolio_return_monthly,
        "expected_return_annual": portfolio_return_monthly * 12,
        "volatility_monthly": portfolio_vol_monthly,
        "volatility_annual": portfolio_vol_annual,
        "sharpe_annual": sharpe_annual,
        "VaR_parametric": VaR_parametric,
        "CVaR_parametric": CVaR_parametric,
        "VaR_historical": VaR_historical,
        "CVaR_historical": CVaR_historical,
        "VaR_t_distribution": VaR_t,
        "jb_stat": jb_stat,
        "jb_pval": jb_pval,
        "normality_rejected": normality_rejected,
        "CVaR_recommended": CVaR_recommended,
    }


# =====================================================================
# Component CVaR (Euler Decomposition)
# =====================================================================

def component_cvar(
    w: np.ndarray,
    R_scenarios: np.ndarray,
    beta: float = CVAR_BETA,
) -> tuple:
    """
    Euler decomposition of CVaR into per-stock contributions.

    Returns (component_cvar, marginal_cvar, total_cvar).
    component_cvar[i] = w_i * MCVaR_i
    Sum of component_cvar equals total CVaR (Euler theorem).
    """
    T, N = R_scenarios.shape
    portfolio_returns = R_scenarios @ w
    portfolio_losses = -portfolio_returns

    var_threshold = np.quantile(portfolio_losses, beta)
    tail_mask = portfolio_losses >= var_threshold

    if not tail_mask.any():
        return np.zeros(N), np.zeros(N), 0.0

    marginal_cvar = -R_scenarios[tail_mask].mean(axis=0)

    comp_cvar = w * marginal_cvar

    total_cvar = float(portfolio_losses[tail_mask].mean())

    decomp_error = abs(comp_cvar.sum() - total_cvar)
    if decomp_error > 0.01 * abs(total_cvar + 1e-10):
        comp_cvar = comp_cvar * (total_cvar / (comp_cvar.sum() + 1e-10))

    return comp_cvar, marginal_cvar, total_cvar


# =====================================================================
# Maximum Drawdown
# =====================================================================

def compute_max_drawdown(returns_series: np.ndarray) -> float:
    """Compute maximum drawdown from a return series."""
    cumulative = np.cumprod(1 + returns_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    return float(np.min(drawdown))


# =====================================================================
# Stress Testing
# =====================================================================

def stress_test(
    w: np.ndarray,
    daily_returns: pd.DataFrame,
    stress_periods: dict,
) -> dict:
    """
    Apply historical stress scenarios to current portfolio weights.

    daily_returns: DataFrame(dates x tickers) of daily returns.
    stress_periods: {name: (start_date, end_date)} for each scenario.

    Returns dict[scenario_name] -> {total_return, max_drawdown, worst_day, CVaR_95}.
    """
    results = {}

    for name, (start, end) in stress_periods.items():
        mask = (daily_returns.index >= start) & (daily_returns.index <= end)
        period_returns = daily_returns.loc[mask]

        if period_returns.empty:
            results[name] = {
                "total_return": np.nan,
                "max_drawdown": np.nan,
                "worst_day": np.nan,
                "cvar_95": np.nan,
                "n_days": 0,
            }
            continue

        w_aligned = np.array([
            w[i] if i < len(w) else 0
            for i in range(period_returns.shape[1])
        ])[:period_returns.shape[1]]

        port_returns = period_returns.values @ w_aligned

        total_ret = float(np.prod(1 + port_returns) - 1)
        max_dd = compute_max_drawdown(port_returns)
        worst_day = float(np.min(port_returns))

        losses = -port_returns
        var_95 = np.quantile(losses, 0.95) if len(losses) > 1 else 0
        cvar_95 = float(losses[losses >= var_95].mean()) \
            if (losses >= var_95).any() else var_95

        results[name] = {
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "worst_day": worst_day,
            "cvar_95": cvar_95,
            "n_days": len(port_returns),
        }

    return results


# =====================================================================
# Risk Attribution Summary
# =====================================================================

def risk_attribution_summary(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    R_scenarios: np.ndarray,
    tickers: list,
    beta: float = CVAR_BETA,
) -> pd.DataFrame:
    """
    Combine risk metrics and component CVaR into a summary table.
    One row per stock.
    """
    comp_cvar, marg_cvar, total_cvar = component_cvar(w, R_scenarios, beta)

    port_vol = np.sqrt(w @ Sigma @ w)
    marginal_vol = Sigma @ w / (port_vol + 1e-10)
    component_vol = w * marginal_vol
    pct_risk_vol = component_vol / (port_vol + 1e-10) * 100

    pct_risk_cvar = comp_cvar / (total_cvar + 1e-10) * 100

    df = pd.DataFrame({
        "ticker": tickers,
        "weight": w,
        "expected_return": mu,
        "marginal_vol": marginal_vol,
        "component_vol": component_vol,
        "pct_risk_vol": pct_risk_vol,
        "marginal_cvar": marg_cvar,
        "component_cvar": comp_cvar,
        "pct_risk_cvar": pct_risk_cvar,
    })

    return df
