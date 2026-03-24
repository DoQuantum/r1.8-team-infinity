"""
Stage 5: Portfolio Optimization Engines

Optimizer 1: Mean-Variance (benchmark, lambda sweep)
Optimizer 2: Mean-CVaR (primary, beta=0.95, Rockafellar-Uryasev LP)
Optimizer 3: Robust Mean-CVaR (main contribution, SOCP with kappa)
Optimizer 4: Equal Risk Contribution (model-free benchmark, scipy SLSQP)

All use consistent constraint interface.
DRO Wasserstein removed per revised methodology (complexity vs. gain).
"""
import numpy as np
from typing import Optional
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import sqrtm

from config import MAX_WEIGHT, MIN_WEIGHT, CVAR_BETA, MAX_TURNOVER, MAX_SECTOR_WEIGHT


# =====================================================================
# Optimizer 1: Mean-Variance (Benchmark)
# =====================================================================

def mean_variance_optimizer(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lambda_risk: float = 2.5,
    w_prev: Optional[np.ndarray] = None,
    gamma_turnover: float = 0.005,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    max_turnover: float = MAX_TURNOVER,
    sector_B: Optional[np.ndarray] = None,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> Optional[np.ndarray]:
    """
    Mean-Variance: max mu'w - lambda * w'Sigma*w - gamma * ||w - w_prev||_1
    s.t. sum(w) = 1, min_weight <= w <= max_weight, turnover <= max_turnover
    """
    import cvxpy as cp

    N = len(mu)
    w = cp.Variable(N)

    portfolio_return = mu @ w
    portfolio_variance = cp.quad_form(w, Sigma)

    obj_expr = portfolio_return - lambda_risk * portfolio_variance
    if w_prev is not None:
        obj_expr -= gamma_turnover * cp.norm1(w - w_prev)

    objective = cp.Maximize(obj_expr)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    if w_prev is not None and max_turnover < 2.0:
        constraints.append(cp.norm1(w - w_prev) <= max_turnover)

    if sector_B is not None:
        for j in range(sector_B.shape[1]):
            constraints.append(
                cp.sum(cp.multiply(sector_B[:, j], w)) <= max_sector_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None

    return np.array(w.value).flatten()


# =====================================================================
# Optimizer 2: Mean-CVaR (Primary)
# =====================================================================

def mean_cvar_optimizer(
    mu: np.ndarray,
    R_scenarios: np.ndarray,
    r_target: float,
    beta: float = CVAR_BETA,
    w_prev: Optional[np.ndarray] = None,
    gamma_turnover: float = 0.005,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    max_turnover: float = MAX_TURNOVER,
    sector_B: Optional[np.ndarray] = None,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> Optional[np.ndarray]:
    """
    Mean-CVaR (Rockafellar-Uryasev formulation):
    min alpha + 1/((1-beta)*T) * sum(z) + gamma * ||w - w_prev||_1
    s.t. z >= 0, z >= -R@w - alpha, mu'w >= r_target,
         sum(w) = 1, min_weight <= w <= max_weight
    """
    import cvxpy as cp

    T, N = R_scenarios.shape
    w = cp.Variable(N)
    alpha_var = cp.Variable()
    z = cp.Variable(T)

    portfolio_returns = R_scenarios @ w
    cvar = alpha_var + (1.0 / ((1 - beta) * T)) * cp.sum(z)

    obj_expr = cvar
    if w_prev is not None:
        obj_expr += gamma_turnover * cp.norm1(w - w_prev)

    objective = cp.Minimize(obj_expr)

    constraints = [
        z >= 0,
        z >= -portfolio_returns - alpha_var,
        mu @ w >= r_target,
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    if w_prev is not None and max_turnover < 2.0:
        constraints.append(cp.norm1(w - w_prev) <= max_turnover)

    if sector_B is not None:
        for j in range(sector_B.shape[1]):
            constraints.append(
                cp.sum(cp.multiply(sector_B[:, j], w)) <= max_sector_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None

    return np.array(w.value).flatten()


# =====================================================================
# Optimizer 3: Robust Mean-CVaR (Main Contribution — SOCP)
# =====================================================================

def compute_kappa(N: int, T: int, confidence: float = 0.95) -> float:
    """
    Theoretical kappa calibration from chi-squared distribution.
    kappa = sqrt(chi2_quantile(N, confidence) / T)
    """
    return np.sqrt(chi2.ppf(confidence, df=N) / T)


def robust_cvar_optimizer(
    mu: np.ndarray,
    Sigma: np.ndarray,
    R_scenarios: np.ndarray,
    r_target: float,
    kappa: Optional[float] = None,
    beta: float = CVAR_BETA,
    w_prev: Optional[np.ndarray] = None,
    gamma_turnover: float = 0.005,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    max_turnover: float = MAX_TURNOVER,
    sector_B: Optional[np.ndarray] = None,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> Optional[np.ndarray]:
    """
    Robust CVaR with ellipsoidal uncertainty on expected returns.
    Penalty: kappa * ||Sigma_mu^{1/2} @ w||_2  (SOCP term)
    where Sigma_mu = Sigma / T (estimation error of mu).

    min CVaR + kappa * ||Sigma_mu_sqrt @ w||_2
    This is a Second-Order Cone Program.
    """
    import cvxpy as cp

    T, N = R_scenarios.shape

    if kappa is None:
        kappa = compute_kappa(N, T)

    Sigma_mu = Sigma / T
    Sigma_mu_sqrt = np.real(sqrtm(Sigma_mu + np.eye(N) * 1e-10))

    w = cp.Variable(N)
    alpha_var = cp.Variable()
    z = cp.Variable(T)

    portfolio_returns = R_scenarios @ w
    cvar = alpha_var + (1.0 / ((1 - beta) * T)) * cp.sum(z)

    robust_penalty = kappa * cp.norm(Sigma_mu_sqrt @ w, 2)

    obj_expr = cvar + robust_penalty
    if w_prev is not None:
        obj_expr += gamma_turnover * cp.norm1(w - w_prev)

    objective = cp.Minimize(obj_expr)

    constraints = [
        z >= 0,
        z >= -portfolio_returns - alpha_var,
        mu @ w >= r_target,
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    if w_prev is not None and max_turnover < 2.0:
        constraints.append(cp.norm1(w - w_prev) <= max_turnover)

    if sector_B is not None:
        for j in range(sector_B.shape[1]):
            constraints.append(
                cp.sum(cp.multiply(sector_B[:, j], w)) <= max_sector_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None

    return np.array(w.value).flatten()


# =====================================================================
# Optimizer 4: Equal Risk Contribution (ERC) — scipy SLSQP
# =====================================================================

def erc_optimizer(
    Sigma: np.ndarray,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
) -> np.ndarray:
    """
    Equal Risk Contribution (Risk Parity) portfolio.
    Each stock contributes equally to portfolio volatility.
    Uses scipy.optimize.minimize with SLSQP — no expected returns needed.
    """
    N = Sigma.shape[0]

    def erc_objective(w):
        w = np.array(w)
        port_var = w @ Sigma @ w
        if port_var <= 0:
            return 1e10
        port_vol = np.sqrt(port_var)
        marginal_risk = Sigma @ w / port_vol
        risk_contributions = w * marginal_risk
        target_rc = port_vol / N
        return np.sum((risk_contributions - target_rc) ** 2)

    eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(max(min_weight, 1e-4), min(max_weight, 1.0))] * N
    w0 = np.ones(N) / N

    result = minimize(
        erc_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[eq_constraint],
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    w_erc = result.x
    w_erc = np.maximum(w_erc, 0)
    w_erc = w_erc / w_erc.sum()

    return w_erc


# =====================================================================
# Solve with Progressive Fallback
# =====================================================================

def solve_with_fallback(
    mu: np.ndarray,
    Sigma: np.ndarray,
    R_scenarios: np.ndarray,
    optimizer_name: str,
    r_target: float = 0.0,
    kappa: Optional[float] = None,
    w_prev: Optional[np.ndarray] = None,
    sector_B: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple:
    """
    Run optimizer with progressive constraint relaxation on infeasibility.
    Relaxation order: min_weight → turnover → sector → equal weight fallback.
    Returns (weights, method_used).
    """
    N = len(mu)

    common = dict(
        w_prev=w_prev,
        sector_B=sector_B,
        **kwargs,
    )

    if optimizer_name == "mean_variance":
        w = mean_variance_optimizer(mu, Sigma, **common)
        if w is not None:
            return w, "mean_variance"

        w = mean_variance_optimizer(mu, Sigma, min_weight=0.0, **{
            k: v for k, v in common.items() if k != "min_weight"})
        if w is not None:
            return w, "mean_variance_relaxed_minw"

    elif optimizer_name == "mean_cvar":
        w = mean_cvar_optimizer(mu, R_scenarios, r_target, **common)
        if w is not None:
            return w, "mean_cvar"

        w = mean_cvar_optimizer(mu, R_scenarios, r_target,
                                min_weight=0.0, **{
                                    k: v for k, v in common.items()
                                    if k != "min_weight"})
        if w is not None:
            return w, "mean_cvar_relaxed_minw"

        w = mean_cvar_optimizer(mu, R_scenarios, r_target,
                                min_weight=0.0, max_turnover=2.0, **{
                                    k: v for k, v in common.items()
                                    if k not in ("min_weight", "max_turnover")})
        if w is not None:
            return w, "mean_cvar_relaxed_all"

    elif optimizer_name == "robust_cvar":
        w = robust_cvar_optimizer(mu, Sigma, R_scenarios, r_target,
                                  kappa=kappa, **common)
        if w is not None:
            return w, "robust_cvar"

        w = robust_cvar_optimizer(mu, Sigma, R_scenarios, r_target,
                                  kappa=kappa, min_weight=0.0, **{
                                      k: v for k, v in common.items()
                                      if k != "min_weight"})
        if w is not None:
            return w, "robust_cvar_relaxed_minw"

        w = robust_cvar_optimizer(mu, Sigma, R_scenarios, r_target,
                                  kappa=kappa, min_weight=0.0,
                                  max_turnover=2.0, **{
                                      k: v for k, v in common.items()
                                      if k not in ("min_weight", "max_turnover")})
        if w is not None:
            return w, "robust_cvar_relaxed_all"

    elif optimizer_name == "erc":
        w = erc_optimizer(Sigma,
                          max_weight=kwargs.get("max_weight", MAX_WEIGHT),
                          min_weight=kwargs.get("min_weight", MIN_WEIGHT))
        return w, "erc"

    w_eq = np.ones(N) / N
    return w_eq, "equal_weight_fallback"
