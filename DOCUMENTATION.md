# Portfolio Optimization Research — Team Documentation

---

## What This Project Actually Does

We're building a **research-grade portfolio optimization system** that selects 30 diversified stocks from the S&P 500 and compares four different optimization strategies head-to-head. The core research question: **does Robust CVaR optimization produce better risk-adjusted returns than simpler methods (Mean-Variance, standard CVaR, Risk Parity) in a realistic backtesting environment?**

The pipeline runs a rolling monthly backtest from Jan 2020 to Dec 2023, rebalancing every month. At each rebalance, it estimates expected returns using Black-Litterman with ML-generated views, estimates the covariance matrix using shrinkage, and then runs all four optimizers to produce portfolio weights. We track out-of-sample returns and compare.

---

## Quick Start

```bash
conda activate port7
python main_pipeline.py
```

Output files land in the `data/` folder:
- `final_weights.csv` — last-period weights for all 4 strategies
- `performance_summary.csv` — annualized return, vol, Sharpe, max drawdown per strategy
- `oos_returns.csv` — monthly out-of-sample returns (the real test)
- `risk_attribution_robust.csv` — per-stock risk decomposition for the Robust CVaR portfolio

---

## Project Structure

```
portfoliofinal/
├── config.py                  # All parameters in one place (tau, kappa, constraints, etc.)
├── portfo77.py                # Original stock selection script (standalone, not imported)
├── data_loader.py             # Stage 1: Bloomberg CSV parser + yfinance macro data
├── fundamentals_loader.py     # Stage 1.2: Quarterly fundamentals from yfinance
├── feature_engineering.py     # Stage 2: Momentum, risk, liquidity features + composites
├── covariance_estimation.py   # Stage 4: Ledoit-Wolf shrinkage + VIX regime blend
├── expected_returns.py        # Stage 3: Black-Litterman + Elastic-Net ML views
├── optimization_engines.py    # Stage 5: MV, CVaR, Robust CVaR, ERC optimizers
├── constraints.py             # Stage 6: Sector, turnover, liquidity constraints
├── risk_metrics.py            # Stage 7: VaR, CVaR, stress tests, attribution
├── main_pipeline.py           # The orchestrator — runs the full rolling backtest
├── utils_ticker_map.py        # Bloomberg "LYB UN Equity" → yfinance "LYB" mapping
├── requirements.txt           # Dependencies
└── data/                      # Output folder (created at runtime)
```

---

## The 7 Stages, Explained Simply

### Stage 1: Data Collection (`data_loader.py`, `fundamentals_loader.py`)

**What it does:** Loads daily OHLCV prices from a Bloomberg Terminal CSV export, selects 30 maximally diversified stocks using K-Medoids clustering on a correlation-distance matrix, then pulls supplementary data (market caps, fundamentals, VIX, SPY benchmark, risk-free rate) from yfinance.

**The stock selection trick:** We don't just pick 30 stocks at random. We compute a correlation-distance matrix across all S&P 500 constituents (distance = sqrt(2 * (1 - correlation))), then run K-Medoids clustering with 30 clusters. Each cluster's medoid becomes our representative stock. This gives us a naturally diversified universe where stocks are as uncorrelated as possible. The logic lives in `portfo77.py` (the original standalone script) and is replicated in `data_loader.get_selected_tickers()`.

**Known limitation — be aware:** yfinance fundamentals use fiscal period end dates, not actual earnings release dates. This creates look-ahead bias (we "know" the quarter's numbers before they were actually public). For production, you'd need Bloomberg or FactSet with actual release dates. We document this in the paper as a limitation.

**Data quality checks:**
- Stocks with >5% missing price data get flagged
- Daily returns exceeding |20%| get flagged (could be real events or data errors — we flag but don't auto-remove)
- Stocks with <99% coverage in the training window get dropped before clustering (prevents IPO artifacts like DASH or COIN from sneaking in with backfilled flat lines)

---

### Stage 2: Feature Engineering (`feature_engineering.py`)

**What it does:** Computes ~18 cross-sectional features per stock per date, Z-scores them across the 30 stocks at each date, and organizes them into a panel for the ML model.

**Feature groups:**

| Category | Features | Why These |
|----------|----------|-----------|
| **Momentum** | 12-2 month return, 1-month return, 52-week high ratio | Standard academic momentum factors. 12-2 skips the last month to avoid short-term reversal contamination. |
| **Risk** | 21d realized vol, 60d beta, 60d idiosyncratic vol, 120d skewness | Beta and idio vol are needed for the BL prior. We use 120d for skewness (not 60d) because 3rd moments need more data. |
| **Liquidity** | Amihud illiquidity (21d avg), ADV (21d avg) | Amihud is the standard academic liquidity measure. ADV is used for position-sizing constraints. |
| **Fundamental** | ROE, ROA, Gross Margin, EBIT/EV, Revenue Growth, D/E, 1/PE, 1/PB, delta-ROE | We invert P/E and P/B so that higher = cheaper = more attractive (consistent sign direction). |

**Cross-sectional Z-scoring is critical.** At each date, every feature is standardized across the 30 stocks (mean 0, std 1). This removes time-series level shifts and focuses on *relative attractiveness* — stock A is cheap compared to the other 29, not compared to its own history. This is the right normalization for cross-sectional regression.

**What we tried and dropped (and why):**

| Dropped | Reason |
|---------|--------|
| 5-day returns | No clean interpretation at monthly rebalancing frequency — just noise |
| 1-day reversal | Same issue — monthly rebalancing can't exploit daily reversals |
| Kurtosis (60d) | 4th moment estimation with 60 observations is extremely noisy. Added no value over skewness. |
| Max drawdown | Highly correlated with realized volatility — redundant signal |
| Corwin-Schultz bid-ask spread | Designed for less liquid markets. Our 30 stocks are large-cap S&P 500 — liquidity isn't the binding constraint. The OHLC-based estimator also has known biases with modern market microstructure. |
| Forward P/E | Requires analyst consensus estimates (I/B/E/S via WRDS). Not available in yfinance. |
| Price-to-Sales | Correlated with P/E and P/B — adding a third valuation ratio with 30 stocks creates multicollinearity. |
| Interest Coverage, Current Ratio | Correlated with D/E ratio. For large-cap industrials/tech, these are less informative. |
| EPS Revision | Requires historical consensus time-series. Complex data pipeline for marginal value. |

---

### Stage 3: Expected Returns (`expected_returns.py`)

This is where the Black-Litterman model comes together. Three layers:

**Layer A — Equilibrium Prior (what the market "thinks"):**

The idea: if the market is in equilibrium, the expected returns that would make a market-cap-weighted investor happy are Pi = lambda * Sigma * w_mkt. We reverse-engineer what the market "expects" each stock to return, given its weight in the universe and the covariance structure.

- lambda (risk aversion) is computed from actual SPY data: lambda = E[excess return] / Var(return). Typical value: 2.5-3.5. We fallback to 2.5 if the data gives something unreasonable.
- w_mkt = market_cap_i / sum(market_caps) — each stock's weight proportional to its market cap *within our 30-stock universe*.

**Layer B — ML Alpha Signal (our "views"):**

We train an Elastic-Net regression to predict next-month excess returns using the feature matrix from Stage 2.

Why Elastic-Net and not LightGBM/CatBoost?
- We have 30 stocks. Cross-sectionally, that's 30 data points per month. Even with pooling across months (30 stocks x 24 months = 720 training observations), a gradient boosted tree with depth 4+ will memorize the training data.
- Elastic-Net handles multicollinearity (our features are correlated) through L1+L2 regularization.
- Coefficients are interpretable for the research paper ("higher ROE predicts higher returns" is a testable hypothesis).
- We initially tried a 3-model ensemble (Elastic-Net + LightGBM + CatBoost). It was overkill and overfitted. We dropped the tree models.

Training protocol:
- Expanding window: at month t, train on all (stock, month) pairs from month 1 to month t-2
- Embargo: skip month t-1 to prevent overlap between training targets and the test period
- Sample weighting: exponential decay with 12-month half-life (recent data matters more)
- Time-series cross-validation: 5-fold chronological split
- Hyperparameter tuning: grid search over regularization strength alpha

After prediction, we **scale the signal** to a target cross-sectional standard deviation of 0.5% per month. This preserves the stock ranking while controlling the magnitude so it doesn't overwhelm the equilibrium prior in the BL formula.

**Layer C — Black-Litterman Posterior (blending market and ML views):**

The BL formula with P = I (stock-level views):

```
mu_BL = [(tau*Sigma)^-1 + Omega^-1]^-1 @ [(tau*Sigma)^-1 @ Pi + Omega^-1 @ q]
```

**Key parameter: tau = 0.05.** This is the standard choice in BL literature (He & Litterman 1999). It means the uncertainty in the prior Pi is 5% of the uncertainty in returns. We initially used tau = 0.01, which over-weighted the prior and barely moved the portfolio from market-cap weights. Sensitivity analysis across tau in {0.01, 0.025, 0.05, 0.1} is recommended for the paper.

**Critical correction — Omega:** The original spec had Omega_ii = (CV RMSE for stock i)^2. This is dimensionally inconsistent unless RMSE is measured in the same units as the covariance matrix. We switched to the He-Litterman proportional method: Omega = diag(tau * Sigma). This is cleaner — view uncertainty is proportional to prior uncertainty — and has one less tuning parameter.

Post-processing: clip mu_BL at +/- 2.5 sigma (not 3.0) because with 30 stocks, 3-sigma still allows extreme values.

---

### Stage 4: Covariance Estimation (`covariance_estimation.py`)

**Primary method: Ledoit-Wolf shrinkage on non-overlapping monthly returns.**

This is worth explaining because it's a subtle but important choice.

We initially estimated daily covariance and scaled: Sigma_monthly = 21 * Sigma_daily. This scaling only works if returns are i.i.d. — which they are not (autocorrelation, volatility clustering). Using actual non-overlapping monthly returns avoids this assumption entirely.

With 48 months of data and 30 stocks, the ratio T/N = 48/30 = 1.6. This is below the random matrix theory threshold of ~5 where sample covariance becomes reliable. This is *exactly* why we use Ledoit-Wolf shrinkage — it regularizes the covariance by shrinking toward a structured target (constant correlation matrix), preventing the extreme eigenvalues that make sample covariance unstable.

**Regime-conditional adjustment:** We blend two Ledoit-Wolf estimates — one from "normal" months (VIX <= 25) and one from "stress" months (VIX > 25). The blend weight depends on the current VIX level. This captures the well-documented phenomenon that correlations spike during stress (COVID crash, rate hike selloff).

Why VIX > 25 and not an HMM?
- HMM with 2 states sounds sophisticated but has identification problems — the model may not cleanly separate regimes when fit on a rolling window, especially in early 2020 when it hadn't seen stress data yet.
- VIX > 25 has documented support in the literature and is transparent/reproducible.
- If there aren't enough stress observations (fewer than N+1 months), we scale the normal covariance by 2x. Simple, robust, honest.

**PCA validation:** We run PCA on the monthly returns to check if the first 3-5 components explain >50% of variance. If yes, Ledoit-Wolf is capturing real factor structure. If not, we'd consider an explicit factor model (Fama-French 5). This is a diagnostic check, not a replacement.

**What we dropped:**
- DCC-GARCH: 2-4 weeks of implementation for marginal improvement. The numerical optimization for DCC with 30 series is finicky and hard to debug. Not worth it.
- HMM regime detection: replaced by the simpler VIX threshold rule (see above).

---

### Stage 5: Optimization (`optimization_engines.py`)

We run 4 optimizers at every rebalance date. This comparison is the core academic contribution.

**Optimizer 1: Mean-Variance (benchmark)**
The Markowitz classic. Maximize mu'w - lambda * w'Sigma*w. We include turnover penalty and constraints. This is the baseline everyone knows — if our fancier methods can't beat this, we have a problem.

**Optimizer 2: Mean-CVaR (primary)**
Minimize CVaR at 95% confidence using the Rockafellar-Uryasev LP reformulation. Instead of just penalizing variance (which treats upside and downside symmetrically), CVaR focuses on the worst 5% of outcomes. The scenario matrix is the actual historical monthly returns — no simulation.

The LP formulation:
```
min  alpha + 1/((1-beta)*T) * sum(z_t)
s.t. z_t >= 0
     z_t >= -R_t @ w - alpha    (captures losses exceeding VaR)
     mu'w >= r_target            (minimum return constraint)
     sum(w) = 1, 0.01 <= w <= 0.10
```

**Optimizer 3: Robust CVaR (our main contribution)**
Same as CVaR but adds a penalty for estimation uncertainty in the expected returns:

```
min  CVaR + kappa * ||Sigma_mu^{1/2} @ w||_2
```

This is a Second-Order Cone Program (SOCP). The penalty term kappa * ||Sigma_mu_sqrt @ w||_2 says: "I'm not sure about my expected return estimates, so penalize portfolios that are heavily exposed to stocks where my estimates are most uncertain."

kappa is calibrated from chi-squared distribution: kappa = sqrt(chi2_quantile(N, 0.95) / T). This has a clean statistical interpretation — we're hedging against the 95% worst case for estimation error.

**Optimizer 4: Equal Risk Contribution / Risk Parity (model-free benchmark)**
Each stock contributes equally to total portfolio risk. This optimizer doesn't use expected returns at all — only the covariance matrix. It's solved with scipy's SLSQP (not cvxpy) because the ERC objective is nonlinear and non-convex. We verified that risk contributions are equalized to within 1e-7 standard deviation.

Why scipy and not cvxpy for ERC? cvxpy requires convex formulations. The ERC objective involves dividing by portfolio volatility and multiplying by weights — this creates a ratio that isn't DCP-compliant. scipy's SLSQP handles it directly.

**Progressive fallback:** If any optimizer hits infeasibility (usually from conflicting constraints), we relax in order: (1) drop minimum weight, (2) relax turnover limit, (3) fall back to equal weight. This is logged so we know when it happens.

**Solver note:** We use CLARABEL (the default solver in cvxpy 1.7+). We originally hardcoded ECOS but it's no longer bundled. CLARABEL handles QP, LP, and SOCP problems out of the box.

---

### Stage 6: Constraints (`constraints.py`)

All constraints are enforced inside the optimizer, not as post-processing. Constraint parameters live in `config.py`.

| Constraint | Value | Type | Notes |
|-----------|-------|------|-------|
| Full investment | sum(w) = 1 | Hard | Never relax |
| Long only | w >= 0 | Hard | Never relax |
| Max weight | w <= 10% | Hard | Prevents concentration |
| Min weight | w >= 1% | Semi-hard | First to relax on infeasibility |
| Sector cap | <= 30% per GICS sector | Semi-hard | Only for sectors with 2+ stocks |
| Turnover | sum(\|w_new - w_prev\|) <= 30% | Soft | Two-way turnover; relaxed second |
| Turnover penalty | gamma * \|\|w - w_prev\|\|_1 in objective | Soft | gamma = 0.005 |

**Why 1% minimum weight?** With 30 stocks and min 1% each, that's 30% locked in the lower bounds, leaving 70% to allocate. With max 10%: 30 * 3.33% average fits easily. But combined with sector constraints, infeasibility can occur — hence the progressive relaxation.

**ADV liquidity cap (available but not yet integrated into main loop):** w_i <= 5% * ADV_i / portfolio_value. For large-cap S&P 500 stocks this is rarely binding, but it's there for robustness.

---

### Stage 7: Risk Measurement (`risk_metrics.py`)

**Ex-ante risk metrics (computed at each rebalance):**
- Portfolio volatility (monthly, annualized)
- Sharpe ratio (annualized)
- VaR at 95%: parametric (Gaussian), historical, t-distribution
- CVaR at 95%: parametric, historical
- Jarque-Bera normality test: if rejected (p < 0.05), we recommend historical CVaR over parametric. This matters because equity returns have fat tails — the Gaussian CVaR understates the true tail risk.

**Component CVaR (Euler decomposition):**
Breaks total portfolio CVaR into per-stock contributions. Uses the property that CVaR is a positive homogeneous risk measure, so Euler's theorem applies: component_cvar_i = w_i * E[-R_i | portfolio loss >= VaR]. The sum of components equals total CVaR — we verify this at runtime.

**Stress testing:**
We apply the final portfolio weights to actual daily returns during historical stress episodes:
- **COVID Crash** (Feb 20 - Mar 23, 2020): The liquidity crisis test
- **2022 Rate Hikes** (Jan 1 - Dec 31, 2022): The duration/growth rotation test
- **Q4 2018 Selloff** (Oct 1 - Dec 31, 2018): The volatility spike test

For each scenario, we report total return, max drawdown, worst single day, and CVaR.

---

## Key Parameters (all in `config.py`)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `TAU` | 0.05 | Standard BL literature (He & Litterman 1999). Controls prior vs. view weighting. |
| `CLIP_SIGMA` | 2.5 | Tighter than 3.0 because N=30 is small — 3-sigma still allows extremes |
| `CVAR_BETA` | 0.95 | Standard confidence level for tail risk |
| `VIX_STRESS_THRESHOLD` | 25 | Documented in literature as elevated stress boundary |
| `MAX_WEIGHT` | 0.10 | 10% max prevents concentration |
| `MIN_WEIGHT` | 0.01 | 1% min prevents dust positions |
| `MAX_TURNOVER` | 0.30 | 30% two-way turnover is practical for monthly rebalancing |
| `MAX_SECTOR_WEIGHT` | 0.30 | 30% sector cap prevents overexposure |
| `MIN_TRAIN_MONTHS` | 24 | 2 years minimum training before going out-of-sample |
| `ELASTIC_NET_L1_RATIO` | 0.5 | Equal L1/L2 penalty (true elastic net) |
| `SAMPLE_WEIGHT_HALFLIFE_MONTHS` | 12 | 1-year halflife for sample decay |
| `CV_FOLDS` | 5 | 5-fold time-series CV for hyperparameter tuning |

---

## Trial and Error Log

Here's an honest account of what we tried, what broke, and what we learned.

### 1. The LightGBM/CatBoost Experiment

**What we tried:** Built a 3-model ensemble (Elastic-Net + LightGBM + CatBoost) for the alpha model.

**What happened:** With 30 stocks per cross-section, even a shallow tree (depth 4) memorized the training set. The ensemble had great in-sample IC (0.15+) but out-of-sample IC near zero. Classic overfitting.

**What we learned:** N=30 is too small for nonlinear models to discover meaningful interactions. Elastic-Net's linear structure is actually a feature here, not a limitation — it can't overfit as badly, and the coefficients are interpretable.

**Resolution:** Dropped both tree models. Elastic-Net only. If anyone wants to revisit this, try Random Forest with max_depth=2 and min_samples_leaf=20 as a robustness check.

### 2. The Omega Debacle

**What we tried:** Set Omega_ii = (cross-validation RMSE for stock i)^2 as view uncertainty in BL.

**What went wrong:** The RMSE from the ML model is in 21-day return units. But if you're working with monthly covariance matrix, Sigma is also in monthly units. Mixing these doesn't crash — it just silently produces wrong portfolio weights that over- or under-weight the ML signal relative to the equilibrium prior.

**What we learned:** Dimensional consistency is non-negotiable. The He-Litterman proportional Omega (Omega = diag(tau * Sigma)) is self-consistent by construction.

**Resolution:** Switched to proportional Omega. One less hyperparameter, cleaner theory.

### 3. The Daily-to-Monthly Covariance Scaling Trap

**What we tried:** Estimated Sigma from daily returns and scaled: Sigma_monthly = 21 * Sigma_daily.

**Why it's wrong:** This assumes returns are i.i.d. (no autocorrelation, no volatility clustering). Real equity returns have both. The scaling inflates/deflates the covariance depending on whether volatility clustering is positive or negative during the estimation window.

**Resolution:** Compute Sigma directly from non-overlapping monthly returns. With T=48 months and N=30 stocks, Ledoit-Wolf handles the low T/N ratio.

### 4. The DRO Wasserstein Attempt

**What we tried:** Implemented Distributionally Robust Optimization with Wasserstein uncertainty sets, per Mohajerin-Esfahani & Kuhn (2018).

**What went wrong:** The norm used in the regularization (L1 vs L2 vs L-infinity) depends on the exact Wasserstein distance definition. Different papers use different formulations. We had a subtle L1/L-infinity mismatch that produced wrong dual norms. The mathematical complexity was high and the empirical improvement over our simpler Robust CVaR was negligible for 30 stocks.

**Resolution:** Dropped DRO entirely. The Robust CVaR with chi-squared kappa calibration provides the same intuition (hedge against estimation uncertainty) with a cleaner implementation. If someone wants to pursue DRO, read the original Mohajerin-Esfahani & Kuhn paper very carefully and verify the norm pairings.

### 5. The ECOS Solver Disappearance

**What happened:** Upgraded cvxpy to 1.7.5 and all optimizers broke with "ECOS solver not installed."

**Why:** cvxpy 1.7+ no longer bundles ECOS by default. The new default solver is CLARABEL.

**Resolution:** Replaced all `solver=cp.ECOS` with `solver=cp.CLARABEL`. CLARABEL handles QP, LP, and SOCP natively. No performance difference for our problem sizes.

### 6. The HMM Regime Detection Dead End

**What we tried:** Hidden Markov Model with 2 states (normal/stress) fit on VIX + realized vol + return sign.

**What happened:** The HMM had convergence issues with different random initializations giving different regime assignments. When fit on a rolling window, it couldn't cleanly identify COVID as stress in early 2020 (it had only seen normal data up to that point).

**Resolution:** Replaced with simple VIX > 25 threshold rule. It's transparent, reproducible, and captures the main effect (correlation spikes in stress) without the identification headaches. The VIX threshold is supported by academic literature.

### 7. The Date Alignment Bug

**What happened:** Monthly return dates (calendar month-end from `resample('ME')`) didn't match trading day dates (last business day of month). The Elastic-Net training function was looking up features by calendar dates in a dictionary keyed by trading dates, causing silent key misses and empty training sets.

**Resolution:** Separated `feature_dates` (trading days for feature lookup) from `return_dates` (calendar month-ends for return lookup) in `build_training_data()`. The function now accepts both and uses the correct one for each lookup.

---

## Sensitivity Analyses (Required for Paper)

The following sensitivity tests should be run and reported. The pipeline includes helper functions for the first two:

1. **tau sensitivity:** `run_tau_sensitivity()` — test tau in {0.01, 0.025, 0.05, 0.1}. Show that portfolio weights are not dramatically different. This proves the results aren't fragile to prior calibration.

2. **kappa sensitivity:** `run_kappa_sensitivity()` — test kappa in {0.1, 0.5, 1.0, 2.0, 5.0}. Plot the robustness-performance tradeoff curve (higher kappa = more conservative = lower Sharpe but better stress performance). This is a key figure.

3. **IC-weighted vs. equal-weighted composites** — Test if weighting the 4 composites (Value, Quality, Momentum, Growth) by their rolling IC improves out-of-sample Sharpe. Prediction: with N=30, IC estimates are too noisy and equal weights will win.

4. **Sample covariance vs. Ledoit-Wolf** — Compare out-of-sample portfolio variance. LW should win, especially in earlier months when T/N is lowest.

5. **Regime adjustment on/off** — Run the backtest with and without the VIX-based covariance regime blend. Report the difference in stress-period performance (COVID, 2022).

---

## Known Limitations (Document in Paper)

1. **Look-ahead bias in fundamentals.** yfinance provides fiscal-period-end dates, not actual release dates. A stock's Q3 earnings might be known in our model on Sept 30 when they weren't actually released until Oct 25. For true point-in-time data, you need Bloomberg/COMPUSTAT via WRDS.

2. **Small cross-section (N=30).** This limits the power of ML models and makes IC estimates noisy. It's a deliberate choice — we want a concentrated, diversified portfolio — but it constrains methodology.

3. **No transaction costs.** We have turnover constraints but don't model actual trading costs (bid-ask spread, market impact). For large-cap S&P 500 stocks with monthly rebalancing, this is a minor issue but should be noted.

4. **Market cap is a snapshot, not time-varying.** We fetch current market caps from yfinance, not historical. The equilibrium prior uses these throughout the backtest. In reality, market caps change over time.

5. **No survivorship bias handling.** The Bloomberg CSV contains current constituents. Stocks that were removed from the S&P 500 during 2020-2023 are not included. This is a mild bias toward stocks that survived.

---

## Dependencies

All packages are in `requirements.txt`. Key ones:

| Package | Version | What For |
|---------|---------|----------|
| pandas | >= 2.0 | Data manipulation |
| numpy | >= 1.24 | Numerical computation |
| scikit-learn | >= 1.3 | Ledoit-Wolf, Elastic-Net, PCA, TimeSeriesSplit |
| scikit-learn-extra | >= 0.3 | K-Medoids clustering for stock selection |
| yfinance | >= 0.2.28 | Market caps, fundamentals, macro data |
| scipy | >= 1.11 | Chi-squared quantile, matrix square root, SLSQP optimizer, statistical tests |
| cvxpy | >= 1.4 | Convex optimization (MV, CVaR, Robust CVaR) |
| statsmodels | >= 0.14 | Statistical tests (available but not heavily used) |

Install everything in the port7 conda env:
```bash
conda activate port7
pip install -r requirements.txt
```

---

## References

1. **Black, F. and Litterman, R.** (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28-43. — The original BL paper.

2. **He, G. and Litterman, R.** (1999). "The Intuition Behind Black-Litterman Model Portfolios." *Goldman Sachs Investment Management Research*. — Proportional Omega formulation we use.

3. **Ledoit, O. and Wolf, M.** (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365-411. — The shrinkage estimator we use for covariance.

4. **Rockafellar, R.T. and Uryasev, S.** (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21-42. — LP formulation for CVaR optimization.

5. **Maillard, S., Roncalli, T. and Teiletche, J.** (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management*, 36(4), 60-70. — ERC/Risk Parity theory.

6. **Mohajerin Esfahani, P. and Kuhn, D.** (2018). "Data-driven distributionally robust optimization using the Wasserstein metric." *Mathematical Programming*, 171, 115-166. — DRO formulation (we tried and dropped this, but cite it to explain why).

7. **Zou, H. and Hastie, T.** (2005). "Regularization and variable selection via the elastic net." *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320. — Elastic-Net regression.

8. **Amihud, Y.** (2002). "Illiquidity and stock returns: cross-section and time-series effects." *Journal of Financial Markets*, 5(1), 31-56. — The Amihud illiquidity measure.

9. **Jegadeesh, N. and Titman, S.** (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1), 65-91. — The 12-2 month momentum factor.

10. **George, T.J. and Hwang, C.Y.** (2004). "The 52-Week High and Momentum Investing." *Journal of Finance*, 59(5), 2145-2176. — The 52-week high ratio as momentum signal.

---

## Questions?

If anything is unclear, check the docstrings in the relevant module first — every function is documented with its inputs, outputs, and mathematical formulation. For conceptual questions about *why* we made a particular choice, check the Trial and Error Log above. For parameter questions, check `config.py` — everything is in one place with the reasoning in this doc.
