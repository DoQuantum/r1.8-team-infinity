"""
Stage 6: Constraints and Practical Requirements

Constraint helpers:
- Sector mapping and constraint matrix
- ADV-based liquidity cap
- Turnover computation
"""
import numpy as np
import pandas as pd
from typing import Optional
from config import MAX_WEIGHT, MAX_SECTOR_WEIGHT


# =====================================================================
# Sector Mapping
# =====================================================================

def get_sector_mapping(tickers: list) -> dict:
    """Fetch GICS sector for each ticker from yfinance."""
    import yfinance as yf
    from utils_ticker_map import bloomberg_to_yfinance

    mapping = {}
    for t in tickers:
        yf_t = bloomberg_to_yfinance(t) if " " in str(t) else t
        try:
            info = yf.Ticker(yf_t).info
            sector = info.get("sector", "Unknown")
            mapping[t] = sector
        except Exception:
            mapping[t] = "Unknown"
    return mapping


def sector_constraint_matrix(
    tickers: list,
    sector_map: Optional[dict] = None,
) -> tuple:
    """
    Build sector dummy matrix B: (n_stocks, n_sectors).
    B[i,s] = 1 if stock i in sector s.
    Only creates constraints for sectors with >= 2 stocks.
    """
    if sector_map is None:
        sector_map = get_sector_mapping(tickers)

    sectors = sorted(set(sector_map.values()))
    N, K = len(tickers), len(sectors)
    B = np.zeros((N, K))
    for i, t in enumerate(tickers):
        s = sector_map.get(t, "Unknown")
        j = sectors.index(s)
        B[i, j] = 1

    cols_with_multiple = [j for j in range(K) if B[:, j].sum() >= 2]
    if cols_with_multiple:
        B_filtered = B[:, cols_with_multiple]
        sectors_filtered = [sectors[j] for j in cols_with_multiple]
    else:
        B_filtered = None
        sectors_filtered = []

    return B_filtered, sectors_filtered


# =====================================================================
# ADV Liquidity Cap
# =====================================================================

def adv_liquidity_cap(
    adv_values: np.ndarray,
    portfolio_value: float = 1e7,
    max_pct_adv: float = 0.05,
    upper_bound: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Compute per-stock weight cap based on average daily volume.
    w_i <= max_pct_adv * ADV_i / portfolio_value
    Capped at upper_bound.
    """
    cap = max_pct_adv * adv_values / (portfolio_value + 1e-10)
    cap = np.minimum(cap, upper_bound)
    cap = np.maximum(cap, 0.001)
    return cap


# =====================================================================
# Turnover Computation
# =====================================================================

def compute_turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """Two-way turnover: sum(|w_new - w_prev|)."""
    return np.abs(w_new - w_prev).sum()


def compute_one_way_turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """One-way turnover: sum of buys (or sells)."""
    return np.maximum(w_new - w_prev, 0).sum()
