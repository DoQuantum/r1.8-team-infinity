"""
Stage 1: Data Collection and Preparation
- Bloomberg CSV price data (OHLCV)
- Market cap from yfinance
- Macro overlay: VIX, 10yr yield, DXY from yfinance
- Benchmark: SPY total return
- Risk-free rate: 3-month T-bill proxy
- Data quality checks
"""
import os
import numpy as np
import pandas as pd
from typing import Optional
from config import BLOOMBERG_CSV_PATH, START_DATE, END_DATE, N_STOCKS
from utils_ticker_map import bloomberg_to_yfinance, get_ticker_mapping


# =====================================================================
# Bloomberg CSV Loading
# =====================================================================

def _load_bloomberg_header(csv_path: str):
    """Parse Bloomberg CSV header structure."""
    raw_header = pd.read_csv(csv_path, nrows=10, header=None)
    metric_row_idx = None
    for r in range(len(raw_header)):
        if str(raw_header.iloc[r, 0]).strip() == "Dates":
            metric_row_idx = r
            break
    if metric_row_idx is None:
        raise ValueError("Could not find 'Dates' header row in Bloomberg CSV")
    equity_row_idx = max(0, metric_row_idx - 2)
    equity_row = raw_header.iloc[equity_row_idx]
    metric_row = raw_header.iloc[metric_row_idx]

    col_specs = []
    for i in range(1, len(metric_row), 5):
        if i + 4 >= len(equity_row):
            break
        metric = str(metric_row.iloc[i]).strip()
        if metric != "PX_LAST":
            continue
        eq_name = str(equity_row.iloc[i]).strip()
        if not eq_name or "nan" in eq_name.lower():
            continue
        if "Equity" not in eq_name and "UW" not in eq_name:
            continue
        if eq_name.upper() == "XYZ UN EQUITY":
            continue
        m_map = {}
        for k in range(5):
            m = str(metric_row.iloc[i + k]).strip()
            m_map[m] = i + k
        col_specs.append((i, eq_name, m_map))
    return metric_row_idx, col_specs


def load_bloomberg_price_data(
    csv_path: str = BLOOMBERG_CSV_PATH,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple:
    """
    Load OHLCV from Bloomberg CSV.
    Returns (adj_close, open_, high, low, volume, equity_names).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Bloomberg CSV not found: {csv_path}")

    metric_row_idx, col_specs = _load_bloomberg_header(csv_path)
    if not col_specs:
        raise ValueError("No PX_LAST columns found in Bloomberg CSV")

    df_raw = pd.read_csv(csv_path, skiprows=metric_row_idx + 1,
                         header=None, low_memory=False)
    df_raw[0] = pd.to_datetime(df_raw[0], format="%m/%d/%Y", errors="coerce")
    df_raw = df_raw.dropna(subset=[0]).set_index(0)
    df_raw.index.name = "Date"

    adj_close_d, open_d, high_d, low_d, volume_d = {}, {}, {}, {}, {}
    for px_col, eq_name, m_map in col_specs:
        if px_col >= df_raw.shape[1]:
            continue
        adj_close_d[eq_name] = pd.to_numeric(df_raw[px_col], errors="coerce")
        for field, key in [("PX_OPEN", "open"), ("PX_HIGH", "high"),
                           ("PX_LOW", "low"), ("PX_VOLUME", "volume")]:
            if field in m_map and m_map[field] < df_raw.shape[1]:
                target = {"open": open_d, "high": high_d,
                          "low": low_d, "volume": volume_d}[key]
                target[eq_name] = pd.to_numeric(df_raw[m_map[field]],
                                                errors="coerce")

    adj_close = pd.DataFrame(adj_close_d)
    open_ = pd.DataFrame(open_d).reindex(columns=adj_close.columns)
    high = pd.DataFrame(high_d).reindex(columns=adj_close.columns)
    low = pd.DataFrame(low_d).reindex(columns=adj_close.columns)
    volume = pd.DataFrame(volume_d).reindex(columns=adj_close.columns)

    mask = (adj_close.index >= start_date) & (adj_close.index <= end_date)
    adj_close = adj_close.loc[mask]
    open_ = open_.loc[mask]
    high = high.loc[mask]
    low = low.loc[mask]
    volume = volume.loc[mask]

    thresh = int(0.95 * len(adj_close))
    adj_close = adj_close.dropna(axis=1, thresh=thresh)
    for name in ["open_", "high", "low", "volume"]:
        pass  # reindex below
    open_ = open_.reindex(columns=adj_close.columns)
    high = high.reindex(columns=adj_close.columns)
    low = low.reindex(columns=adj_close.columns)
    volume = volume.reindex(columns=adj_close.columns)

    adj_close = adj_close.ffill().bfill()
    open_ = open_.ffill().bfill()
    high = high.ffill().bfill()
    low = low.ffill().bfill()
    volume = volume.ffill().bfill()

    return adj_close, open_, high, low, volume, list(adj_close.columns)


# =====================================================================
# Stock Selection (K-Medoids / K-Means from portfo77 logic)
# =====================================================================

def get_selected_tickers(
    csv_path: str = BLOOMBERG_CSV_PATH,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    n_clusters: int = N_STOCKS,
) -> list:
    """Run portfo77 selection logic to get diversified tickers."""
    from sklearn.cluster import KMeans

    adj_close, _, _, _, _, _ = load_bloomberg_price_data(
        csv_path, start_date, end_date)
    data = adj_close.dropna(axis=1, thresh=int(0.99 * len(adj_close)))
    data = data.ffill().bfill().dropna(axis=1)

    if data.shape[1] < n_clusters:
        raise ValueError(
            f"Only {data.shape[1]} stocks remain, need {n_clusters}")

    returns = data.pct_change().dropna()
    corr = returns.corr().fillna(0)
    dist = np.sqrt(np.clip(2 * (1 - corr.values), 0, None))

    try:
        from sklearn_extra.cluster import KMedoids
        km = KMedoids(n_clusters=n_clusters, metric="precomputed",
                      random_state=42, init="k-medoids++")
        km.fit(dist)
        selected = [data.columns[i] for i in km.medoid_indices_]
    except ImportError:
        from sklearn.manifold import MDS
        n_comp = max(2, min(50, len(corr) - 1, dist.shape[0] - 1))
        mds = MDS(n_components=n_comp, dissimilarity="precomputed",
                  random_state=42)
        coords = mds.fit_transform(dist)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(coords)
        selected = []
        for i in range(n_clusters):
            idx = np.where(kmeans.labels_ == i)[0]
            if len(idx) == 0:
                continue
            c = kmeans.cluster_centers_[i]
            d = [np.linalg.norm(coords[j] - c) for j in idx]
            selected.append(data.columns[idx[np.argmin(d)]])

    return selected


# =====================================================================
# Market Cap
# =====================================================================

def fetch_market_cap(tickers: list) -> pd.Series:
    """Fetch current market cap for each ticker via yfinance."""
    import yfinance as yf

    mapping = get_ticker_mapping(tickers) if any(
        " " in str(t) for t in tickers) else {t: t for t in tickers}
    result = {}
    for bbg, yf_t in mapping.items():
        if not yf_t:
            result[bbg] = np.nan
            continue
        try:
            info = yf.Ticker(yf_t).info
            result[bbg] = info.get("marketCap", np.nan)
        except Exception:
            result[bbg] = np.nan
    return pd.Series(result, dtype=float)


# =====================================================================
# Macro Data (VIX, Benchmark, Risk-Free)
# =====================================================================

def fetch_macro_data(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict:
    """
    Fetch macro overlay data from yfinance.
    Returns dict with keys: vix, tnx (10yr yield), dxy, spy, rf_daily.
    """
    import yfinance as yf

    tickers_map = {
        "vix": "^VIX",
        "tnx": "^TNX",
        "dxy": "DX-Y.NYB",
        "spy": "SPY",
        "irx": "^IRX",
    }

    result = {}
    for key, yf_sym in tickers_map.items():
        try:
            df = yf.download(yf_sym, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            if df.empty:
                result[key] = pd.Series(dtype=float)
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                result[key] = df["Close"].squeeze()
        except Exception:
            result[key] = pd.Series(dtype=float)

    vix = result.get("vix", pd.Series(dtype=float))
    tnx = result.get("tnx", pd.Series(dtype=float))
    dxy = result.get("dxy", pd.Series(dtype=float))
    spy_close = result.get("spy", pd.Series(dtype=float))
    irx = result.get("irx", pd.Series(dtype=float))

    spy_returns = spy_close.pct_change().dropna() if len(spy_close) > 1 else pd.Series(dtype=float)

    rf_daily = (irx / 100 / 252) if len(irx) > 0 else pd.Series(0.0, index=vix.index)
    rf_daily = rf_daily.ffill().fillna(0.0)

    macro = {
        "vix": vix,
        "vix_5d_change": vix.pct_change(5) if len(vix) > 5 else pd.Series(dtype=float),
        "tnx": tnx,
        "tnx_change": tnx.diff() if len(tnx) > 1 else pd.Series(dtype=float),
        "dxy": dxy,
        "dxy_change": dxy.pct_change() if len(dxy) > 1 else pd.Series(dtype=float),
        "spy_returns": spy_returns,
        "spy_close": spy_close,
        "rf_daily": rf_daily,
    }
    return macro


# =====================================================================
# Data Quality
# =====================================================================

def run_data_quality_checks(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict:
    """
    Data quality audit per revised methodology:
    - Missing value check (>5% in prices → investigate)
    - Return outlier detection (|r| > 20% → flag)
    """
    report = {"missing_pct": {}, "return_outliers": [], "status": "OK"}

    for c in adj_close.columns:
        miss = adj_close[c].isna().mean() * 100
        if miss > 5:
            report["missing_pct"][c] = round(miss, 2)

    returns = adj_close.pct_change()
    for c in returns.columns:
        mask = returns[c].abs() > 0.20
        if mask.any():
            dates = returns.index[mask].tolist()
            report["return_outliers"].append({
                "ticker": c,
                "count": int(mask.sum()),
                "dates": [str(d.date()) for d in dates[:5]],
            })

    if report["missing_pct"] or report["return_outliers"]:
        report["status"] = "REVIEW"

    return report


if __name__ == "__main__":
    print("Loading Bloomberg price data...")
    adj_close, open_, high, low, volume, eq_names = load_bloomberg_price_data()
    print(f"Loaded {len(eq_names)} equities, {len(adj_close)} dates")

    print("\nGetting selected 30 tickers...")
    selected = get_selected_tickers()
    print(f"Selected: {selected[:5]}...")

    print("\nFetching macro data...")
    macro = fetch_macro_data()
    for k, v in macro.items():
        if isinstance(v, pd.Series) and len(v) > 0:
            print(f"  {k}: {len(v)} obs, last={v.iloc[-1]:.4f}")

    print("\nData quality checks...")
    report = run_data_quality_checks(adj_close, volume)
    print(f"  Status: {report['status']}")
    print(f"  Outlier tickers: {len(report['return_outliers'])}")
