"""
Portfolio Optimization Research: Maximally Uncorrelated Stock Selection
Uses K-Medoids clustering on correlation-distance matrix for diversified portfolio.
Loads historical price data from Bloomberg terminal CSV export (no yfinance).
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURATION
# ==========================================
BLOOMBERG_CSV_PATH = r"C:\Users\manoj\Downloads\SPX as of Oct 08 20251 (1).csv"
start_date = '2020-01-01'   # In-sample training start
end_date = '2023-12-31'     # In-sample training end
n_clusters = 30
USE_KMEDOIDS = True

# ==========================================
# STEP 1: Load Bloomberg CSV Data
# ==========================================
if not os.path.isfile(BLOOMBERG_CSV_PATH):
    raise FileNotFoundError(f"Bloomberg CSV not found: {BLOOMBERG_CSV_PATH}")

print(f"Loading Bloomberg data from {BLOOMBERG_CSV_PATH}...")

# Read first 10 rows to parse header structure
raw_header = pd.read_csv(BLOOMBERG_CSV_PATH, nrows=10, header=None)

# Find the row containing "Dates" (metric header) and the row above it (equity names)
# Bloomberg format: equity names row, then Dates,PX_LAST,PX_VOLUME,PX_OPEN,PX_HIGH,PX_LOW per equity
metric_row_idx = None
for r in range(len(raw_header)):
    first_cell = str(raw_header.iloc[r, 0]).strip()
    if first_cell == 'Dates':
        metric_row_idx = r
        break
if metric_row_idx is None:
    raise ValueError("Could not find 'Dates' header row in Bloomberg CSV")
equity_row_idx = max(0, metric_row_idx - 2)  # Equity names typically 2 rows above
equity_row = raw_header.iloc[equity_row_idx]
metric_row = raw_header.iloc[metric_row_idx]

# Each equity has 5 columns: PX_LAST, PX_VOLUME, PX_OPEN, PX_HIGH, PX_LOW
# PX_LAST (closing price) is at position 1, 6, 11, 16, ... for each equity
px_last_cols = []
for i in range(1, len(metric_row), 5):
    if i >= len(equity_row):
        break
    metric = str(metric_row.iloc[i]).strip()
    if metric == 'PX_LAST':
        eq_name = str(equity_row.iloc[i]).strip()
        if eq_name and 'nan' not in eq_name.lower() and ('Equity' in eq_name or 'UW' in eq_name):
            if eq_name.upper() == 'XYZ UN EQUITY':  # Bloomberg phantom ticker - exclude
                continue
            px_last_cols.append((i, eq_name))

if len(px_last_cols) == 0:
    raise ValueError("No PX_LAST columns found. Check Bloomberg CSV format (expect Dates, PX_LAST, PX_VOLUME, ... per equity).")
print(f"Found {len(px_last_cols)} equities in Bloomberg CSV.")

# Read full data (skip header rows up to and including metric row)
df_raw = pd.read_csv(BLOOMBERG_CSV_PATH, skiprows=metric_row_idx + 1, header=None, low_memory=False)

# Parse date column (column 0)
df_raw = df_raw.copy()
df_raw[0] = pd.to_datetime(df_raw[0], format='%m/%d/%Y', errors='coerce')
df_raw = df_raw.dropna(subset=[0])
df_raw = df_raw.set_index(0)
df_raw.index.name = 'Date'

# Build price DataFrame: index=dates, columns=equity names
data = pd.DataFrame(index=df_raw.index)
for col_idx, eq_name in px_last_cols:
    if col_idx < df_raw.shape[1]:
        series = pd.to_numeric(df_raw[col_idx], errors='coerce')
        data[eq_name] = series

# Filter by date range
data = data.loc[(data.index >= start_date) & (data.index <= end_date)]

# ---> Institutional-grade data cleaning (order matters: drop BEFORE fill to avoid IPO trap)
# 1. Drop stocks with insufficient coverage FIRST (before any fill)
#    Use 99% thresh to exclude post-window IPOs (e.g. DASH Dec 2020, COIN Apr 2021)
#    Backfilling before drop would create flatlines and falsely select them as "uncorrelated"
data = data.dropna(axis=1, thresh=int(0.99 * len(data)))
# 2. Forward fill tiny 1-day gaps, then backward fill (only for stocks that passed)
data = data.ffill().bfill()
# 3. Drop any still-incomplete columns, fill remaining edge-case NaNs
data = data.dropna(axis=1)
data = data.ffill().bfill()
data = data.dropna(axis=1)

print(f"Remaining stocks after filter: {data.shape[1]}")
print(f"Date range: {data.index.min()} to {data.index.max()}")

if data.shape[1] < n_clusters:
    raise ValueError(f"Only {data.shape[1]} stocks remain after filtering, but n_clusters={n_clusters}. Reduce n_clusters or relax the 99%% completeness threshold.")

# ---> Sanity check for research: verify sample values match Bloomberg
# Expected from CSV: LYB=94.48, AXP=124.49, VZ=61.4 on 2020-01-01
_check_date = pd.Timestamp('2020-01-01')
if _check_date in data.index:
    for _eq, _exp in [('LYB UN Equity', 94.48), ('AXP UN Equity', 124.49), ('VZ UN Equity', 61.4)]:
        if _eq in data.columns:
            _got = data.loc[_check_date, _eq]
            _ok = abs(float(_got) - _exp) < 0.02
            print(f"  [Check] {_eq} on 2020-01-01: {_got:.2f} (expected ~{_exp}) {'OK' if _ok else 'VERIFY!'}")

# ==========================================
# STEP 2: Calculate Distance Matrix (Correlation-Based)
# ==========================================
print("Calculating daily returns and correlation matrix...")
returns = data.pct_change().dropna()
correlation_matrix = returns.corr()
n_before = len(correlation_matrix)

# Drop zero-variance stocks (NaN correlations from halted/stale data) before distance calc
correlation_matrix = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
correlation_matrix = correlation_matrix.fillna(0)  # Neutral correlation for any remaining edge cases
if len(correlation_matrix) < n_before:
    print(f"  Dropped {n_before - len(correlation_matrix)} zero-variance stock(s) from correlation matrix.")
if len(correlation_matrix) < n_clusters:
    raise ValueError(f"Only {len(correlation_matrix)} stocks have valid correlations after dropping zero-variance, but n_clusters={n_clusters}.")

# Correlation distance: sqrt(2*(1 - r)) for clustering
inner_value = np.clip(2 * (1 - correlation_matrix), 0, None)
distance_matrix = np.sqrt(inner_value)
distance_matrix = distance_matrix.fillna(0)
distance_values = distance_matrix.values.astype(np.float64)

# ==========================================
# STEP 3: Clustering (K-Medoids or K-Means)
# ==========================================
selected_tickers = None

if USE_KMEDOIDS:
    try:
        from sklearn_extra.cluster import KMedoids  # type: ignore[import-untyped]
        print(f"Running K-Medoids Clustering to select {n_clusters} diversified stocks...")
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42, init='k-medoids++')
        kmedoids.fit(distance_values)
        selected_tickers = [distance_matrix.columns[i] for i in kmedoids.medoid_indices_]
    except (ImportError, ValueError) as e:
        print(f"K-Medoids unavailable ({e}), falling back to K-Means...")
        selected_tickers = None

if selected_tickers is None:
    # K-Means cannot use precomputed distance matrix. Use MDS to embed distances
    # into Euclidean space, then cluster. This preserves the correlation structure.
    from sklearn.manifold import MDS
    n_components = min(50, len(distance_matrix) - 1, distance_values.shape[0] - 1)
    n_components = max(2, n_components)
    print(f"Running K-Means fallback (MDS embedding + K-Means) to select {n_clusters} diversified stocks...")
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(coords)

    selected_tickers = []
    for i in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[i]
        distances = [np.linalg.norm(coords[idx] - centroid) for idx in cluster_indices]
        closest_idx = cluster_indices[np.argmin(distances)]
        selected_tickers.append(distance_matrix.columns[closest_idx])

# Validation
assert len(selected_tickers) == n_clusters, f"Expected {n_clusters}, got {len(selected_tickers)}"
assert len(selected_tickers) == len(set(selected_tickers)), "Duplicate tickers found."

# ==========================================
# STEP 4: Output the Final List
# ==========================================
print(f"\n--- UNCORRELATED {n_clusters}-STOCK SELECTION FOR PORTFOLIO OPTIMIZATION ---")
print(selected_tickers)
