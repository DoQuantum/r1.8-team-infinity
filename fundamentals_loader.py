"""
Stage 1.2: Fundamental Data Collection
Simplified per revised methodology — keep only variables with clear alpha contribution.

Kept: ROE, ROA, Gross Margin, EBIT/EV proxy, Revenue Growth YoY,
      D/E ratio, P/E trailing, P/B
Dropped: Forward P/E, P/S, Current Ratio, Interest Coverage, EPS Revision

LIMITATION: yfinance uses fiscal period end dates, not actual release dates.
This creates look-ahead bias. Document in research paper.
"""
import numpy as np
import pandas as pd
from typing import Optional
from utils_ticker_map import bloomberg_to_yfinance


def fetch_fundamentals(tickers: list) -> dict:
    """
    Fetch quarterly fundamental data from yfinance.
    Returns dict[ticker] -> dict of metric DataFrames/scalars.
    """
    import yfinance as yf

    result = {}
    for t in tickers:
        yf_ticker = bloomberg_to_yfinance(t) if " " in str(t) else t
        if not yf_ticker:
            continue
        try:
            stock = yf.Ticker(yf_ticker)
            info = stock.info or {}
            inc = stock.quarterly_income_stmt
            bal = stock.quarterly_balance_sheet

            data = {}

            data["roe"] = info.get("returnOnEquity", np.nan)
            data["roa"] = info.get("returnOnAssets", np.nan)
            data["gross_margin"] = info.get("grossMargins", np.nan)
            data["de_ratio"] = info.get("debtToEquity", np.nan)
            data["trailing_pe"] = info.get("trailingPE", np.nan)
            data["price_to_book"] = info.get("priceToBook", np.nan)
            data["revenue_growth"] = info.get("revenueGrowth", np.nan)
            data["ev_to_ebitda"] = info.get("enterpriseToEbitda", np.nan)

            if data["ev_to_ebitda"] is not None and data["ev_to_ebitda"] > 0:
                data["ebit_ev"] = 1.0 / data["ev_to_ebitda"]
            else:
                data["ebit_ev"] = np.nan

            quarterly_roe = {}
            if inc is not None and bal is not None and not inc.empty and not bal.empty:
                for col in inc.columns[:16]:
                    try:
                        net_income = None
                        for ni_key in ["Net Income", "Net Income Common Stockholders"]:
                            if ni_key in inc.index:
                                net_income = inc.loc[ni_key, col]
                                break
                        equity = None
                        for eq_key in ["Stockholders Equity",
                                       "Total Stockholder Equity",
                                       "Common Stock Equity"]:
                            if eq_key in bal.index and col in bal.columns:
                                equity = bal.loc[eq_key, col]
                                break
                        if net_income is not None and equity is not None and equity != 0:
                            quarterly_roe[col] = float(net_income) / float(equity)
                    except Exception:
                        pass
            data["quarterly_roe"] = quarterly_roe

            quarterly_revenue = {}
            if inc is not None and not inc.empty:
                for col in inc.columns[:16]:
                    for rev_key in ["Total Revenue", "Revenue"]:
                        if rev_key in inc.index:
                            try:
                                quarterly_revenue[col] = float(inc.loc[rev_key, col])
                            except Exception:
                                pass
                            break
            data["quarterly_revenue"] = quarterly_revenue

            result[t] = data
        except Exception:
            result[t] = {}

    return result


def build_fundamental_features(
    fundamentals: dict,
    tickers: list,
    rebalance_dates: pd.DatetimeIndex,
) -> dict:
    """
    Build cross-sectional fundamental feature DataFrames.
    Returns dict of feature_name -> DataFrame(index=rebalance_dates, columns=tickers).

    Uses current snapshot values from yfinance .info, held constant.
    For quarterly time-series (ROE, revenue), computes deltas where possible.
    """
    features = {}

    roe_vals = {t: fundamentals.get(t, {}).get("roe", np.nan) for t in tickers}
    roa_vals = {t: fundamentals.get(t, {}).get("roa", np.nan) for t in tickers}
    gm_vals = {t: fundamentals.get(t, {}).get("gross_margin", np.nan) for t in tickers}
    de_vals = {t: fundamentals.get(t, {}).get("de_ratio", np.nan) for t in tickers}
    pe_vals = {t: fundamentals.get(t, {}).get("trailing_pe", np.nan) for t in tickers}
    pb_vals = {t: fundamentals.get(t, {}).get("price_to_book", np.nan) for t in tickers}
    rg_vals = {t: fundamentals.get(t, {}).get("revenue_growth", np.nan) for t in tickers}
    ebit_ev_vals = {t: fundamentals.get(t, {}).get("ebit_ev", np.nan) for t in tickers}

    def _to_df(vals_dict):
        row = pd.Series(vals_dict, dtype=float)
        return pd.DataFrame(
            np.tile(row.values, (len(rebalance_dates), 1)),
            index=rebalance_dates,
            columns=tickers,
        )

    features["roe"] = _to_df(roe_vals)
    features["roa"] = _to_df(roa_vals)
    features["gross_margin"] = _to_df(gm_vals)
    features["neg_de_ratio"] = _to_df({t: -v if not np.isnan(v) else np.nan
                                        for t, v in de_vals.items()})
    features["ebit_ev"] = _to_df(ebit_ev_vals)
    features["revenue_growth"] = _to_df(rg_vals)

    inv_pe = {}
    for t, v in pe_vals.items():
        if v is not None and not np.isnan(v) and v > 0:
            inv_pe[t] = 1.0 / v
        else:
            inv_pe[t] = np.nan
    features["inv_pe"] = _to_df(inv_pe)

    inv_pb = {}
    for t, v in pb_vals.items():
        if v is not None and not np.isnan(v) and v > 0:
            inv_pb[t] = 1.0 / v
        else:
            inv_pb[t] = np.nan
    features["inv_pb"] = _to_df(inv_pb)

    delta_roe = {}
    for t in tickers:
        qroe = fundamentals.get(t, {}).get("quarterly_roe", {})
        if len(qroe) >= 2:
            sorted_dates = sorted(qroe.keys(), reverse=True)
            delta_roe[t] = qroe[sorted_dates[0]] - qroe[sorted_dates[1]]
        else:
            delta_roe[t] = np.nan
    features["delta_roe"] = _to_df(delta_roe)

    return features


if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "JPM"]
    print("Fetching fundamentals...")
    fund = fetch_fundamentals(test_tickers)
    for t, data in fund.items():
        print(f"\n{t}:")
        for k, v in data.items():
            if k.startswith("quarterly"):
                print(f"  {k}: {len(v)} quarters")
            else:
                print(f"  {k}: {v}")
