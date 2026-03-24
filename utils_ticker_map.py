"""
Map Bloomberg equity names to yfinance tickers.
Bloomberg format: 'LYB UN Equity', 'AXP UN Equity', etc.
"""
import re

def bloomberg_to_yfinance(bloomberg_name: str) -> str:
    """Extract yfinance ticker from Bloomberg equity name."""
    if not bloomberg_name or "nan" in str(bloomberg_name).lower():
        return ""
    # "LYB UN Equity" -> "LYB", "AAPL UW Equity" -> "AAPL"
    parts = str(bloomberg_name).strip().split()
    if parts:
        return parts[0].upper()
    return ""

def get_ticker_mapping(bloomberg_names: list[str]) -> dict[str, str]:
    """Map Bloomberg names to yfinance tickers."""
    return {b: bloomberg_to_yfinance(b) for b in bloomberg_names}
