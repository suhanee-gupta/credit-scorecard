"""Download and cache annual financial statements for NSE tickers from Yahoo Finance."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FINANCIALS_PATH = RAW_DIR / "financials.csv"
COMPANY_INFO_PATH = RAW_DIR / "company_info.csv"
COVERAGE_PATH = RAW_DIR / "coverage.csv"

BALANCE_SHEET_FIELDS = {
    "Total Assets": "total_assets",
    "Current Assets": "current_assets",
    "Current Liabilities": "current_liabilities",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Total Debt": "total_debt",
    "Stockholders Equity": "book_equity",
    "Retained Earnings": "retained_earnings",
    "Capital Stock": "share_capital",
    "Working Capital": "working_capital",
    "Ordinary Shares Number": "shares_outstanding",
}

INCOME_STATEMENT_FIELDS = {
    "Total Revenue": "revenue",
    "EBIT": "ebit",
    "EBITDA": "ebitda",
    "Interest Expense": "interest_expense",
    "Pretax Income": "pretax_income",
    "Net Income": "net_income",
}

CASH_FLOW_FIELDS = {
    "Operating Cash Flow": "operating_cash_flow",
}

log = logging.getLogger(__name__)


def load_universe() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "universe.csv")


def load_default_events() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / "default_events.csv",
        parse_dates=["event_date", "resolution_date"],
    )


def _extract(statement: pd.DataFrame, fields: dict) -> pd.DataFrame:
    if statement is None or statement.empty:
        return pd.DataFrame()
    present = [f for f in fields if f in statement.index]
    out = statement.loc[present].T.rename(columns=fields)
    out.index = pd.to_datetime(out.index).normalize()
    out.index.name = "fiscal_year_end"
    return out.reindex(columns=list(fields.values()))


def _fiscal_year_end_prices(ticker: yf.Ticker, dates: pd.DatetimeIndex) -> pd.Series:
    if len(dates) == 0:
        return pd.Series(dtype=float)
    start = dates.min() - pd.Timedelta(days=30)
    end = dates.max() + pd.Timedelta(days=5)
    hist = ticker.history(start=start, end=end, auto_adjust=False)
    if hist.empty:
        return pd.Series(np.nan, index=dates)
    close = hist["Close"]
    close.index = close.index.tz_localize(None).normalize()
    return close.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=10))


def _fx_to_inr(currency: str, dates: pd.DatetimeIndex) -> pd.Series:
    """INR per unit of ``currency`` on each date (1.0 for INR)."""
    if not currency or currency == "INR":
        return pd.Series(1.0, index=dates)
    fx = yf.Ticker(f"{currency}INR=X").history(
        start=dates.min() - pd.Timedelta(days=30),
        end=dates.max() + pd.Timedelta(days=5),
    )
    if fx.empty:
        return pd.Series(np.nan, index=dates)
    rate = fx["Close"]
    rate.index = rate.index.tz_localize(None).normalize()
    return rate.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=10))


def fetch_ticker(symbol: str, retries: int = 3, pause: float = 2.0) -> tuple[pd.DataFrame, dict]:
    """Return (annual financials, company metadata) for a single NSE symbol."""
    last_error = None
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            frames = [
                _extract(ticker.balance_sheet, BALANCE_SHEET_FIELDS),
                _extract(ticker.income_stmt, INCOME_STATEMENT_FIELDS),
                _extract(ticker.cashflow, CASH_FLOW_FIELDS),
            ]
            frames = [f for f in frames if not f.empty]
            if not frames:
                return pd.DataFrame(), {"ticker": symbol}

            fin = pd.concat(frames, axis=1).sort_index()
            fin = fin.dropna(subset=["total_assets"]) if "total_assets" in fin else pd.DataFrame()
            if fin.empty:
                return pd.DataFrame(), {"ticker": symbol}

            info = ticker.info or {}
            currency = info.get("financialCurrency") or "INR"
            price = _fiscal_year_end_prices(ticker, fin.index)
            fx = _fx_to_inr(currency, fin.index)
            fin["price_fy_end"] = price.values
            fin["market_cap"] = (fin["shares_outstanding"] * price / fx).values
            fin.insert(0, "ticker", symbol)
            fin.insert(1, "financial_currency", currency)

            meta = {
                "ticker": symbol,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
            return fin.reset_index(), meta
        except Exception as exc:  # network / parsing failures from Yahoo
            last_error = exc
            time.sleep(pause * (attempt + 1))
    log.warning("Failed to fetch %s: %s", symbol, last_error)
    return pd.DataFrame(), {"ticker": symbol}


def fetch_universe(tickers: list[str], pause: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    financials, metadata, coverage = [], [], []
    for symbol in tickers:
        fin, meta = fetch_ticker(symbol)
        metadata.append(meta)
        coverage.append({"ticker": symbol, "fiscal_years": len(fin)})
        if not fin.empty:
            financials.append(fin)
        log.info("%-12s %d fiscal years", symbol, len(fin))
        time.sleep(pause)

    fin_df = pd.concat(financials, ignore_index=True) if financials else pd.DataFrame()
    return fin_df, pd.DataFrame(metadata), pd.DataFrame(coverage)


def load_raw_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached raw financials and company metadata, downloading them if absent."""
    if refresh or not FINANCIALS_PATH.exists() or not COMPANY_INFO_PATH.exists():
        tickers = load_universe()["ticker"].tolist()
        fin, info, coverage = fetch_universe(tickers)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        fin.to_csv(FINANCIALS_PATH, index=False)
        info.to_csv(COMPANY_INFO_PATH, index=False)
        coverage.to_csv(COVERAGE_PATH, index=False)

    fin = pd.read_csv(FINANCIALS_PATH, parse_dates=["fiscal_year_end"])
    info = pd.read_csv(COMPANY_INFO_PATH)
    return fin, info
