"""Firm-year credit features and default labels."""

import pandas as pd

FEATURES = [
    "debt_to_assets",
    "liabilities_to_assets",
    "interest_coverage",
    "roa",
    "current_ratio",
    "asset_growth",
]

EXCLUDED_SECTORS = {"Financial Services"}

PRE_DEFAULT_HORIZON = pd.DateOffset(months=24)

INTEREST_COVERAGE_CAP = 50.0


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(den != 0)
    return num / den


def _interest_coverage(ebit: pd.Series, interest: pd.Series) -> pd.Series:
    interest = interest.abs()
    cov = _safe_div(ebit, interest)
    no_interest = interest.isna() | (interest == 0)
    cov = cov.mask(no_interest & (ebit > 0), INTEREST_COVERAGE_CAP)
    cov = cov.mask(no_interest & (ebit <= 0), -INTEREST_COVERAGE_CAP)
    return cov.clip(-INTEREST_COVERAGE_CAP, INTEREST_COVERAGE_CAP)


def compute_features(fin: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio features from raw annual statements, one row per ticker and fiscal year."""
    df = fin.sort_values(["ticker", "fiscal_year_end"]).copy()

    df["ebit"] = df["ebit"].fillna(df["pretax_income"] + df["interest_expense"].abs())
    df["working_capital"] = df["working_capital"].fillna(
        df["current_assets"] - df["current_liabilities"]
    )

    prev_assets = df.groupby("ticker")["total_assets"].shift(1)
    avg_assets = ((df["total_assets"] + prev_assets) / 2).fillna(df["total_assets"])

    df["debt_to_assets"] = _safe_div(df["total_debt"].fillna(0), df["total_assets"]).clip(0, 5)
    df["liabilities_to_assets"] = _safe_div(df["total_liabilities"], df["total_assets"]).clip(0, 10)
    df["interest_coverage"] = _interest_coverage(df["ebit"], df["interest_expense"])
    df["roa"] = _safe_div(df["net_income"], avg_assets).clip(-2, 1)
    df["current_ratio"] = _safe_div(df["current_assets"], df["current_liabilities"]).clip(0, 10)
    df["asset_growth"] = (_safe_div(df["total_assets"], prev_assets) - 1).clip(-1, 3)

    return df


def assign_default_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Flag a firm-year as defaulted if its fiscal year end lies within the
    pre-default horizon of an event or before that event's resolution."""
    label = pd.Series(0, index=df.index, dtype=int)
    for ev in events.itertuples():
        start = ev.event_date - PRE_DEFAULT_HORIZON
        end = ev.resolution_date if pd.notna(ev.resolution_date) else pd.Timestamp.max
        mask = (
            (df["ticker"] == ev.ticker)
            & (df["fiscal_year_end"] >= start)
            & (df["fiscal_year_end"] <= end)
        )
        label[mask] = 1
    return label


def build_dataset(fin: pd.DataFrame, info: pd.DataFrame, events: pd.DataFrame,
                  min_features: int = 4) -> pd.DataFrame:
    """Assemble the modelling table: features, labels and company metadata."""
    df = compute_features(fin)
    df = df.merge(info[["ticker", "name", "sector", "industry"]], on="ticker", how="left")
    df = df[~df["sector"].isin(EXCLUDED_SECTORS)]
    df = df[df[FEATURES].notna().sum(axis=1) >= min_features].copy()
    df["default"] = assign_default_labels(df, events)
    df["fiscal_year"] = df["fiscal_year_end"].dt.year
    return df.reset_index(drop=True)


def latest_fiscal_year(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("ticker")["fiscal_year_end"].idxmax()
    return df.loc[idx].reset_index(drop=True)

