"""Altman Z-score: original 1968 model for manufacturers and Z'' (1995) for non-manufacturers."""

import numpy as np
import pandas as pd

Z_1968_WEIGHTS = {"x1": 1.2, "x2": 1.4, "x3": 3.3, "x4": 0.6, "x5": 1.0}
Z_1968_ZONES = (1.81, 2.99)

Z_DOUBLE_PRIME_WEIGHTS = {"x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05}
Z_DOUBLE_PRIME_ZONES = (1.10, 2.60)

NON_MANUFACTURING_SECTORS = {
    "Communication Services",
    "Technology",
    "Utilities",
    "Real Estate",
    "Financial Services",
}

NON_MANUFACTURING_INDUSTRIES = {
    "Engineering & Construction",
    "Airlines",
    "Infrastructure Operations",
    "Marine Shipping",
    "Railroads",
    "Restaurants",
    "Travel Services",
    "Apparel Retail",
    "Discount Stores",
    "Education & Training Services",
    "Medical Care Facilities",
    "Entertainment",
}


def is_manufacturing(sector: pd.Series, industry: pd.Series) -> pd.Series:
    return ~(sector.isin(NON_MANUFACTURING_SECTORS) | industry.isin(NON_MANUFACTURING_INDUSTRIES))


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


def z_1968(df: pd.DataFrame) -> pd.Series:
    """Z = 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MVE/TL + 1.0 Sales/TA.

    Book equity substitutes for market value where no fiscal-year-end price is available.
    """
    ta = df["total_assets"]
    equity_value = df["market_cap"].fillna(df["book_equity"])
    x = {
        "x1": _ratio(df["working_capital"], ta),
        "x2": _ratio(df["retained_earnings"], ta),
        "x3": _ratio(df["ebit"], ta),
        "x4": _ratio(equity_value, df["total_liabilities"]),
        "x5": _ratio(df["revenue"], ta),
    }
    return sum(Z_1968_WEIGHTS[k] * v for k, v in x.items())


def z_double_prime(df: pd.DataFrame) -> pd.Series:
    """Z'' = 6.56 WC/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BVE/TL."""
    ta = df["total_assets"]
    x = {
        "x1": _ratio(df["working_capital"], ta),
        "x2": _ratio(df["retained_earnings"], ta),
        "x3": _ratio(df["ebit"], ta),
        "x4": _ratio(df["book_equity"], df["total_liabilities"]),
    }
    return sum(Z_DOUBLE_PRIME_WEIGHTS[k] * v for k, v in x.items())


def _zone(z: pd.Series, distress_cutoff: float, safe_cutoff: float) -> pd.Series:
    zone = np.select(
        [z.isna(), z < distress_cutoff, z > safe_cutoff],
        ["N/A", "Distress", "Safe"],
        default="Grey",
    )
    return pd.Series(zone, index=z.index)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return Z-score, model variant and rating zone for each row of ``df``."""
    manufacturing = is_manufacturing(df["sector"], df["industry"])
    z68 = z_1968(df)
    zpp = z_double_prime(df)

    z = z68.where(manufacturing, zpp)
    zone = _zone(z68, *Z_1968_ZONES).where(manufacturing, _zone(zpp, *Z_DOUBLE_PRIME_ZONES))

    return pd.DataFrame(
        {
            "z_score": z,
            "z_model": np.where(manufacturing, "Z (1968)", "Z'' (1995)"),
            "z_zone": zone,
        },
        index=df.index,
    )
