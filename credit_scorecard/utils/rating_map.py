"""Map probabilities of default to letter rating grades."""

import numpy as np
import pandas as pd

# Upper PD bound (exclusive) for each grade; anything at or above the last bound is D.
RATING_BANDS = [
    ("AAA", 0.0005),
    ("AA", 0.0015),
    ("A", 0.0040),
    ("BBB", 0.0120),
    ("BB", 0.0400),
    ("B", 0.1200),
    ("CCC", 0.4000),
]
DEFAULT_GRADE = "D"
GRADES = [g for g, _ in RATING_BANDS] + [DEFAULT_GRADE]


def adjust_to_population(pd_sample: np.ndarray, sample_rate: float,
                         population_rate: float) -> np.ndarray:
    """Rescale PDs estimated on an oversampled-default dataset to a target base rate.

    Applies the prior-correction odds adjustment
    odds_pop = odds_sample * [pi / (1 - pi)] / [rho / (1 - rho)],
    where rho is the sample default rate and pi the population default rate.
    """
    p = np.clip(np.asarray(pd_sample, dtype=float), 1e-9, 1 - 1e-9)
    factor = (population_rate / (1 - population_rate)) / (sample_rate / (1 - sample_rate))
    odds = p / (1 - p) * factor
    return odds / (1 + odds)


def pd_to_rating(pd_values) -> pd.Series:
    """Assign a rating grade to each PD using ``RATING_BANDS``."""
    p = np.asarray(pd_values, dtype=float)
    bounds = np.array([b for _, b in RATING_BANDS])
    idx = np.searchsorted(bounds, p, side="right")
    grades = np.array(GRADES, dtype=object)[idx]
    grades[np.isnan(p)] = None
    return pd.Series(grades, index=getattr(pd_values, "index", None), name="rating")


def rating_table() -> pd.DataFrame:
    lower = [0.0] + [b for _, b in RATING_BANDS]
    upper = [b for _, b in RATING_BANDS] + [1.0]
    return pd.DataFrame({"rating": GRADES, "pd_lower": lower, "pd_upper": upper})
