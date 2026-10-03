"""Discrimination and calibration metrics plus company-grouped validation helpers."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from credit_scorecard.utils.feature_engineering import FEATURES


def gini(y_true, score) -> float:
    return 2 * roc_auc_score(y_true, score) - 1


def ks_statistic(y_true, score) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def discrimination(y_true, score) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "gini": float(gini(y_true, score)),
        "ks": ks_statistic(y_true, score),
    }


def summarise(y_true, pd_pred) -> dict:
    return {
        **discrimination(y_true, pd_pred),
        "brier": float(brier_score_loss(y_true, pd_pred)),
        "n": int(len(y_true)),
        "defaults": int(np.sum(y_true)),
    }


def grouped_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedGroupKFold:
    """Folds that keep every firm-year of a company together."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def group_train_test_split(df: pd.DataFrame, test_folds: int = 4,
                           random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out roughly 1/``test_folds`` of companies, stratified on default."""
    splitter = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["default"], groups=df["ticker"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def out_of_fold_pd(estimator, df: pd.DataFrame, n_splits: int = 5,
                   random_state: int = 42) -> np.ndarray:
    """Predict each firm-year with a model that never saw that company during fitting."""
    oof = np.full(len(df), np.nan)
    X, y, groups = df[FEATURES], df["default"], df["ticker"]
    for train_idx, test_idx in grouped_cv(n_splits, random_state).split(X, y, groups):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return oof


def zone_default_rates(df: pd.DataFrame, zone_col: str = "z_zone") -> pd.DataFrame:
    out = df.groupby(zone_col)["default"].agg(firm_years="size", defaults="sum")
    out["default_rate"] = out["defaults"] / out["firm_years"]
    return out.reset_index()


def rating_default_rates(df: pd.DataFrame, rating_col: str, grades: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(rating_col)["default"]
        .agg(firm_years="size", defaults="sum")
        .reindex(grades, fill_value=0)
    )
    out["default_rate"] = (out["defaults"] / out["firm_years"]).where(out["firm_years"] > 0)
    return out.rename_axis("rating").reset_index()
