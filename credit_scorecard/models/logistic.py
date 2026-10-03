"""Logistic regression probability-of-default model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_scorecard.utils.feature_engineering import FEATURES


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to quantiles learned on the training data."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("winsorize", Winsorizer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=5000, random_state=random_state)),
        ]
    )


def tune(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv,
         param_grid: dict | None = None) -> GridSearchCV:
    """Select the inverse regularisation strength by grouped cross-validated AUC."""
    param_grid = param_grid or {"clf__C": np.logspace(-3, 2, 11)}
    search = GridSearchCV(build_pipeline(), param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X[FEATURES], y, groups=groups)
    return search


def coefficients(model: Pipeline) -> pd.DataFrame:
    """Standardised coefficients and odds ratios per one-standard-deviation move."""
    clf = model.named_steps["clf"]
    coef = clf.coef_.ravel()
    return pd.DataFrame(
        {"feature": FEATURES, "coefficient": coef, "odds_ratio": np.exp(coef)}
    ).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)


def predict_pd(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURES])[:, 1]
