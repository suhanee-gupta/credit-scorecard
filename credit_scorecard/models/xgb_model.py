"""Gradient-boosted tree probability-of-default model."""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from credit_scorecard.utils.feature_engineering import FEATURES

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    "gamma": [0.0, 0.5, 1.0],
}


def build_model(random_state: int = 42, **params) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
        **params,
    )


def tune(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv,
         n_iter: int = 40, random_state: int = 42) -> RandomizedSearchCV:
    """Randomised hyperparameter search scored by grouped cross-validated AUC."""
    search = RandomizedSearchCV(
        build_model(random_state=random_state),
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X[FEATURES], y, groups=groups)
    return search


def feature_importance(model: XGBClassifier, importance_type: str = "gain") -> pd.DataFrame:
    scores = model.get_booster().get_score(importance_type=importance_type)
    imp = pd.DataFrame(
        {"feature": FEATURES, "importance": [scores.get(f, 0.0) for f in FEATURES]}
    )
    total = imp["importance"].sum()
    imp["share"] = imp["importance"] / total if total > 0 else 0.0
    return imp.sort_values("importance", ascending=False, ignore_index=True)


def predict_pd(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURES])[:, 1]
