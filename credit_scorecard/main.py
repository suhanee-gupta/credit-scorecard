"""End-to-end credit scorecard pipeline."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from credit_scorecard.models import altman, logistic, xgb_model
from credit_scorecard.utils import evaluation, plotting, rating_map
from credit_scorecard.utils.data_loader import (
    PROCESSED_DIR,
    load_default_events,
    load_raw_data,
)
from credit_scorecard.utils.feature_engineering import FEATURES, build_dataset, latest_fiscal_year

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"

log = logging.getLogger("credit_scorecard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit risk scorecard for NSE-listed companies")
    parser.add_argument("--refresh", action="store_true", help="re-download data from Yahoo Finance")
    parser.add_argument("--population-default-rate", type=float, default=0.02,
                        help="annual default rate used to rescale model PDs before rating")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--xgb-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for d in (PROCESSED_DIR, PLOTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    fin, info = load_raw_data(refresh=args.refresh)
    events = load_default_events()
    df = build_dataset(fin, info, events)
    df = df.join(altman.score(df))
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)

    uncovered = sorted(set(events["ticker"]) - set(fin["ticker"]))
    log.info("Firm-years: %d | companies: %d | default firm-years: %d (%.1f%%)",
             len(df), df["ticker"].nunique(), df["default"].sum(), 100 * df["default"].mean())
    if uncovered:
        log.info("Labelled defaults without Yahoo Finance statements: %s", ", ".join(uncovered))

    train, test = evaluation.group_train_test_split(df, random_state=args.seed)
    cv = evaluation.grouped_cv(args.cv_folds, args.seed)
    log.info("Train: %d firm-years (%d defaults) | Test: %d firm-years (%d defaults)",
             len(train), train["default"].sum(), len(test), test["default"].sum())

    lr_search = logistic.tune(train, train["default"], train["ticker"], cv)
    xgb_search = xgb_model.tune(train, train["default"], train["ticker"], cv,
                                n_iter=args.xgb_iter, random_state=args.seed)
    lr, xgb = lr_search.best_estimator_, xgb_search.best_estimator_
    log.info("LR best C=%.4g (CV AUC %.3f)", lr_search.best_params_["clf__C"], lr_search.best_score_)
    log.info("XGB best params %s (CV AUC %.3f)", xgb_search.best_params_, xgb_search.best_score_)

    test_scores = {
        "Logistic regression": logistic.predict_pd(lr, test),
        "XGBoost": xgb_model.predict_pd(xgb, test),
    }
    z_risk = -test["z_score"].fillna(test["z_score"].median())

    metrics = {
        "sample": {
            "firm_years": int(len(df)),
            "companies": int(df["ticker"].nunique()),
            "default_firm_years": int(df["default"].sum()),
            "train_companies": int(train["ticker"].nunique()),
            "test_companies": int(test["ticker"].nunique()),
            "labelled_defaults_without_data": uncovered,
        },
        "test": {
            "logistic_regression": evaluation.summarise(test["default"], test_scores["Logistic regression"]),
            "xgboost": evaluation.summarise(test["default"], test_scores["XGBoost"]),
            "altman_z": evaluation.discrimination(test["default"], z_risk),
        },
        "cv": {
            "logistic_regression_auc": float(lr_search.best_score_),
            "xgboost_auc": float(xgb_search.best_score_),
        },
        "hyperparameters": {
            "logistic_regression": {k: float(v) for k, v in lr_search.best_params_.items()},
            "xgboost": xgb_search.best_params_,
        },
        "population_default_rate": args.population_default_rate,
    }

    plotting.roc_plot(
        test["default"],
        {**test_scores, "Altman Z (inverted)": z_risk.to_numpy()},
        PLOTS_DIR / "roc_curve.png",
        f"ROC on held-out companies (n={len(test)} firm-years)",
    )

    importance = xgb_model.feature_importance(xgb)
    plotting.feature_importance_plot(importance, PLOTS_DIR / "feature_importance.png")
    plotting.z_score_plot(df, PLOTS_DIR / "z_score_by_status.png")
    importance.to_csv(RESULTS_DIR / "xgb_feature_importance.csv", index=False)
    logistic.coefficients(lr).to_csv(RESULTS_DIR / "lr_coefficients.csv", index=False)

    df["lr_pd_raw"] = evaluation.out_of_fold_pd(lr, df, args.cv_folds, args.seed)
    df["xgb_pd_raw"] = evaluation.out_of_fold_pd(xgb, df, args.cv_folds, args.seed)
    metrics["out_of_fold"] = {
        "logistic_regression": evaluation.summarise(df["default"], df["lr_pd_raw"]),
        "xgboost": evaluation.summarise(df["default"], df["xgb_pd_raw"]),
    }

    sample_rate = df["default"].mean()
    for model in ("lr", "xgb"):
        df[f"{model}_pd"] = rating_map.adjust_to_population(
            df[f"{model}_pd_raw"], sample_rate, args.population_default_rate)
        df[f"{model}_rating"] = rating_map.pd_to_rating(df[f"{model}_pd"])

    metrics["altman_zone_default_rates"] = evaluation.zone_default_rates(df).to_dict("records")
    metrics["rating_default_rates"] = {
        m: evaluation.rating_default_rates(df, f"{m}_rating", rating_map.GRADES).to_dict("records")
        for m in ("lr", "xgb")
    }

    firm_year_cols = ["ticker", "name", "sector", "fiscal_year", "default", *FEATURES,
                      "z_score", "z_model", "z_zone",
                      "lr_pd_raw", "lr_pd", "lr_rating", "xgb_pd_raw", "xgb_pd", "xgb_rating"]
    df[firm_year_cols].to_csv(RESULTS_DIR / "firm_year_scores.csv", index=False)

    scorecard = latest_fiscal_year(df)
    scorecard["assigned_rating"] = rating_map.pd_to_rating(
        (scorecard["lr_pd"] + scorecard["xgb_pd"]) / 2)
    scorecard = scorecard[[
        "ticker", "name", "sector", "fiscal_year", "default",
        "z_score", "z_model", "z_zone",
        "lr_pd", "lr_rating", "xgb_pd", "xgb_rating", "assigned_rating",
    ]].sort_values("xgb_pd", ascending=False, ignore_index=True)
    scorecard.to_csv(RESULTS_DIR / "scorecard.csv", index=False, float_format="%.6g")
    rating_map.rating_table().to_csv(RESULTS_DIR / "rating_scale.csv", index=False)

    counts = pd.DataFrame({
        "Logistic regression": scorecard["lr_rating"].value_counts(),
        "XGBoost": scorecard["xgb_rating"].value_counts(),
    }).reindex(rating_map.GRADES, fill_value=0)
    plotting.rating_distribution_plot(counts, PLOTS_DIR / "rating_distribution.png")

    with open(RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    for name, m in metrics["test"].items():
        log.info("Test %-20s AUC %.3f  Gini %.3f", name, m["auc"], m["gini"])
    log.info("Scorecard written for %d companies", len(scorecard))


if __name__ == "__main__":
    main()
