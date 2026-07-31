"""Train and validate the freight rate model.

Validation strategy
--------------------
The labeled data (data/train_test.csv) spans 2025-01-01 .. 2025-10-31, and the
data we must finally score (data/validation.csv) spans 2025-11-01 ..
2025-12-31 -- i.e. entirely *after* the labeled window. That is a forecasting
problem, not an interpolation problem, so a random K-fold split would leak
future-like information backwards and overstate accuracy.

We therefore use a **time-based holdout**: sort the labeled data by date and
hold out the most recent ~15% of days (2025-09-14 .. 2025-10-31) as the
validation set, training only on data strictly before that window. This
mirrors the real deployment gap (train on the past, predict an unseen future
window) and is the number we trust for judging generalization.

As a secondary sanity check we also report 5-fold random cross-validation
metrics; if the two disagree sharply it's a signal of temporal drift, which
is useful to know but the time-based holdout is the primary number.

Once the holdout confirms the model behaves well, we refit on *all* of
data/train_test.csv (still with the same feature pipeline) to produce the
final model used for validation_predictions.csv and the December chart --
using every labeled example available is preferable once the validation
protocol above has already told us how the model performs on unseen future
data.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from features import FEATURE_COLUMNS, CATEGORICAL_FEATURE_NAMES, FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODEL_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=25,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
    verbosity=-1,
)
HOLDOUT_START = "2025-09-14"  # last ~15% of labeled days


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**MODEL_PARAMS)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def time_based_holdout(df: pd.DataFrame) -> dict:
    train_part = df[df["date"] < HOLDOUT_START].reset_index(drop=True)
    holdout_part = df[df["date"] >= HOLDOUT_START].reset_index(drop=True)

    builder = FeatureBuilder().fit(train_part)
    train_feat = builder.transform(train_part)
    holdout_feat = builder.transform(holdout_part)

    model = build_model()
    model.fit(
        train_feat[FEATURE_COLUMNS],
        np.log1p(train_feat["posted_rate"]),
        eval_set=[(holdout_feat[FEATURE_COLUMNS], np.log1p(holdout_feat["posted_rate"]))],
        eval_metric="mae",
        categorical_feature=CATEGORICAL_FEATURE_NAMES,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred_log = model.predict(holdout_feat[FEATURE_COLUMNS], num_iteration=model.best_iteration_)
    pred = np.expm1(pred_log)
    metrics = evaluate(holdout_feat["posted_rate"].values, pred)
    metrics["n_train"] = int(len(train_part))
    metrics["n_holdout"] = int(len(holdout_part))
    metrics["holdout_date_range"] = [holdout_part["date"].min(), holdout_part["date"].max()]
    metrics["best_iteration"] = int(model.best_iteration_)

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_.tolist()), key=lambda x: -x[1]
    )
    metrics["top_features"] = importances[:10]
    return metrics


def random_kfold_cv(df: pd.DataFrame, n_splits: int = 5) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_part = df.iloc[train_idx].reset_index(drop=True)
        test_part = df.iloc[test_idx].reset_index(drop=True)

        builder = FeatureBuilder().fit(train_part)
        train_feat = builder.transform(train_part)
        test_feat = builder.transform(test_part)

        model = build_model()
        model.set_params(n_estimators=600)  # fixed budget, no early stopping needed for CV sanity check
        model.fit(
            train_feat[FEATURE_COLUMNS],
            np.log1p(train_feat["posted_rate"]),
            categorical_feature=CATEGORICAL_FEATURE_NAMES,
        )
        pred = np.expm1(model.predict(test_feat[FEATURE_COLUMNS]))
        fold_metrics.append(evaluate(test_feat["posted_rate"].values, pred))

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    return {"per_fold": fold_metrics, "average": avg}


def train_final_model(df: pd.DataFrame, best_iteration: int) -> tuple[lgb.LGBMRegressor, FeatureBuilder]:
    builder = FeatureBuilder().fit(df)
    feat = builder.transform(df)
    model = build_model()
    model.set_params(n_estimators=max(best_iteration, 200))
    model.fit(
        feat[FEATURE_COLUMNS],
        np.log1p(feat["posted_rate"]),
        categorical_feature=CATEGORICAL_FEATURE_NAMES,
    )
    return model, builder


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "train_test.csv")

    print("=== Time-based holdout (primary validation) ===")
    holdout_metrics = time_based_holdout(df)
    print(json.dumps({k: v for k, v in holdout_metrics.items() if k != "top_features"}, indent=2))
    print("Top features:")
    for name, score in holdout_metrics["top_features"]:
        print(f"  {name}: {score}")

    print("\n=== 5-fold random CV (secondary sanity check) ===")
    cv_metrics = random_kfold_cv(df)
    print(json.dumps(cv_metrics["average"], indent=2))

    with open(REPORTS / "validation_metrics.json", "w") as f:
        json.dump({"time_based_holdout": holdout_metrics, "random_cv": cv_metrics}, f, indent=2)

    print("\n=== Refitting final model on all labeled data ===")
    final_model, final_builder = train_final_model(df, holdout_metrics["best_iteration"])

    import joblib

    joblib.dump({"model": final_model, "builder": final_builder}, REPORTS / "final_model.joblib")
    print(f"Saved final model to {REPORTS / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
