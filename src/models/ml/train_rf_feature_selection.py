from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RFSelectConfig:
    features_parquet: Path = Path("data/processed/features_h24_v1.parquet")
    model_out: Path = Path("models/ml/rf_fs_yh24.joblib")
    reports_dir: Path = Path("reports/tables")

    experiment_name: str = "pm25_h24"
    run_name: str = "rf_feature_selection_yh24"

    horizon: int = 24
    target_h: int = 24  # use y_h24 for feature selection

    random_state: int = 42

    # RF hyperparams (baseline reasonable)
    n_estimators: int = 800
    max_depth: int | None = None
    min_samples_leaf: int = 2
    max_features: str | float = "sqrt"  # "sqrt" works well as baseline

    n_jobs: int = -1  # <- multicore
    oob_score: bool = True  # useful diagnostic


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # Exclude identifiers and targets
    y_cols = [c for c in df.columns if c.startswith("y_h")]
    exclude = {"timestamp", "split"} | set(y_cols)
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if c == "station_id"]
    num_cols = [c for c in feature_cols if c != "station_id"]

    preproc = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preproc


def extract_feature_names(
    preproc: ColumnTransformer, feature_cols: List[str]
) -> List[str]:
    """
    Return names after preprocessing (including one-hot expanded station_id).
    Works for sklearn >=1.0.
    """
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        # fallback: crude names
        return feature_cols


def main(cfg: RFSelectConfig) -> None:
    cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading features: %s", cfg.features_parquet)
    df = pd.read_parquet(cfg.features_parquet)
    df = df[df["split"].isin(["train", "test", "val"])].copy()

    # Choose target
    y_col = f"y_h{cfg.target_h:02d}"
    assert y_col in df.columns, f"Missing target column: {y_col}"

    feature_cols = get_feature_columns(df)

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    X_train_raw = train[feature_cols]
    y_train = train[y_col].to_numpy()

    X_test_raw = test[feature_cols]
    y_test = test[y_col].to_numpy()

    X_val_raw = val[feature_cols]
    y_val = val[y_col].to_numpy()

    preproc = build_preprocessor(feature_cols)

    rf = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        oob_score=cfg.oob_score,
        bootstrap=True,
    )

    model = Pipeline(
        steps=[
            ("preproc", preproc),
            ("rf", rf),
        ]
    )

    # MLflow tracking
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "model_family": "random_forest",
                "task": "feature_selection",
                "target": y_col,
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "min_samples_leaf": cfg.min_samples_leaf,
                "max_features": cfg.max_features,
                "oob_score": cfg.oob_score,
                "n_jobs": cfg.n_jobs,
                "random_state": cfg.random_state,
                "features_file": str(cfg.features_parquet),
            }
        )
        mlflow.set_tags({"run_type": "train", "selection_method": "rf_importance"})

        LOGGER.info("Fitting RF for %s on train (%d rows)", y_col, len(train))
        model.fit(X_train_raw, y_train)

        # Evaluate
        pred_test = model.predict(X_test_raw)
        pred_val = model.predict(X_val_raw)

        rmse_test = root_mean_squared_error(y_test, pred_test)
        r2_test = r2_score(y_test, pred_test)
        rmse_val = root_mean_squared_error(y_val, pred_val)
        r2_val = r2_score(y_val, pred_val)

        mlflow.log_metrics(
            {
                "rmse_test": rmse_test,
                "r2_test": r2_test,
                "rmse_val": rmse_val,
                "r2_val": r2_val,
            }
        )

        # OOB (if enabled and available)
        try:
            oob = model.named_steps["rf"].oob_score_
            mlflow.log_metric("oob_r2_train", float(oob))
        except Exception:
            pass

        # Feature importances (post-preprocessing space)
        preproc_fitted = model.named_steps["preproc"]
        rf_fitted = model.named_steps["rf"]

        feat_names = extract_feature_names(preproc_fitted, feature_cols)
        importances = rf_fitted.feature_importances_

        imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        imp["importance_cum"] = imp["importance"].cumsum()

        # Save importance table
        out_imp = cfg.reports_dir / f"rf_importance_{y_col}.csv"
        imp.to_csv(out_imp, index=False)

        # A compact “top-k” artifact for quick inspection
        out_top = cfg.reports_dir / f"rf_importance_{y_col}_top50.csv"
        imp.head(50).to_csv(out_top, index=False)

        # Log artifacts
        mlflow.log_artifact(str(out_imp))
        mlflow.log_artifact(str(out_top))

        # Save model
        joblib.dump(model, cfg.model_out)
        mlflow.log_artifact(str(cfg.model_out))

        # Also log selected sets by cumulative threshold (e.g., 90%, 95%)
        for thr in [0.90, 0.95, 0.99]:
            k = int((imp["importance_cum"] <= thr).sum())
            mlflow.log_param(f"n_features_cum_{int(thr*100)}", k)

        LOGGER.info(
            "Test  RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            rmse_test,
            r2_test,
            rmse_val,
            r2_val,
        )
        LOGGER.info("Saved importance: %s", out_imp)
        LOGGER.info("Saved model: %s", cfg.model_out)


def parse_args() -> RFSelectConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_h24_v1.parquet")
    p.add_argument("--experiment", default="pm25_h24")
    p.add_argument("--run-name", default="rf_feature_selection_yh24")
    p.add_argument("--target-h", type=int, default=24)

    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--max-features", default="sqrt")
    p.add_argument("--oob-score", action="store_true")

    args = p.parse_args()

    # Parse max_depth
    max_depth = None if args.max_depth in (None, "None") else int(args.max_depth)

    # Parse max_features
    mf = args.max_features
    try:
        mf = float(mf)  # allow 0.5, etc.
    except Exception:
        pass

    return RFSelectConfig(
        features_parquet=Path(args.features),
        experiment_name=args.experiment,
        run_name=args.run_name,
        target_h=args.target_h,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=mf,
        oob_score=bool(args.oob_score),
    )


if __name__ == "__main__":
    configure_logging()
    cfg = parse_args()
    main(cfg)
