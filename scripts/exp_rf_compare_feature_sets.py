from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cfg:
    experiment_name: str = "pm25_h24"
    run_name: str = "rf_compare_feature_sets_y_h24"

    # Datasets
    ds_all: Path = Path("data/processed/features_h24_v1.parquet")
    ds_perm90: Path = Path("data/processed/features_h24_perm90_yh24.parquet")
    ds_perm95: Path = Path("data/processed/features_h24_perm95_yh24.parquet")

    # Output
    reports_dir: Path = Path("reports/tables")

    # Target
    target_h: int = 24

    # RF hyperparams (idénticos para comparar)
    n_estimators: int = 400
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    max_features: str = "sqrt"  # "sqrt" suele ir bien
    n_jobs: int = -1
    random_state: int = 42

    # Sampling (opcional para acelerar en ALL)
    max_train_rows: int = 0  # 0 = no submuestrear


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def get_feature_columns(df: pd.DataFrame, y_col: str) -> List[str]:
    # Excluir identifiers y targets (incluyendo y_h**)
    y_cols = [c for c in df.columns if c.startswith("y_h")]
    exclude = {"timestamp", "split"} | set(y_cols)
    feature_cols = [c for c in df.columns if c not in exclude]

    # Seguridad: asegúrate que y_col no se quede por error
    feature_cols = [c for c in feature_cols if c != y_col]
    return feature_cols


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    # En tu pipeline actual, "station_id" es categórica
    cat_cols = [c for c in feature_cols if c == "station_id"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        raise ValueError(
            "No se encontraron columnas de features (numéricas o categóricas)."
        )

    return ColumnTransformer(transformers=transformers)


def load_and_split(ds_path: Path, y_col: str) -> Dict[str, pd.DataFrame]:
    df = pd.read_parquet(ds_path)
    df = df[df["split"].isin(["train", "test", "val"])].copy()

    # Limpieza mínima
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # El target debe existir
    if y_col not in df.columns:
        raise KeyError(f"Target {y_col} no existe en {ds_path}")

    # No podemos evaluar donde y es NaN
    df = df[df[y_col].notna()].copy()

    out = {
        "train": df[df["split"] == "train"].copy(),
        "test": df[df["split"] == "test"].copy(),
        "val": df[df["split"] == "val"].copy(),
    }
    return out


def maybe_subsample_train(
    train: pd.DataFrame, max_rows: int, random_state: int
) -> pd.DataFrame:
    if max_rows and len(train) > max_rows:
        return train.sample(n=max_rows, random_state=random_state)
    return train


def fit_eval_one(feature_set_name: str, ds_path: Path, cfg: Cfg) -> Dict[str, float]:
    y_col = f"y_h{cfg.target_h:02d}"
    splits = load_and_split(ds_path, y_col=y_col)

    train = maybe_subsample_train(splits["train"], cfg.max_train_rows, cfg.random_state)
    test = splits["test"]
    val = splits["val"]

    feature_cols = get_feature_columns(train, y_col=y_col)

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
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

    model = Pipeline(steps=[("preprocess", preproc), ("rf", rf)])

    LOGGER.info(
        "[%s] Fitting RF on train (%d rows, %d raw features)",
        feature_set_name,
        len(train),
        len(feature_cols),
    )
    t0 = time.time()
    model.fit(X_train_raw, y_train)
    fit_s = time.time() - t0

    pred_test = model.predict(X_test_raw)
    pred_val = model.predict(X_val_raw)

    rmse_test = root_mean_squared_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)
    rmse_val = root_mean_squared_error(y_val, pred_val)
    r2_val = r2_score(y_val, pred_val)

    # número de features después del preproc (útil para comparar)
    try:
        n_after = len(model.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        n_after = np.nan

    result = {
        "feature_set": feature_set_name,
        "dataset_path": str(ds_path),
        "target": y_col,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_val": int(len(val)),
        "n_features_raw": int(len(feature_cols)),
        "n_features_after_preproc": float(n_after),
        "fit_seconds": float(fit_s),
        "rmse_test": float(rmse_test),
        "r2_test": float(r2_test),
        "rmse_val": float(rmse_val),
        "r2_val": float(r2_val),
    }
    return result


def main(cfg: Cfg) -> None:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    y_col = f"y_h{cfg.target_h:02d}"

    feature_sets = {
        "ALL": cfg.ds_all,
        "PERM90": cfg.ds_perm90,
        "PERM95": cfg.ds_perm95,
    }

    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "model_family": "random_forest",
                "task": "compare_feature_sets",
                "target": y_col,
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth if cfg.max_depth is not None else "None",
                "min_samples_leaf": cfg.min_samples_leaf,
                "max_features": cfg.max_features,
                "n_jobs": cfg.n_jobs,
                "random_state": cfg.random_state,
                "max_train_rows": cfg.max_train_rows,
            }
        )
        mlflow.set_tags(
            {"run_type": "experiment", "experiment": "rf_feature_set_comparison"}
        )

        rows = []
        for fs_name, ds_path in feature_sets.items():
            with mlflow.start_run(run_name=f"RF_{fs_name}", nested=True):
                mlflow.log_param("feature_set", fs_name)
                mlflow.log_param("dataset_path", str(ds_path))

                res = fit_eval_one(fs_name, ds_path, cfg)
                rows.append(res)

                mlflow.log_metrics(
                    {
                        "rmse_test": res["rmse_test"],
                        "r2_test": res["r2_test"],
                        "rmse_val": res["rmse_val"],
                        "r2_val": res["r2_val"],
                        "fit_seconds": res["fit_seconds"],
                        "n_features_raw": res["n_features_raw"],
                        "n_features_after_preproc": res["n_features_after_preproc"],
                    }
                )

                LOGGER.info(
                    "[%s] Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f | raw=%d after=%.0f",
                    fs_name,
                    res["rmse_test"],
                    res["r2_test"],
                    res["rmse_val"],
                    res["r2_val"],
                    res["n_features_raw"],
                    res["n_features_after_preproc"],
                )

        out = pd.DataFrame(rows).sort_values("rmse_val")
        out_path = cfg.reports_dir / f"rf_compare_feature_sets_{y_col}.csv"
        out.to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path))

        # Log “ganador” por val RMSE
        winner = out.iloc[0].to_dict()
        mlflow.log_params(
            {
                "winner_feature_set": winner["feature_set"],
                "winner_rmse_val": winner["rmse_val"],
                "winner_r2_val": winner["r2_val"],
            }
        )

        LOGGER.info("Saved comparison table: %s", out_path)
        LOGGER.info(
            "Winner (by val RMSE): %s | rmse_val=%.4f r2_val=%.4f",
            winner["feature_set"],
            winner["rmse_val"],
            winner["r2_val"],
        )


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="pm25_h24")
    p.add_argument("--run-name", default="rf_compare_feature_sets_y_h24")

    p.add_argument("--ds-all", default="data/processed/features_h24_v1.parquet")
    p.add_argument(
        "--ds-perm90", default="data/processed/features_h24_perm90_yh24.parquet"
    )
    p.add_argument(
        "--ds-perm95", default="data/processed/features_h24_perm95_yh24.parquet"
    )

    p.add_argument("--reports-dir", default="reports/tables")
    p.add_argument("--target-h", type=int, default=24)

    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=0)  # 0 => None
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--max-features", default="sqrt")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-train-rows", type=int, default=0)

    a = p.parse_args()

    md = None if a.max_depth == 0 else int(a.max_depth)

    return Cfg(
        experiment_name=a.experiment,
        run_name=a.run_name,
        ds_all=Path(a.ds_all),
        ds_perm90=Path(a.ds_perm90),
        ds_perm95=Path(a.ds_perm95),
        reports_dir=Path(a.reports_dir),
        target_h=a.target_h,
        n_estimators=a.n_estimators,
        max_depth=md,
        min_samples_leaf=a.min_samples_leaf,
        max_features=a.max_features,
        n_jobs=a.n_jobs,
        random_state=a.random_state,
        max_train_rows=a.max_train_rows,
    )


if __name__ == "__main__":
    configure_logging()
    cfg = parse_args()
    main(cfg)
