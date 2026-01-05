from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import mlflow
from sklearn.metrics import root_mean_squared_error, r2_score


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistenceConfig:
    features_parquet: Path = Path("data/processed/features_h24_v1.parquet")
    reports_dir: Path = Path("reports/tables")

    experiment_name: str = "pm25_h24"
    run_name: str = "baseline_persistence_h24"

    horizon: int = 24


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    horizon = y_true.shape[1]
    rows = []

    for i in range(horizon):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = np.isfinite(yt) & np.isfinite(yp)

        if mask.sum() == 0:
            rmse = np.nan
            r2 = np.nan
        else:
            rmse = root_mean_squared_error(yt[mask], yp[mask])
            r2 = r2_score(yt[mask], yp[mask]) if mask.sum() >= 2 else np.nan

        rows.append(
            {
                "horizon_h": i + 1,
                "n_eval": int(mask.sum()),
                "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                "r2": float(r2) if np.isfinite(r2) else np.nan,
            }
        )

    m = pd.DataFrame(rows)

    # Resumen (ignorando NaNs)
    s = {
        "rmse_mean": float(m["rmse"].mean(skipna=True)),
        "r2_mean": float(m["r2"].mean(skipna=True)),
        "rmse_h24": float(m.loc[m["horizon_h"] == horizon, "rmse"].iloc[0]),
        "r2_h24": float(m.loc[m["horizon_h"] == horizon, "r2"].iloc[0]),
        "n_eval_mean": float(m["n_eval"].mean()),
        "n_eval_h24": int(m.loc[m["horizon_h"] == horizon, "n_eval"].iloc[0]),
    }
    return m, s


def run(cfg: PersistenceConfig) -> None:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading features: %s", cfg.features_parquet)
    df = pd.read_parquet(cfg.features_parquet)
    df = df[df["split"].isin(["train", "test", "val"])].copy()

    y_cols = [f"y_h{h:02d}" for h in range(1, cfg.horizon + 1)]
    # Baseline uses current pm25 as prediction for every horizon
    # Keep rows where pm25 exists and y_h24 exists (should already hold)
    df = df[~df["pm25"].isna() & ~df[f"y_h{cfg.horizon:02d}"].isna()].copy()

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "model_family": "baseline",
                "baseline_type": "persistence",
                "horizon": cfg.horizon,
                "features_file": str(cfg.features_parquet),
            }
        )
        mlflow.set_tags({"run_type": "baseline", "task": "pm25_multi_horizon"})

        outputs = []
        summaries = []

        for split in ["test", "val"]:
            part = df[df["split"] == split]
            y_true = part[y_cols].to_numpy()
            y_pred = np.repeat(
                part["pm25"].to_numpy().reshape(-1, 1), cfg.horizon, axis=1
            )

            m, s = compute_metrics(y_true, y_pred)
            m["split"] = split
            outputs.append(m)
            summaries.append({"split": split, **s})

            # Log summary metrics
            mlflow.log_metrics(
                {
                    f"rmse_{split}_mean": s["rmse_mean"],
                    f"r2_{split}_mean": s["r2_mean"],
                    f"rmse_{split}_h24": s["rmse_h24"],
                    f"r2_{split}_h24": s["r2_h24"],
                }
            )

            # Log per-horizon metrics as stepped series (optional but useful)
            for _, row in m.iterrows():
                h = int(row["horizon_h"])
                mlflow.log_metric(f"rmse_{split}", float(row["rmse"]), step=h)
                mlflow.log_metric(f"r2_{split}", float(row["r2"]), step=h)

        metrics = pd.concat(outputs, ignore_index=True)
        out_by_h = cfg.reports_dir / "persistence_h24_metrics_by_horizon.csv"
        metrics.to_csv(out_by_h, index=False)

        summary = pd.DataFrame(summaries)
        out_sum = cfg.reports_dir / "persistence_h24_metrics_summary.csv"
        summary.to_csv(out_sum, index=False)

        mlflow.log_artifact(str(out_by_h))
        mlflow.log_artifact(str(out_sum))

        LOGGER.info("Saved: %s", out_by_h)
        LOGGER.info("Saved: %s", out_sum)
        LOGGER.info("Summary:\n%s", summary.to_string(index=False))


def parse_args() -> PersistenceConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_h24_v1.parquet")
    p.add_argument("--experiment", default="pm25_h24")
    p.add_argument("--run-name", default="baseline_persistence_h24")
    p.add_argument("--reports-dir", default="reports/tables")
    args = p.parse_args()

    return PersistenceConfig(
        features_parquet=Path(args.features),
        experiment_name=args.experiment,
        run_name=args.run_name,
        reports_dir=Path(args.reports_dir),
    )


if __name__ == "__main__":
    configure_logging()
    cfg = parse_args()
    run(cfg)
