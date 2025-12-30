# src/features/make_features_h24.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturesH24Config:
    master_long_parquet: Path = Path("data/interim/master_long_v1.parquet")
    neighbors_json: Path = Path("data/interim/neighbors_by_station_k5.json")

    out_processed_parquet: Path = Path("data/processed/features_h24_v1.parquet")
    out_reports_tables_dir: Path = Path("reports/tables")

    horizon: int = 24
    neighbor_agg: str = "mean"  # mean | median
    k_neighbors: int = 5  # informational (from json)

    # Lags/rollings for pm25
    pm25_lags: List[int] = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    pm25_roll_windows: List[int] = (24, 72, 168)

    # Optional exogenous
    use_pm10: bool = True
    use_o3: bool = True
    exog_lags: List[int] = (1, 24)
    exog_roll_windows: List[int] = (24,)

    # Spatial features at these lags
    neighbor_lags: List[int] = (1, 24)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_group_shift(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level=0).shift(n)  # groupby station_id on multiindex


def _rolling_by_station(s: pd.Series, window: int, fn: str) -> pd.Series:
    g = s.groupby(level=0)
    if fn == "mean":
        return (
            g.rolling(window=window, min_periods=max(1, window // 3))
            .mean()
            .reset_index(level=0, drop=True)
        )
    if fn == "std":
        return (
            g.rolling(window=window, min_periods=max(1, window // 3))
            .std()
            .reset_index(level=0, drop=True)
        )
    raise ValueError(f"Unsupported rolling fn: {fn}")


def load_neighbors(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Keep only neighbor ids, preserve order
    nbr = {sid: [x["neighbor"] for x in lst] for sid, lst in raw.items()}
    return nbr


def build_features(cfg: FeaturesH24Config) -> None:
    cfg.out_processed_parquet.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_reports_tables_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading master long: %s", cfg.master_long_parquet)
    df = pd.read_parquet(cfg.master_long_parquet)

    # Basic filtering
    df = df[df["split"].isin(["train", "test", "val"])].copy()
    df["station_id"] = df["station_id"].astype(str)

    # Sort for time series ops
    df = df.sort_values(["station_id", "timestamp"])

    # Set multiindex for efficient grouped shifts/rollings
    df = df.set_index(["station_id", "timestamp"])

    # --- Targets y_h01..y_h24 (future values) ---
    for h in range(1, cfg.horizon + 1):
        df[f"y_h{h:02d}"] = df["pm25"].groupby(level=0).shift(-h)

    # --- Temporal lags for pm25 ---
    for lag in cfg.pm25_lags:
        df[f"pm25_lag{lag}"] = df["pm25"].groupby(level=0).shift(lag)

    # --- Rolling stats for pm25 (past-only: use shift(1) before rolling to avoid including current) ---
    pm25_past = df["pm25"].groupby(level=0).shift(1)
    for w in cfg.pm25_roll_windows:
        df[f"pm25_roll_mean{w}"] = (
            pm25_past.groupby(level=0)
            .rolling(window=w, min_periods=max(1, w // 3))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"pm25_roll_std{w}"] = (
            pm25_past.groupby(level=0)
            .rolling(window=w, min_periods=max(1, w // 3))
            .std()
            .reset_index(level=0, drop=True)
        )

    # --- Optional exogenous: pm10, o3 ---
    if cfg.use_pm10 and "pm10" in df.columns:
        df["has_pm10"] = (~df["pm10"].isna()).astype(int)
        for lag in cfg.exog_lags:
            df[f"pm10_lag{lag}"] = df["pm10"].groupby(level=0).shift(lag)
        pm10_past = df["pm10"].groupby(level=0).shift(1)
        for w in cfg.exog_roll_windows:
            df[f"pm10_roll_mean{w}"] = (
                pm10_past.groupby(level=0)
                .rolling(window=w, min_periods=max(1, w // 3))
                .mean()
                .reset_index(level=0, drop=True)
            )
    else:
        df["has_pm10"] = 0

    if cfg.use_o3 and "o3" in df.columns:
        df["has_o3"] = (~df["o3"].isna()).astype(int)
        for lag in cfg.exog_lags:
            df[f"o3_lag{lag}"] = df["o3"].groupby(level=0).shift(lag)
        o3_past = df["o3"].groupby(level=0).shift(1)
        for w in cfg.exog_roll_windows:
            df[f"o3_roll_mean{w}"] = (
                o3_past.groupby(level=0)
                .rolling(window=w, min_periods=max(1, w // 3))
                .mean()
                .reset_index(level=0, drop=True)
            )
    else:
        df["has_o3"] = 0

    # --- Neighbor (spatial) features using pm25 from neighbor stations ---
    # Build a lookup: for each timestamp, neighbor pm25 values.
    # Approach:
    # 1) pivot pm25 to wide: index timestamp, columns station_id
    # 2) for each station_id, aggregate neighbor values at lagged timestamps
    LOGGER.info("Building neighbor features from %s", cfg.neighbors_json)
    neighbors = load_neighbors(cfg.neighbors_json)

    # back to columns to pivot
    df_reset = df.reset_index()
    pm25_wide = df_reset.pivot(
        index="timestamp", columns="station_id", values="pm25"
    ).sort_index()

    def agg_neighbors_for_station(sid: str, lag: int, fn: str) -> pd.Series:
        nbrs = neighbors.get(sid, [])
        if not nbrs:
            return pd.Series(index=pm25_wide.index, dtype=float)
        X = pm25_wide[nbrs].shift(lag)
        if fn == "mean":
            return X.mean(axis=1)
        if fn == "median":
            return X.median(axis=1)
        raise ValueError(fn)

    agg_fn = cfg.neighbor_agg
    rows = []
    for sid in df_reset["station_id"].unique():
        tmp = pd.DataFrame({"timestamp": pm25_wide.index})
        tmp["station_id"] = sid
        for lag in cfg.neighbor_lags:
            col = f"nbr_pm25_{agg_fn}_lag{lag}"
            tmp[col] = agg_neighbors_for_station(sid, lag=lag, fn=agg_fn).to_numpy()
        rows.append(tmp)

    neighbor_df = pd.concat(rows, ignore_index=True)

    # Validación fuerte: neighbor_df debe ser único por clave
    assert neighbor_df.duplicated(["station_id", "timestamp"]).sum() == 0

    # Merge seguro (si falla, levanta error y evita “duplicar silenciosamente”)
    df_reset = df_reset.merge(
        neighbor_df,
        on=["station_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    df = df_reset.set_index(["station_id", "timestamp"]).sort_index()

    # --- Keep only rows with targets available (y_h24 not null) and pm25 present ---
    df = df[~df["pm25"].isna()].copy()
    df = df[~df[f"y_h{cfg.horizon:02d}"].isna()].copy()

    # --- Coverage report ---
    feat_cols = [
        c
        for c in df.columns
        if c
        not in ["split", "hour", "dow", "month", "is_weekend", "pm25", "pm10", "o3"]
        and not c.startswith("y_h")
    ]
    coverage = (
        (1 - df[feat_cols].isna().mean()).sort_values(ascending=True).reset_index()
    )
    coverage.columns = ["feature", "coverage_non_na"]
    coverage.to_csv(
        cfg.out_reports_tables_dir / "features_h24_coverage_v1.csv", index=False
    )

    # Row counts by split
    rowcounts = df.reset_index().groupby("split").size().reset_index(name="n_rows")
    rowcounts.to_csv(
        cfg.out_reports_tables_dir / "features_h24_rowcounts_v1.csv", index=False
    )

    # Save processed parquet
    out = cfg.out_processed_parquet
    df_reset_out = df.reset_index()
    df_reset_out.to_parquet(out, index=False)
    LOGGER.info("Saved features: %s", out)
    LOGGER.info("Rowcounts:\n%s", rowcounts.to_string(index=False))


if __name__ == "__main__":
    _configure_logging()
    cfg = FeaturesH24Config()
    build_features(cfg)
