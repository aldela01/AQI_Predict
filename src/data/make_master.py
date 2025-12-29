# src/data/make_master.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MasterBuildConfig:
    raw_master_csv: Path
    coords_csv: Path

    # CSV parsing
    sep: str = ";"
    decimal: str = ","
    timestamp_col: str = "Fecha_Hora"

    # coords schema (override if your file uses different names)
    coords_station_col: str = "station_id"
    coords_lat_col: str = "lat"
    coords_lon_col: str = "lon"

    # outputs
    out_interim_dir: Path = Path("data/interim")
    out_reports_tables_dir: Path = Path("reports/tables")

    # neighbors
    k_neighbors: int = 5

    # split boundaries (inclusive ends)
    train_start: str = "2019-01-01"
    train_end: str = "2021-12-31 23:00:00"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31 23:00:00"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31 23:00:00"


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance (km)."""
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def assign_split(ts: pd.Series, cfg: MasterBuildConfig) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    split = np.select(
        [
            (ts >= cfg.train_start) & (ts <= cfg.train_end),
            (ts >= cfg.test_start) & (ts <= cfg.test_end),
            (ts >= cfg.val_start) & (ts <= cfg.val_end),
        ],
        ["train", "test", "val"],
        default="out_of_scope",
    )
    return pd.Series(split, index=ts.index)


def validate_hourly_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Return a table of gaps/duplicates for reporting."""
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    out = []

    n_null = int(ts.isna().sum())
    if n_null:
        out.append({"issue": "null_timestamp", "count": n_null})

    # duplicates
    dup = ts.duplicated(keep=False)
    if dup.any():
        out.append({"issue": "duplicate_timestamp_rows", "count": int(dup.sum())})

    # gaps (assuming intended hourly grid)
    ts_clean = ts.dropna().sort_values()
    if len(ts_clean) >= 2:
        diffs = ts_clean.diff().dropna()
        non_hour = diffs != pd.Timedelta(hours=1)
        if non_hour.any():
            # list top 200 anomalies for reporting
            bad = pd.DataFrame(
                {
                    "prev_ts": ts_clean.shift(1)[diffs.index],
                    "ts": ts_clean.loc[diffs.index],
                    "delta": diffs.astype(str),
                }
            )
            bad = bad.loc[non_hour].head(200)
            # keep as separate csv later
            return bad

    return pd.DataFrame(columns=["prev_ts", "ts", "delta"])


def coerce_measurements(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    meas_cols = [c for c in df.columns if c not in exclude_cols]
    for c in meas_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # defensive rule for SIATA convention
        df.loc[df[c] == -9999, c] = np.nan
    return df, meas_cols


def melt_suffix(df: pd.DataFrame, id_cols: List[str], meas_cols: List[str], suffix: str, value_name: str) -> pd.DataFrame:
    cols = [c for c in meas_cols if c.endswith(suffix)]
    if not cols:
        return pd.DataFrame(columns=id_cols + ["station_id", value_name])

    tmp = df[id_cols + cols].melt(
        id_vars=id_cols,
        value_vars=cols,
        var_name="station_var",
        value_name=value_name,
    )
    tmp["station_id"] = tmp["station_var"].str.split("__").str[0]
    return tmp.drop(columns=["station_var"])


def build_neighbors(stations: pd.DataFrame, k: int) -> Dict[str, List[Dict[str, float]]]:
    st = stations.dropna(subset=["lat", "lon"]).copy()
    ids = st["station_id"].astype(str).tolist()
    lat = st["lat"].to_numpy(dtype=float)
    lon = st["lon"].to_numpy(dtype=float)

    n = len(st)
    if n == 0:
        return {}

    # distance matrix
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, :] = haversine_km(lat[i], lon[i], lat, lon)

    neighbors: Dict[str, List[Dict[str, float]]] = {}
    for i, sid in enumerate(ids):
        idx = np.argsort(D[i])[1 : min(k + 1, n)]  # exclude self
        neighbors[sid] = [{"neighbor": ids[j], "dist_km": float(D[i, j])} for j in idx]
    return neighbors


def build_master(cfg: MasterBuildConfig) -> None:
    cfg.out_interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_reports_tables_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading raw master: %s", cfg.raw_master_csv)
    df = pd.read_csv(cfg.raw_master_csv, sep=cfg.sep, decimal=cfg.decimal, low_memory=False)

    # validate timestamps & report gaps
    gaps_df = validate_hourly_index(df, cfg.timestamp_col)
    if not gaps_df.empty:
        gaps_path = cfg.out_reports_tables_dir / "timestamp_gaps_v1.csv"
        gaps_df.to_csv(gaps_path, index=False)
        LOGGER.warning("Found non-hourly gaps/steps. See %s", gaps_path)

    # timestamp + basic ordering
    df["timestamp"] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df = df.drop(columns=[cfg.timestamp_col]).sort_values("timestamp")

    # split + calendar
    df["split"] = assign_split(df["timestamp"], cfg)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # coerce measurements
    exclude_cols = ["timestamp", "split", "hour", "dow", "month", "is_weekend"]
    df, meas_cols = coerce_measurements(df, exclude_cols)

    # coverage by variable (wide)
    cov_var = (1 - df[meas_cols].isna().mean()).sort_values(ascending=True).reset_index()
    cov_var.columns = ["variable", "coverage_non_na"]
    (cfg.out_reports_tables_dir / "coverage_by_variable_v1.csv").write_text(
        cov_var.to_csv(index=False), encoding="utf-8"
    )

    # save wide parquet
    wide_out = cfg.out_interim_dir / "master_wide_v1.parquet"
    df.to_parquet(wide_out, index=False)
    LOGGER.info("Saved: %s", wide_out)

    # build long (merge only the suffixes you care about; extend later as needed)
    id_cols = ["timestamp", "split", "hour", "dow", "month", "is_weekend"]

    pm25 = melt_suffix(df, id_cols, meas_cols, "__PM2.5", "pm25")
    pm10 = melt_suffix(df, id_cols, meas_cols, "__PM10", "pm10")
    o3 = melt_suffix(df, id_cols, meas_cols, "__O3", "o3")

    # meteorology examples (adjust suffixes if your columns differ)
    tair = melt_suffix(df, id_cols, meas_cols, "__TAire10_SSR", "tair_10m")
    rh = melt_suffix(df, id_cols, meas_cols, "__HAire10_SSR", "rh_10m")
    press = melt_suffix(df, id_cols, meas_cols, "__P_SSR", "press")
    rglob = melt_suffix(df, id_cols, meas_cols, "__RGlobal_SSR", "r_global")
    wspd = melt_suffix(df, id_cols, meas_cols, "__VViento_SSR", "wind_speed")
    wdir = melt_suffix(df, id_cols, meas_cols, "__DViento_SSR", "wind_dir")

    # progressive merges
    long = pm25
    for add in [pm10, o3, tair, rh, press, rglob, wspd, wdir]:
        if not add.empty:
            long = long.merge(add, on=id_cols + ["station_id"], how="left")

    long_out = cfg.out_interim_dir / "master_long_v1.parquet"
    long.to_parquet(long_out, index=False)
    LOGGER.info("Saved: %s", long_out)

    # stations catalog + coords
    stations = pd.DataFrame({"station_id": sorted(long["station_id"].dropna().astype(str).unique())})

    LOGGER.info("Reading coords: %s", cfg.coords_csv)
    coords = pd.read_csv(cfg.coords_csv)

    coords = coords.rename(
        columns={
            cfg.coords_station_col: "station_id",
            cfg.coords_lat_col: "lat",
            cfg.coords_lon_col: "lon",
        }
    )
    coords["station_id"] = coords["station_id"].astype(str)

    stations = stations.merge(coords[["station_id", "lat", "lon"]], on="station_id", how="left")

    # coverage by station (long)
    vars_for_cov = [c for c in ["pm25", "pm10", "o3", "tair_10m", "rh_10m", "press", "r_global", "wind_speed", "wind_dir"] if c in long.columns]
    cov_station = (1 - long.groupby("station_id")[vars_for_cov].apply(lambda x: x.isna().mean())).reset_index()
    cov_station = cov_station.rename(columns={c: f"coverage_{c}" for c in vars_for_cov})
    stations = stations.merge(cov_station, on="station_id", how="left")

    stations_out = cfg.out_interim_dir / "stations_catalog_v1.parquet"
    stations.to_parquet(stations_out, index=False)
    LOGGER.info("Saved: %s", stations_out)

    cov_station_csv = cfg.out_reports_tables_dir / "coverage_by_station_v1.csv"
    stations[["station_id"] + [c for c in stations.columns if c.startswith("coverage_")]].to_csv(cov_station_csv, index=False)

    # neighbors
    neighbors = build_neighbors(stations, cfg.k_neighbors)
    neigh_out = cfg.out_interim_dir / f"neighbors_by_station_k{cfg.k_neighbors}.json"
    with open(neigh_out, "w", encoding="utf-8") as f:
        json.dump(neighbors, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved: %s", neigh_out)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    _configure_logging()

    # Minimal default wiring; prefer passing via scripts/run_preprocess.py
    cfg = MasterBuildConfig(
        raw_master_csv=Path("data/raw/siata_merged_data.csv"),
        coords_csv=Path("data/external/stations_coords.csv"),
    )
    build_master(cfg)
