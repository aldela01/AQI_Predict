from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from statsmodels.tsa.statespace.sarimax import SARIMAX

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarimaConfig:
    order: Tuple[int, int, int] = (1, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)
    trend: str | None = "c"  # constant
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 300

    # Parallelism
    n_jobs: int = -1

    # Data handling
    freq: str = "h"
    interp_limit_hours: int = 6  # only interpolate short gaps in training


def _prepare_station_series(
    df_station: pd.DataFrame, cfg: SarimaConfig
) -> Tuple[pd.Series, pd.Timestamp]:
    """
    Return:
      - full hourly pm25 series (may contain NaNs)
      - train_end timestamp
    """
    df = df_station.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[["timestamp", "pm25", "split"]]

    # Build hourly index
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Empty timestamp range")

    full_idx = pd.date_range(start=start, end=end, freq=cfg.freq)
    s = df.set_index("timestamp")["pm25"].reindex(full_idx)

    # Train end is last timestamp labeled train in the original df (not the reindexed)
    train_end = df.loc[df["split"] == "train", "timestamp"].max()
    if pd.isna(train_end):
        raise ValueError("Station has no train data")

    return s, train_end


def _fit_forecast_station(
    station_id: str,
    df_station: pd.DataFrame,
    eval_rows: pd.DataFrame,
    horizon_h: int,
    cfg: SarimaConfig,
) -> Optional[pd.DataFrame]:
    """
    Fits SARIMA on train portion only, then produces a *single* long out-of-sample forecast path
    from train_end+1 to end+H.

    Prediction for an origin at time t is taken as forecasted value at t+H.
    This is efficient and avoids expensive rolling re-forecast loops.

    Returns a dataframe with columns:
      station_id, timestamp (origin), y_true, y_pred
    """
    try:
        series_full, train_end = _prepare_station_series(df_station, cfg)
    except Exception as e:
        LOGGER.warning("[SARIMA] %s skipped: %s", station_id, e)
        return None

    # Build training endog (hourly, up to train_end)
    s_train = series_full.loc[:train_end].copy()

    # Interpolate short gaps only, then ffill/bfill in train segment
    if s_train.isna().any():
        s_train = s_train.interpolate(method="time", limit=cfg.interp_limit_hours)
        s_train = s_train.ffill().bfill()

    # If still NaN or too short, skip
    if s_train.isna().any() or len(s_train) < 24 * 7:
        LOGGER.warning(
            "[SARIMA] %s skipped: insufficient clean train length (%d)",
            station_id,
            len(s_train),
        )
        return None

    # Fit SARIMA
    try:
        model = SARIMAX(
            endog=s_train,
            order=cfg.order,
            seasonal_order=cfg.seasonal_order,
            trend=cfg.trend,
            enforce_stationarity=cfg.enforce_stationarity,
            enforce_invertibility=cfg.enforce_invertibility,
        )
        res = model.fit(disp=False, maxiter=cfg.maxiter)
    except Exception as e:
        LOGGER.warning("[SARIMA] %s fit failed: %s", station_id, e)
        return None

    # Forecast from train_end+1 to end+H
    end_full = series_full.index.max()
    steps = int(((end_full - train_end) / pd.Timedelta(hours=1))) + horizon_h
    if steps <= 0:
        return None

    try:
        fc = res.get_forecast(steps=steps).predicted_mean
        fc.index = pd.date_range(
            start=train_end + pd.Timedelta(hours=1), periods=steps, freq=cfg.freq
        )
    except Exception as e:
        LOGGER.warning("[SARIMA] %s forecast failed: %s", station_id, e)
        return None

    # Map each origin timestamp -> forecast at origin+h
    eval_rows = eval_rows.sort_values("timestamp").copy()
    eval_rows["timestamp"] = pd.to_datetime(eval_rows["timestamp"], errors="coerce")
    eval_rows = eval_rows.dropna(subset=["timestamp"])

    target_times = eval_rows["timestamp"] + pd.to_timedelta(horizon_h, unit="h")
    yhat = fc.reindex(target_times).to_numpy()

    out = pd.DataFrame(
        {
            "station_id": station_id,
            "timestamp": eval_rows["timestamp"].to_numpy(),
            "y_true": eval_rows[f"y_h{horizon_h:02d}"].to_numpy(),
            "y_pred": yhat,
        }
    )

    out = out[np.isfinite(out["y_true"]) & np.isfinite(out["y_pred"])].copy()
    if len(out) == 0:
        return None
    return out


def sarima_predict_horizon(
    df: pd.DataFrame,
    horizon_h: int = 24,
    cfg: Optional[SarimaConfig] = None,
    split_eval: str = "test",
) -> pd.DataFrame:
    """
    Run seasonal SARIMA per station in parallel and return concatenated predictions.

    Parameters
    ----------
    df : DataFrame containing at least: station_id, timestamp, split, pm25, y_hXX
    horizon_h : horizon to evaluate (e.g., 24)
    split_eval : "test" or "val"
    """
    if cfg is None:
        cfg = SarimaConfig()

    need_cols = {"station_id", "timestamp", "split", "pm25", f"y_h{horizon_h:02d}"}
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df_eval = df[df["split"] == split_eval].copy()
    df_eval = df_eval[df_eval[f"y_h{horizon_h:02d}"].notna()].copy()
    if df_eval.empty:
        raise ValueError(f"No evaluation rows found for split={split_eval}")

    # Group station data and eval rows
    stations = sorted(df["station_id"].dropna().unique().tolist())

    def _station_task(st):
        df_st = df[df["station_id"] == st].copy()
        eval_st = df_eval[df_eval["station_id"] == st].copy()
        if eval_st.empty:
            return None
        return _fit_forecast_station(st, df_st, eval_st, horizon_h, cfg)

    results = Parallel(n_jobs=cfg.n_jobs, prefer="processes")(
        delayed(_station_task)(st) for st in stations
    )

    parts = [r for r in results if r is not None and len(r) > 0]
    if not parts:
        raise RuntimeError(
            "SARIMA produced no predictions. Check logs for skipped stations."
        )
    return pd.concat(parts, ignore_index=True)
