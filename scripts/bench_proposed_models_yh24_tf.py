from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# MLflow
try:
    import mlflow

    MLFLOW_OK = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_OK = False

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.ar.sarima_seasonal import SarimaConfig, sarima_predict_horizon
from src.models.dl.tf_models import (
    TFConfig,
    configure_tf_runtime,
    build_mlp,
    build_lstm,
    build_transformer,
    compile_and_fit,
    predict_numpy,
)

try:
    import tensorflow as tf
except Exception:
    tf = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cfg:
    experiment_name: str = "pm25_h24"
    run_name: str = "benchmark_proposed_models_yh24_tf"

    dataset: Path = Path("data/processed/features_h24_perm90_yh24.parquet")
    reports_dir: Path = Path("reports/tables")
    target_h: int = 24

    # SARIMA
    sarima_order: Tuple[int, int, int] = (1, 0, 1)
    sarima_seasonal: Tuple[int, int, int, int] = (1, 0, 1, 24)
    sarima_n_jobs: int = -1

    # Linear regression
    ridge_alpha: float = 1.0

    # Random Forest
    rf_n_estimators: int = 300
    rf_max_depth: int = 0  # 0=None
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"  # sqrt|log2|auto|<float>|<int>
    rf_n_jobs: int = -1

    # TF runtime
    tf_seed: int = 42
    tf_gpu_growth: bool = True
    tf_mixed_precision: bool = False
    tf_xla: bool = False
    tf_intra_threads: int = 0
    tf_inter_threads: int = 0

    # DL train
    epochs: int = 15
    batch_size: int = 2048
    seq_batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3

    # Tabular MLP
    emb_dim: int = 8
    mlp_hidden: str = "256,256,128"
    dropout: float = 0.1

    # Sequence
    lookback: int = 168
    lstm_units: int = 128
    lstm_layers: int = 2
    tf_d_model: int = 128
    tf_heads: int = 8
    tf_layers: int = 3

    # Controls
    max_train_rows: int = 0  # 0=all
    seq_max_rows: int = 250000  # cap seq samples for speed/memory, per split (0=all)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "n": 0.0}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "rmse": float(root_mean_squared_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "n": float(mask.sum()),
    }


def maybe_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n and len(df) > n:
        return df.sample(n=n, random_state=seed)
    return df


def build_tabular_xy(
    df: pd.DataFrame, y_col: str
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    y_cols = [c for c in df.columns if c.startswith("y_h")]
    exclude = set(y_cols) | {"timestamp", "split"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[y_col].to_numpy(dtype=float)
    return X, y, feature_cols


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if c == "station_id"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def fit_eval_ridge(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float]:
    y_col = f"y_h{cfg.target_h:02d}"

    train = maybe_subsample(df[df["split"] == "train"].copy(), cfg.max_train_rows)
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    train = train[train[y_col].notna()].copy()
    test = test[test[y_col].notna()].copy()
    val = val[val[y_col].notna()].copy()

    X_train, y_train, feat = build_tabular_xy(train, y_col)
    X_test, y_test, _ = build_tabular_xy(test, y_col)
    X_val, y_val, _ = build_tabular_xy(val, y_col)

    pre = build_preprocessor(feat)
    ridge = Ridge(alpha=cfg.ridge_alpha, random_state=42)
    pipe = Pipeline([("preprocess", pre), ("model", ridge)])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0

    pred_test = pipe.predict(X_test)
    pred_val = pipe.predict(X_val)

    mt = compute_metrics(y_test, pred_test)
    mv = compute_metrics(y_val, pred_val)

    return {
        "model": "ridge",
        "fit_seconds": float(fit_s),
        "rmse_test": mt["rmse"],
        "r2_test": mt["r2"],
        "n_test": mt["n"],
        "rmse_val": mv["rmse"],
        "r2_val": mv["r2"],
        "n_val": mv["n"],
        "notes": f"alpha={cfg.ridge_alpha}",
    }


def fit_eval_rf(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float]:
    """
    Random Forest regression baseline for direct horizon y_hXX.

    Notes on encoding:
    - We use OrdinalEncoder for station_id to avoid a very large dense one-hot matrix,
      which can explode memory and model size.
    - Numeric features are median-imputed; no scaling is applied (trees do not require it).
    """
    y_col = f"y_h{cfg.target_h:02d}"

    train = maybe_subsample(df[df["split"] == "train"].copy(), cfg.max_train_rows)
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    train = train[train[y_col].notna()].copy()
    test = test[test[y_col].notna()].copy()
    val = val[val[y_col].notna()].copy()

    X_train, y_train, feat = build_tabular_xy(train, y_col)
    X_test, y_test, _ = build_tabular_xy(test, y_col)
    X_val, y_val, _ = build_tabular_xy(val, y_col)

    cat_cols = ["station_id"] if "station_id" in feat else []
    num_cols = [c for c in feat if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols)
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    max_depth = None if cfg.rf_max_depth == 0 else cfg.rf_max_depth

    # Parse max_features
    mf: object
    if cfg.rf_max_features in ("sqrt", "log2", "auto"):
        mf = cfg.rf_max_features
    else:
        # allow float like "0.5" or int like "20"
        try:
            mf = int(cfg.rf_max_features)
        except Exception:
            mf = float(cfg.rf_max_features)

    rf = RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_depth=max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        max_features=mf,
        n_jobs=cfg.rf_n_jobs,
        random_state=42,
        verbose=0,
    )

    pipe = Pipeline([("preprocess", pre), ("model", rf)])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0

    pred_test = pipe.predict(X_test)
    pred_val = pipe.predict(X_val)

    mt = compute_metrics(y_test, pred_test)
    mv = compute_metrics(y_val, pred_val)

    return {
        "model": "random_forest",
        "fit_seconds": float(fit_s),
        "rmse_test": mt["rmse"],
        "r2_test": mt["r2"],
        "n_test": mt["n"],
        "rmse_val": mv["rmse"],
        "r2_val": mv["r2"],
        "n_val": mv["n"],
        "notes": f"n_estimators={cfg.rf_n_estimators} max_depth={max_depth} min_leaf={cfg.rf_min_samples_leaf} max_features={mf} n_jobs={cfg.rf_n_jobs}",
    }


def fit_eval_sarima(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float]:
    sar_cfg = SarimaConfig(
        order=cfg.sarima_order,
        seasonal_order=cfg.sarima_seasonal,
        n_jobs=cfg.sarima_n_jobs,
    )

    t0 = time.time()
    pred_test_df = sarima_predict_horizon(
        df, horizon_h=cfg.target_h, cfg=sar_cfg, split_eval="test"
    )
    pred_val_df = sarima_predict_horizon(
        df, horizon_h=cfg.target_h, cfg=sar_cfg, split_eval="val"
    )
    fit_s = time.time() - t0

    mt = compute_metrics(
        pred_test_df["y_true"].to_numpy(), pred_test_df["y_pred"].to_numpy()
    )
    mv = compute_metrics(
        pred_val_df["y_true"].to_numpy(), pred_val_df["y_pred"].to_numpy()
    )

    return {
        "model": "sarima_seasonal",
        "fit_seconds": float(fit_s),
        "rmse_test": mt["rmse"],
        "r2_test": mt["r2"],
        "n_test": mt["n"],
        "rmse_val": mv["rmse"],
        "r2_val": mv["r2"],
        "n_val": mv["n"],
        "notes": f"order={cfg.sarima_order} seasonal={cfg.sarima_seasonal}",
    }


def _build_station_index(train_station_ids: pd.Series) -> Dict[str, int]:
    uniq = sorted(train_station_ids.dropna().unique().tolist())
    return {s: i + 1 for i, s in enumerate(uniq)}  # 0 reserved for unknown


def _tabular_arrays(
    df: pd.DataFrame, y_col: str, station_to_idx: Dict[str, int], scaler=None
):
    y_cols = [c for c in df.columns if c.startswith("y_h")]
    exclude = set(y_cols) | {"timestamp", "split"}
    feat_cols = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in feat_cols if c != "station_id"]

    X_num = df[num_cols].to_numpy(dtype=np.float32)
    # impute with train medians (passed via scaler tuple)
    if scaler is None:
        med = np.nanmedian(X_num, axis=0)
        mu = np.nanmean(np.where(np.isfinite(X_num), X_num, np.nan), axis=0)
        sd = np.nanstd(np.where(np.isfinite(X_num), X_num, np.nan), axis=0) + 1e-6
        scaler = (med, mu, sd)
    med, mu, sd = scaler

    inds = np.where(~np.isfinite(X_num))
    if len(inds[0]) > 0:
        X_num[inds] = med[inds[1]]
    X_num = (X_num - mu) / sd

    st = df["station_id"].astype(str).to_numpy()
    st_idx = np.array([station_to_idx.get(s, 0) for s in st], dtype=np.int32)

    y = df[y_col].to_numpy(dtype=np.float32)
    return X_num, st_idx, y, scaler, num_cols


def _make_tabular_ds(X_num, st_idx, y, batch_size: int, shuffle: bool, seed: int):
    ds = tf.data.Dataset.from_tensor_slices(((X_num, st_idx), y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 50000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def fit_eval_mlp_tf(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float]:
    if tf is None:
        raise RuntimeError("TensorFlow not installed")

    y_col = f"y_h{cfg.target_h:02d}"
    train = maybe_subsample(df[df["split"] == "train"].copy(), cfg.max_train_rows)
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    train = train[train[y_col].notna()].copy()
    test = test[test[y_col].notna()].copy()
    val = val[val[y_col].notna()].copy()

    station_to_idx = _build_station_index(train["station_id"])
    dev = train.sample(frac=0.05, random_state=cfg.tf_seed)
    trn = train.drop(index=dev.index)

    Xtr, str_tr, ytr, scaler, num_cols = _tabular_arrays(
        trn, y_col, station_to_idx, scaler=None
    )
    Xdv, str_dv, ydv, _, _ = _tabular_arrays(dev, y_col, station_to_idx, scaler=scaler)
    Xte, str_te, yte, _, _ = _tabular_arrays(test, y_col, station_to_idx, scaler=scaler)
    Xva, str_va, yva, _, _ = _tabular_arrays(val, y_col, station_to_idx, scaler=scaler)

    tr_ds = _make_tabular_ds(
        Xtr, str_tr, ytr, cfg.batch_size, shuffle=True, seed=cfg.tf_seed
    )
    dv_ds = _make_tabular_ds(
        Xdv, str_dv, ydv, cfg.batch_size, shuffle=False, seed=cfg.tf_seed
    )
    te_ds = _make_tabular_ds(
        Xte, str_te, yte, cfg.batch_size, shuffle=False, seed=cfg.tf_seed
    )
    va_ds = _make_tabular_ds(
        Xva, str_va, yva, cfg.batch_size, shuffle=False, seed=cfg.tf_seed
    )

    hidden = tuple(int(x) for x in cfg.mlp_hidden.split(",") if x.strip())
    model = build_mlp(
        n_numeric=Xtr.shape[1],
        n_stations=len(station_to_idx),
        emb_dim=cfg.emb_dim,
        hidden=hidden,
        dropout=cfg.dropout,
    )

    t0 = time.time()
    compile_and_fit(
        model,
        tr_ds,
        dv_ds,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
    )
    fit_s = time.time() - t0

    yhat_test = predict_numpy(model, te_ds)
    yhat_val = predict_numpy(model, va_ds)

    mt = compute_metrics(yte, yhat_test)
    mv = compute_metrics(yva, yhat_val)

    return {
        "model": "mlp_tf",
        "fit_seconds": float(fit_s),
        "rmse_test": mt["rmse"],
        "r2_test": mt["r2"],
        "n_test": mt["n"],
        "rmse_val": mv["rmse"],
        "r2_val": mv["r2"],
        "n_val": mv["n"],
        "notes": f"features={len(num_cols)} emb_dim={cfg.emb_dim}",
    }


def _build_sequences(
    df_all: pd.DataFrame,
    df_split: pd.DataFrame,
    horizon_h: int,
    lookback: int,
    max_rows: int,
    seed: int,
):
    df_all = df_all[["station_id", "timestamp", "pm25"]].copy()
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    df_all = df_all.dropna(subset=["timestamp"])

    df_split = df_split.copy()
    df_split["timestamp"] = pd.to_datetime(df_split["timestamp"], errors="coerce")
    df_split = df_split.dropna(subset=["timestamp"])
    if max_rows and len(df_split) > max_rows:
        df_split = df_split.sample(n=max_rows, random_state=seed)

    # per-station hourly series
    series_map: Dict[str, pd.Series] = {}
    for st, part in df_all.groupby("station_id"):
        part = part.sort_values("timestamp")
        s = part.set_index("timestamp")["pm25"]
        if len(s) == 0:
            continue
        idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
        s = s.reindex(idx).interpolate("time", limit=6).ffill().bfill()
        series_map[str(st)] = s

    ts = df_split["timestamp"]
    hour = ts.dt.hour.to_numpy()
    ang = 2.0 * np.pi * (hour / 24.0)
    hour_sc = np.vstack([np.sin(ang), np.cos(ang)]).T.astype(np.float32)
    dow = (
        df_split.get("dow", ts.dt.dayofweek + 1)
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )
    month = df_split.get("month", ts.dt.month).to_numpy(dtype=np.float32).reshape(-1, 1)
    is_weekend = (
        df_split.get("is_weekend", (ts.dt.dayofweek >= 5).astype(int))
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )
    aux = np.concatenate([hour_sc, dow, month, is_weekend], axis=1).astype(np.float32)

    y_col = f"y_h{horizon_h:02d}"
    y_arr = df_split[y_col].to_numpy(dtype=np.float32)

    seqs, auxs, st_ids, ys = [], [], [], []
    for i, row in enumerate(df_split.itertuples(index=False)):
        st = str(getattr(row, "station_id"))
        t = getattr(row, "timestamp")
        if st not in series_map:
            continue
        s = series_map[st]
        if t not in s.index:
            continue
        loc = s.index.get_loc(t)
        if isinstance(loc, slice) or isinstance(loc, np.ndarray):
            continue
        start = loc - lookback + 1
        if start < 0:
            continue
        window = s.iloc[start : loc + 1].to_numpy(dtype=np.float32)
        if window.shape[0] != lookback or not np.isfinite(window).all():
            continue
        if not np.isfinite(y_arr[i]):
            continue
        seqs.append(window.reshape(lookback, 1))
        auxs.append(aux[i])
        st_ids.append(st)
        ys.append(y_arr[i])

    if not seqs:
        return (
            np.empty((0, lookback, 1), dtype=np.float32),
            np.empty((0, aux.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.stack(seqs, axis=0),
        np.stack(auxs, axis=0),
        np.array(st_ids, dtype=object),
        np.array(ys, dtype=np.float32),
    )


def _make_seq_ds(seq, aux, st_idx, y, batch_size: int, shuffle: bool, seed: int):
    ds = tf.data.Dataset.from_tensor_slices(((seq, aux, st_idx), y))
    if shuffle:
        ds = ds.shuffle(min(len(y), 50000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def fit_eval_lstm_tf(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float]:
    if tf is None:
        raise RuntimeError("TensorFlow not installed")

    y_col = f"y_h{cfg.target_h:02d}"
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    train = train[train[y_col].notna()].copy()
    test = test[test[y_col].notna()].copy()
    val = val[val[y_col].notna()].copy()

    dev = train.sample(frac=0.05, random_state=cfg.tf_seed)
    trn = train.drop(index=dev.index)

    seq_tr, aux_tr, st_tr, y_tr = _build_sequences(
        df, trn, cfg.target_h, cfg.lookback, cfg.seq_max_rows, cfg.tf_seed
    )
    seq_dv, aux_dv, st_dv, y_dv = _build_sequences(
        df,
        dev,
        cfg.target_h,
        cfg.lookback,
        cfg.seq_max_rows // 10 if cfg.seq_max_rows else 0,
        cfg.tf_seed,
    )
    seq_te, aux_te, st_te, y_te = _build_sequences(
        df, test, cfg.target_h, cfg.lookback, cfg.seq_max_rows, cfg.tf_seed + 1
    )
    seq_va, aux_va, st_va, y_va = _build_sequences(
        df, val, cfg.target_h, cfg.lookback, cfg.seq_max_rows, cfg.tf_seed + 2
    )

    station_to_idx = {s: i + 1 for i, s in enumerate(sorted(np.unique(st_tr).tolist()))}
    map_idx = lambda arr: np.array(
        [station_to_idx.get(s, 0) for s in arr], dtype=np.int32
    )

    tr_ds = _make_seq_ds(
        seq_tr,
        aux_tr,
        map_idx(st_tr),
        y_tr,
        cfg.seq_batch_size,
        shuffle=True,
        seed=cfg.tf_seed,
    )
    dv_ds = _make_seq_ds(
        seq_dv,
        aux_dv,
        map_idx(st_dv),
        y_dv,
        cfg.seq_batch_size,
        shuffle=False,
        seed=cfg.tf_seed,
    )
    te_ds = _make_seq_ds(
        seq_te,
        aux_te,
        map_idx(st_te),
        y_te,
        cfg.seq_batch_size,
        shuffle=False,
        seed=cfg.tf_seed,
    )
    va_ds = _make_seq_ds(
        seq_va,
        aux_va,
        map_idx(st_va),
        y_va,
        cfg.seq_batch_size,
        shuffle=False,
        seed=cfg.tf_seed,
    )

    model = build_lstm(
        seq_len=cfg.lookback,
        n_stations=len(station_to_idx),
        emb_dim=cfg.emb_dim,
        aux_dim=aux_tr.shape[1] if aux_tr.size else 5,
        lstm_units=cfg.lstm_units,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    )

    t0 = time.time()
    compile_and_fit(
        model,
        tr_ds,
        dv_ds,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
    )
    fit_s = time.time() - t0

    yhat_test = predict_numpy(model, te_ds)
    yhat_val = predict_numpy(model, va_ds)

    mt = compute_metrics(y_te, yhat_test)
    mv = compute_metrics(y_va, yhat_val)

    return {
        "model": "lstm_tf",
        "fit_seconds": float(fit_s),
        "rmse_test": mt["rmse"],
        "r2_test": mt["r2"],
        "n_test": mt["n"],
        "rmse_val": mv["rmse"],
        "r2_val": mv["r2"],
        "n_val": mv["n"],
        "notes": f"lookback={cfg.lookback} device={'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}",
    }


def fit_eval_transformer_tf(df: pd.DataFrame, cfg: Cfg) -> Dict[str, float]:
    if tf is None:
        raise RuntimeError("TensorFlow not installed")

    y_col = f"y_h{cfg.target_h:02d}"
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "val"].copy()

    train = train[train[y_col].notna()].copy()
    test = test[test[y_col].notna()].copy()
    val = val[val[y_col].notna()].copy()

    dev = train.sample(frac=0.05, random_state=cfg.tf_seed)
    trn = train.drop(index=dev.index)

    seq_tr, aux_tr, st_tr, y_tr = _build_sequences(
        df, trn, cfg.target_h, cfg.lookback, cfg.seq_max_rows, cfg.tf_seed
    )
    seq_dv, aux_dv, st_dv, y_dv = _build_sequences(
        df,
        dev,
        cfg.target_h,
        cfg.lookback,
        cfg.seq_max_rows // 10 if cfg.seq_max_rows else 0,
        cfg.tf_seed,
    )
    seq_te, aux_te, st_te, y_te = _build_sequences(
        df, test, cfg.target_h, cfg.lookback, cfg.seq_max_rows, cfg.tf_seed + 1
    )
    seq_va, aux_va, st_va, y_va = _build_sequences(
        df, val, cfg.target_h, cfg.lookback, cfg.seq_max_rows, cfg.tf_seed + 2
    )

    station_to_idx = {s: i + 1 for i, s in enumerate(sorted(np.unique(st_tr).tolist()))}
    map_idx = lambda arr: np.array(
        [station_to_idx.get(s, 0) for s in arr], dtype=np.int32
    )

    tr_ds = _make_seq_ds(
        seq_tr,
        aux_tr,
        map_idx(st_tr),
        y_tr,
        cfg.seq_batch_size,
        shuffle=True,
        seed=cfg.tf_seed,
    )
    dv_ds = _make_seq_ds(
        seq_dv,
        aux_dv,
        map_idx(st_dv),
        y_dv,
        cfg.seq_batch_size,
        shuffle=False,
        seed=cfg.tf_seed,
    )
    te_ds = _make_seq_ds(
        seq_te,
        aux_te,
        map_idx(st_te),
        y_te,
        cfg.seq_batch_size,
        shuffle=False,
        seed=cfg.tf_seed,
    )
    va_ds = _make_seq_ds(
        seq_va,
        aux_va,
        map_idx(st_va),
        y_va,
        cfg.seq_batch_size,
        shuffle=False,
        seed=cfg.tf_seed,
    )

    model = build_transformer(
        seq_len=cfg.lookback,
        n_stations=len(station_to_idx),
        emb_dim=cfg.emb_dim,
        aux_dim=aux_tr.shape[1] if aux_tr.size else 5,
        d_model=cfg.tf_d_model,
        nhead=cfg.tf_heads,
        layers=cfg.tf_layers,
        dropout=cfg.dropout,
    )

    t0 = time.time()
    compile_and_fit(
        model,
        tr_ds,
        dv_ds,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
    )
    fit_s = time.time() - t0

    yhat_test = predict_numpy(model, te_ds)
    yhat_val = predict_numpy(model, va_ds)

    mt = compute_metrics(y_te, yhat_test)
    mv = compute_metrics(y_va, yhat_val)

    return {
        "model": "transformer_tf",
        "fit_seconds": float(fit_s),
        "rmse_test": mt["rmse"],
        "r2_test": mt["r2"],
        "n_test": mt["n"],
        "rmse_val": mv["rmse"],
        "r2_val": mv["r2"],
        "n_val": mv["n"],
        "notes": f"lookback={cfg.lookback} d_model={cfg.tf_d_model}",
    }


def log_run_child(name: str, res: Dict[str, float], params: Dict[str, object]):
    if not MLFLOW_OK:
        return
    with mlflow.start_run(run_name=name, nested=True):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in res.items()
                if k in ["rmse_test", "r2_test", "rmse_val", "r2_val", "fit_seconds"]
            }
        )


def main(cfg: Cfg) -> None:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.dataset)
    df = df[df["split"].isin(["train", "test", "val"])].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    tf_info = configure_tf_runtime(
        TFConfig(
            seed=cfg.tf_seed,
            enable_gpu_memory_growth=cfg.tf_gpu_growth,
            mixed_precision=cfg.tf_mixed_precision,
            xla=cfg.tf_xla,
            intra_op_threads=cfg.tf_intra_threads,
            inter_op_threads=cfg.tf_inter_threads,
        )
    )
    LOGGER.info("TF runtime: %s", tf_info)

    y_col = f"y_h{cfg.target_h:02d}"

    if MLFLOW_OK:
        mlflow.set_experiment(cfg.experiment_name)

    if MLFLOW_OK:
        mlflow.start_run(run_name=cfg.run_name)
        mlflow.log_params(
            {
                "dataset": str(cfg.dataset),
                "target": y_col,
                "target_h": cfg.target_h,
                "sarima_order": str(cfg.sarima_order),
                "sarima_seasonal": str(cfg.sarima_seasonal),
                "ridge_alpha": cfg.ridge_alpha,
                "rf_n_estimators": cfg.rf_n_estimators,
                "rf_max_depth": cfg.rf_max_depth,
                "rf_min_samples_leaf": cfg.rf_min_samples_leaf,
                "rf_max_features": cfg.rf_max_features,
                "rf_n_jobs": cfg.rf_n_jobs,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "seq_batch_size": cfg.seq_batch_size,
                "lookback": cfg.lookback,
                "mlp_hidden": cfg.mlp_hidden,
                "emb_dim": cfg.emb_dim,
                "tf_info": str(tf_info),
            }
        )
        mlflow.set_tags(
            {
                "stage": "benchmark",
                "scope": y_col,
                "feature_set": "PERM90",
                "dl_framework": "tensorflow",
                "horizon": str(cfg.target_h),
            }
        )

    try:
        rows = []

        LOGGER.info("Running SARIMA seasonal...")
        res = fit_eval_sarima(df, cfg)
        rows.append(res)
        log_run_child("sarima_seasonal", res, {"model_family": "arima"})
        LOGGER.info(
            "SARIMA | Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            res["rmse_test"],
            res["r2_test"],
            res["rmse_val"],
            res["r2_val"],
        )

        LOGGER.info("Running Ridge regression...")
        res = fit_eval_ridge(df, cfg)
        rows.append(res)
        log_run_child(
            "ridge", res, {"model_family": "linear", "alpha": cfg.ridge_alpha}
        )
        LOGGER.info(
            "Ridge | Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            res["rmse_test"],
            res["r2_test"],
            res["rmse_val"],
            res["r2_val"],
        )

        LOGGER.info("Running Random Forest regression...")
        res = fit_eval_rf(df, cfg)
        rows.append(res)
        log_run_child(
            "random_forest",
            res,
            {
                "model_family": "tree",
                "n_estimators": cfg.rf_n_estimators,
                "max_depth": cfg.rf_max_depth,
                "min_samples_leaf": cfg.rf_min_samples_leaf,
                "max_features": cfg.rf_max_features,
                "n_jobs": cfg.rf_n_jobs,
            },
        )
        LOGGER.info(
            "RF | Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            res["rmse_test"],
            res["r2_test"],
            res["rmse_val"],
            res["r2_val"],
        )

        LOGGER.info("Running MLP (TensorFlow)...")
        res = fit_eval_mlp_tf(df, cfg)
        rows.append(res)
        log_run_child("mlp_tf", res, {"model_family": "dl", "arch": "mlp"})
        LOGGER.info(
            "MLP | Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            res["rmse_test"],
            res["r2_test"],
            res["rmse_val"],
            res["r2_val"],
        )

        LOGGER.info("Running LSTM (TensorFlow)...")
        res = fit_eval_lstm_tf(df, cfg)
        rows.append(res)
        log_run_child(
            "lstm_tf",
            res,
            {"model_family": "dl", "arch": "lstm", "lookback": cfg.lookback},
        )
        LOGGER.info(
            "LSTM | Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            res["rmse_test"],
            res["r2_test"],
            res["rmse_val"],
            res["r2_val"],
        )

        LOGGER.info("Running Transformer (TensorFlow)...")
        res = fit_eval_transformer_tf(df, cfg)
        rows.append(res)
        log_run_child(
            "transformer_tf",
            res,
            {"model_family": "dl", "arch": "transformer", "lookback": cfg.lookback},
        )
        LOGGER.info(
            "Transformer | Test RMSE=%.4f R2=%.4f | Val RMSE=%.4f R2=%.4f",
            res["rmse_test"],
            res["r2_test"],
            res["rmse_val"],
            res["r2_val"],
        )

        out = pd.DataFrame(rows).sort_values("rmse_val")
        out_path = cfg.reports_dir / f"benchmark_proposed_models_{y_col}_tf.csv"
        out.to_csv(out_path, index=False)

        if MLFLOW_OK:
            mlflow.log_artifact(str(out_path))

        LOGGER.info("Saved benchmark table: %s", out_path)
        LOGGER.info(
            "Winner by val RMSE: %s (rmse_val=%.4f)",
            out.iloc[0]["model"],
            out.iloc[0]["rmse_val"],
        )
    finally:
        if MLFLOW_OK:
            mlflow.end_run()


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="pm25_h24")
    p.add_argument("--run-name", default="benchmark_proposed_models_yh24_tf")
    p.add_argument(
        "--dataset", default="data/processed/features_h24_perm90_yh24.parquet"
    )
    p.add_argument("--reports-dir", default="reports/tables")
    p.add_argument("--target-h", type=int, default=24)

    p.add_argument("--sarima-order", default="1,0,1")
    p.add_argument("--sarima-seasonal", default="1,0,1,24")
    p.add_argument("--sarima-n-jobs", type=int, default=-1)

    p.add_argument("--ridge-alpha", type=float, default=1.0)

    # Random Forest
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=0, help="0 means None")
    p.add_argument("--rf-min-samples-leaf", type=int, default=1)
    p.add_argument("--rf-max-features", default="sqrt")
    p.add_argument("--rf-n-jobs", type=int, default=-1)

    # TF runtime
    p.add_argument("--tf-seed", type=int, default=42)
    p.add_argument("--tf-gpu-growth", action="store_true")
    p.add_argument("--tf-mixed-precision", action="store_true")
    p.add_argument("--tf-xla", action="store_true")
    p.add_argument("--tf-intra-threads", type=int, default=0)
    p.add_argument("--tf-inter-threads", type=int, default=0)

    # DL
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seq-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--emb-dim", type=int, default=8)
    p.add_argument("--mlp-hidden", default="256,256,128")
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--lstm-units", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=2)
    p.add_argument("--tf-d-model", type=int, default=128)
    p.add_argument("--tf-heads", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=3)

    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--seq-max-rows", type=int, default=250000)

    a = p.parse_args()
    order = tuple(int(x) for x in a.sarima_order.split(","))  # type: ignore
    seas = tuple(int(x) for x in a.sarima_seasonal.split(","))  # type: ignore

    # NOTE: tf-gpu-growth default False; if not passed, you can still enable with env or edit config.
    return Cfg(
        experiment_name=a.experiment,
        run_name=a.run_name,
        dataset=Path(a.dataset),
        reports_dir=Path(a.reports_dir),
        target_h=a.target_h,
        sarima_order=order,  # type: ignore
        sarima_seasonal=seas,  # type: ignore
        sarima_n_jobs=a.sarima_n_jobs,
        ridge_alpha=a.ridge_alpha,
        rf_n_estimators=a.rf_n_estimators,
        rf_max_depth=a.rf_max_depth,
        rf_min_samples_leaf=a.rf_min_samples_leaf,
        rf_max_features=a.rf_max_features,
        rf_n_jobs=a.rf_n_jobs,
        tf_seed=a.tf_seed,
        tf_gpu_growth=bool(a.tf_gpu_growth),
        tf_mixed_precision=bool(a.tf_mixed_precision),
        tf_xla=bool(a.tf_xla),
        tf_intra_threads=a.tf_intra_threads,
        tf_inter_threads=a.tf_inter_threads,
        epochs=a.epochs,
        batch_size=a.batch_size,
        seq_batch_size=a.seq_batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        patience=a.patience,
        emb_dim=a.emb_dim,
        mlp_hidden=a.mlp_hidden,
        dropout=a.dropout,
        lookback=a.lookback,
        lstm_units=a.lstm_units,
        lstm_layers=a.lstm_layers,
        tf_d_model=a.tf_d_model,
        tf_heads=a.tf_heads,
        tf_layers=a.tf_layers,
        max_train_rows=a.max_train_rows,
        seq_max_rows=a.seq_max_rows,
    )


if __name__ == "__main__":
    configure_logging()
    cfg = parse_args()
    main(cfg)
