"""
Baseline forecasting pipeline for the Historico_pm25_filtered dataset.

Run a quick train/validation split and report MAE/RMSE for a given station
and forecast horizon (in hours). The script relies on the existing
AirQualityPreprocessor to create time and lag features.

Example:
    python train_forecasting_model.py \\
        --csv Solicitud_Historica/Historico_pm25_filtered.csv \\
        --target BAR-TORR \\
        --horizon 6
"""

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from air_quality_preprocessor_forecasting import AirQualityPreprocessor


@dataclass
class SupervisedSet:
    X: pd.DataFrame
    y: np.ndarray
    index: pd.DatetimeIndex
    numeric_cols: List[str]
    categorical_cols: List[str]
    preprocessor: AirQualityPreprocessor


def _to_datetime(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def load_hourly_csv(path: str) -> pd.DataFrame:
    """
    Load the historical CSV and standardize the timestamp column name.
    The file is semicolon-separated and the first column holds the time.
    """
    df = pd.read_csv(path, sep=";")
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    return df


def build_supervised_dataset(
    df_raw: pd.DataFrame,
    target_col: str,
    horizon: int = 6,
    freq: str = "1H",
    max_lag: int = 48,
    missing_col_threshold: float = 0.5,
    use_cyclical_time: bool = True,
) -> SupervisedSet:
    """
    Turn the wide hourly table into a supervised learning matrix for a lead time.

    - horizon: how many hours ahead to predict (h steps).
    - max_lag: largest lag to include as features (1..max_lag).
    - missing_col_threshold: drop columns with > threshold missing ratio.
    """
    pre = AirQualityPreprocessor(
        target_col=target_col,
        datetime_col="timestamp",
        freq=freq,
        max_lag=max_lag,
        missing_col_threshold=missing_col_threshold,
        use_cyclical_time=use_cyclical_time,
    )

    pre.fit(df_raw)
    df_feat, _, _, idx = pre.transform(df_raw)

    df_feat = df_feat.copy()
    df_feat["target_future"] = df_feat[target_col].shift(-horizon)

    # Drop rows without full lag context or future target
    start = max_lag if max_lag else 0
    end = len(df_feat) - horizon if horizon > 0 else len(df_feat)
    df_supervised = df_feat.iloc[start:end]

    y = df_supervised["target_future"].to_numpy()
    # Keep contemporaneous measurements (target and other stations) plus lags/time features
    # Only drop the shifted future target
    feature_df = df_supervised.drop(columns=["target_future"], errors="ignore")
    numeric_cols = [
        c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])
    ]
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    mask = ~np.isnan(y)
    X = feature_df.iloc[mask]
    y = y[mask]
    supervised_index = df_supervised.index[mask]

    return SupervisedSet(
        X=X,
        y=y,
        index=supervised_index,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        preprocessor=pre,
    )


def build_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_estimators: int,
    max_depth: int,
) -> Pipeline:
    """Create a preprocessing + model pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def split_by_date_ranges(
    sup: SupervisedSet,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    val_start: str,
    val_end: str,
) -> dict:
    """Slice the supervised set into train/test/val using calendar windows."""
    idx = sup.index
    train_mask = (idx >= _to_datetime(train_start)) & (idx <= _to_datetime(train_end))
    test_mask = (idx >= _to_datetime(test_start)) & (idx <= _to_datetime(test_end))
    val_mask = (idx >= _to_datetime(val_start)) & (idx <= _to_datetime(val_end))

    def pack(mask):
        return {
            "X": sup.X.loc[mask],
            "y": sup.y[mask],
            "index": idx[mask],
        }

    return {"train": pack(train_mask), "test": pack(test_mask), "val": pack(val_mask)}


def train_and_evaluate(
    csv_path: str,
    target_col: str,
    horizon: int,
    max_lag: int,
    n_estimators: int,
    max_depth: int,
    missing_col_threshold: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    val_start: str,
    val_end: str,
) -> None:
    """Train a baseline RandomForest forecaster and report metrics."""
    df_raw = load_hourly_csv(csv_path)
    supervised = build_supervised_dataset(
        df_raw=df_raw,
        target_col=target_col,
        horizon=horizon,
        max_lag=max_lag,
        missing_col_threshold=missing_col_threshold,
    )

    splits = split_by_date_ranges(
        supervised,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        val_start=val_start,
        val_end=val_end,
    )

    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    test_index = splits["test"]["index"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    val_index = splits["val"]["index"]

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError(
            "One of the splits is empty. Please check the date windows or horizon/lag settings."
        )

    pipe = build_pipeline(
        supervised.numeric_cols,
        supervised.categorical_cols,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = pipe.score(X_test, y_test)

    val_mae = val_rmse = val_r2 = None
    if len(X_val) > 0:
        val_preds = pipe.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_rmse = mean_squared_error(y_val, val_preds, squared=False)
        val_r2 = pipe.score(X_val, y_val)

    results = pd.DataFrame(
        {"timestamp": test_index, "y_true": y_test, "y_pred": preds}
    )

    print(f"Target station: {target_col}")
    print(f"Horizon: {horizon} hours ahead")
    print(
        f"Date windows -> train: {train_start} to {train_end}, "
        f"test: {test_start} to {test_end}, "
        f"val: {val_start} to {val_end}"
    )
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}, Val size: {len(y_val)}")
    print(f"Number of raw feature columns (pre-encoding): {len(supervised.X.columns)}")
    print("Feature columns (truncated to first 40):")
    for name in list(supervised.X.columns)[:40]:
        print(f"  - {name}")
    if len(supervised.X.columns) > 40:
        print(f"  ... (+{len(supervised.X.columns) - 40} more)")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    if val_mae is not None:
        print(f"Val MAE:  {val_mae:.2f}")
        print(f"Val RMSE: {val_rmse:.2f}")
        print(f"Val R²:   {val_r2:.3f}")

    results_head = results.head()
    print("\nFirst few predictions:\n", results_head)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline forecasting model for PM2.5."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="Solicitud_Historica/Historico_pm25_filtered.csv",
        help="Path to the semicolon-delimited historical CSV.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="BAR-TORR",
        help="Station column to forecast.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Forecast horizon in hours (lead time).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=48,
        help="Maximum lag (hours) to include as features.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the RandomForest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=15,
        help="Maximum tree depth (use None for fully grown trees).",
    )
    parser.add_argument(
        "--missing-col-threshold",
        type=float,
        default=0.5,
        help="Drop columns whose missing ratio exceeds this threshold.",
    )
    parser.add_argument("--train-start", type=str, default="2019-01-01")
    parser.add_argument("--train-end", type=str, default="2021-12-31")
    parser.add_argument("--test-start", type=str, default="2022-01-01")
    parser.add_argument("--test-end", type=str, default="2023-12-31")
    parser.add_argument("--val-start", type=str, default="2024-01-01")
    parser.add_argument("--val-end", type=str, default="2024-12-31")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        csv_path=args.csv,
        target_col=args.target,
        horizon=args.horizon,
        max_lag=args.max_lag,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        missing_col_threshold=args.missing_col_threshold,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        val_start=args.val_start,
        val_end=args.val_end,
    )
