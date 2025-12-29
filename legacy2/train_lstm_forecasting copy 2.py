"""
Autoregressive LSTM forecasting pipeline using the AirQualityPreprocessor and
TensorFlow's WindowGenerator pattern.

This script:
- Builds time windows with tf.keras.utils.timeseries_dataset_from_array.
- Trains an autoregressive LSTM that rolls predictions forward.
- Reports RMSE and R2 for train/test/val splits.
- Logs parameters, metrics, and artifacts to MLflow.

Adjust hyperparameters in CONFIG for full architecture control.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from air_quality_preprocessor_forecasting import AirQualityPreprocessor


VALID_MASK_COLUMN = "__valid__"
TIME_FEATURE_COLUMNS = (
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)


@dataclass
class Config:
    # Data
    csv_path: str = "siata_merged_data.csv"
    datetime_col: str = "Fecha_Hora"
    target_col: str = "BAR-TORR__PM2.5"
    freq: str = "1h"
    out_steps: int = 24  # forecast steps
    lags: list | None = field(
        default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720]
    )
    # lags: list | None = None
    rolling_windows: list | None = field(
        default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720]
    )
    # rolling_windows: list | None = None
    use_cyclical_time: bool = True
    time_features_at: str = "t"
    lag_other_cols: list | None = None
    missing_col_threshold: float | None = 0.5
    auxiliary_stations: list | None = field(
        default_factory=lambda: [
            "MED-ARAN",
            "MED-BEME",
            "SAB-RAME",
            "CEN-TRAF",
            "CAL-JOAR",
        ]
    )

    # Split ranges
    train_start: str = "2019-01-01"
    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31"

    # Windowing
    seq_len: int = 168
    impute_strategy: str = "mean"
    scale_features: bool = True
    scale_target: bool = True  # scaling applies to all columns in autoregressive mode
    use_seasonal_baseline: bool = True
    seasonal_use_month: bool = True

    # Model architecture
    num_lstm_layers: int = 2
    lstm_units: tuple | int = (64, 32)
    bidirectional: bool = False
    dropout: float = 0.1
    recurrent_dropout: float = 0.0
    l2_reg: float = 0.01
    dense_units: tuple | int = (64, 32)
    dense_activation: str = "relu"
    dense_dropout: float = 0.01
    output_activation: str | None = None

    # Training
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    loss: str = "mse"
    shuffle: bool = True
    shuffle_buffer: int = 20000
    use_tf_dataset: bool = True
    use_early_stopping: bool = True
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    use_reduce_lr: bool = True
    reduce_lr_patience: int = 6
    reduce_lr_factor: float = 0.5
    reduce_lr_min_lr: float = 1e-6
    best_monitor: str = "val_r2"
    best_mode: str = "max"

    # Runtime
    seed: int = 42
    use_mixed_precision: bool = False
    use_xla: bool = False
    enable_gpu_memory_growth: bool = True

    # MLflow
    use_mlflow: bool = True
    mlflow_experiment: str = "AQI_LSTM_Forecasting"
    mlflow_run_name: str = "lstm_forecast_run"

    # Feature importance
    compute_feature_importance: bool = True
    feature_importance_split: str = "val"
    feature_importance_metric: str = "r2"
    feature_importance_max_features: int | None = 50
    feature_importance_n_repeats: int = 1
    feature_importance_seed: int = 42

    # Outputs
    output_dir: str = "lstm_results"
    model_dir: str = "models"


class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name: str = "r2", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.sum_sq = self.add_weight(name="sum_sq", initializer="zeros")
        self.sum_sq_res = self.add_weight(name="sum_sq_res", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(y_true, sample_weight)
            y_pred = tf.multiply(y_pred, sample_weight)
        self.sum.assign_add(tf.reduce_sum(y_true))
        self.sum_sq.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.sum_sq_res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        epsilon = tf.keras.backend.epsilon()
        total = self.sum_sq - tf.square(self.sum) / (self.count + epsilon)
        return 1.0 - (self.sum_sq_res / (total + epsilon))

    def reset_state(self) -> None:
        for var in self.variables:
            var.assign(tf.zeros_like(var))


class WindowGenerator:
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray,
        column_names: list[str],
        label_columns: list[str] | None,
        time_feature_columns: list[str],
        batch_size: int,
        shuffle_buffer: int,
        shuffle: bool,
        seed: int,
        mask_column: str = VALID_MASK_COLUMN,
    ) -> None:
        if input_width < 1:
            raise ValueError("input_width must be >= 1")
        if label_width < 1:
            raise ValueError("label_width must be >= 1")
        if shift < 1:
            raise ValueError("shift must be >= 1")

        self.train_data = np.asarray(train_data, dtype=np.float32)
        self.val_data = np.asarray(val_data, dtype=np.float32)
        self.test_data = np.asarray(test_data, dtype=np.float32)
        self.column_names = list(column_names)
        self.label_columns = list(label_columns) if label_columns else None
        self.time_feature_columns = list(time_feature_columns)
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.shuffle = shuffle
        self.seed = seed

        self.column_indices = {name: i for i, name in enumerate(self.column_names)}
        self.label_columns_indices = None
        if self.label_columns is not None:
            self.label_columns_indices = {
                name: self.column_indices[name] for name in self.label_columns
            }
        self.time_feature_indices = [
            self.column_indices[name]
            for name in self.time_feature_columns
            if name in self.column_indices
        ]

        self.mask_index = self.column_indices.get(mask_column)
        self.num_features = len(self.column_names) - (
            1 if self.mask_index is not None else 0
        )
        if (
            self.mask_index is not None
            and self.mask_index != len(self.column_names) - 1
        ):
            raise ValueError("mask_column must be the last column in column_names.")

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self._example = None

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(
        self, features: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        inputs = features[:, self.input_slice, :]
        labels_full = features[:, self.labels_slice, :]
        label_mask = None
        if self.mask_index is not None:
            label_mask = labels_full[:, :, self.mask_index]
            inputs = inputs[:, :, : self.num_features]
        else:
            label_mask = tf.ones(
                [tf.shape(labels_full)[0], self.label_width], dtype=labels_full.dtype
            )

        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels_full[:, :, self.label_columns_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )
        else:
            labels = labels_full[:, :, : self.num_features]

        if self.time_feature_indices:
            future_covariates = tf.gather(
                labels_full, self.time_feature_indices, axis=-1
            )
        else:
            future_covariates = tf.zeros(
                [tf.shape(labels_full)[0], self.label_width, 0],
                dtype=labels_full.dtype,
            )

        inputs.set_shape([None, self.input_width, self.num_features])
        labels.set_shape(
            [
                None,
                self.label_width,
                len(self.label_columns) if self.label_columns else self.num_features,
            ]
        )
        future_covariates.set_shape(
            [None, self.label_width, len(self.time_feature_indices)]
        )
        return inputs, labels, label_mask, future_covariates

    def make_dataset(self, data: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        data = np.asarray(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size,
        )
        ds = ds.map(self.split_window, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.unbatch()
        if self.mask_index is not None:
            ds = ds.filter(
                lambda inputs, labels, mask, future: tf.reduce_all(mask > 0.0)
            )
        ds = ds.map(
            lambda inputs, labels, mask, future: ((inputs, future), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if shuffle:
            ds = ds.shuffle(
                self.shuffle_buffer,
                seed=self.seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    @property
    def train(self) -> tf.data.Dataset:
        return self.make_dataset(self.train_data, shuffle=self.shuffle)

    @property
    def val(self) -> tf.data.Dataset:
        return self.make_dataset(self.val_data, shuffle=False)

    @property
    def test(self) -> tf.data.Dataset:
        return self.make_dataset(self.test_data, shuffle=False)

    @property
    def example(self) -> tuple[tf.Tensor, tf.Tensor]:
        if self._example is None:
            self._example = next(iter(self.train))
        return self._example


class AutoregressiveLSTM(tf.keras.Model):
    def __init__(self, cfg: Config, out_steps: int) -> None:
        super().__init__()
        if cfg.bidirectional:
            raise ValueError("AutoregressiveLSTM does not support bidirectional=True.")
        if cfg.num_lstm_layers < 1:
            raise ValueError("num_lstm_layers must be >= 1")

        if isinstance(cfg.lstm_units, int):
            lstm_units = [cfg.lstm_units] * cfg.num_lstm_layers
        else:
            lstm_units = list(cfg.lstm_units)
        if cfg.num_lstm_layers < len(lstm_units):
            lstm_units = lstm_units[: cfg.num_lstm_layers]
        if cfg.num_lstm_layers > len(lstm_units):
            lstm_units.extend(
                [lstm_units[-1]] * (cfg.num_lstm_layers - len(lstm_units))
            )

        reg = tf.keras.regularizers.l2(cfg.l2_reg) if cfg.l2_reg > 0 else None
        self.encoder_layers = []
        for i, units in enumerate(lstm_units):
            is_last = i == len(lstm_units) - 1
            self.encoder_layers.append(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=not is_last,
                    return_state=is_last,
                    dropout=cfg.dropout,
                    recurrent_dropout=cfg.recurrent_dropout,
                    kernel_regularizer=reg,
                )
            )

        # Dropout in manual LSTMCell loops can trigger graph capture errors.
        self.decoder_cell = tf.keras.layers.LSTMCell(
            lstm_units[-1],
            dropout=0.0,
            recurrent_dropout=0.0,
            kernel_regularizer=reg,
        )

        dense_layers = []
        dense_units = []
        if isinstance(cfg.dense_units, int):
            dense_units = [cfg.dense_units]
        elif cfg.dense_units:
            dense_units = list(cfg.dense_units)
        for units in dense_units:
            dense_layers.append(
                tf.keras.layers.Dense(units, activation=cfg.dense_activation)
            )
            if cfg.dense_dropout > 0:
                dense_layers.append(tf.keras.layers.Dropout(cfg.dense_dropout))
        self.projection = (
            tf.keras.Sequential(dense_layers, name="projection")
            if dense_layers
            else None
        )
        self.out_steps = out_steps
        self.output_layer = tf.keras.layers.Dense(
            1, activation=cfg.output_activation, dtype="float32"
        )

    def _project(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.projection is not None:
            x = self.projection(x, training=training)
        return self.output_layer(x)

    def warmup(
        self, inputs: tf.Tensor, training: bool | None = None
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        x = inputs
        for layer in self.encoder_layers[:-1]:
            x = layer(x, training=training)
        x, h, c = self.encoder_layers[-1](x, training=training)
        prediction = self._project(x, training=training)
        return prediction, [h, c]

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        predictions = []
        future_covariates = None
        if isinstance(inputs, (tuple, list)):
            inputs, future_covariates = inputs
        prediction, state = self.warmup(inputs, training=training)
        predictions.append(prediction)

        if future_covariates is None:
            future_covariates = tf.zeros(
                [tf.shape(prediction)[0], self.out_steps, 0], dtype=prediction.dtype
            )

        for step in range(1, self.out_steps):
            step_cov = future_covariates[:, step, :]
            decoder_input = tf.concat([prediction, step_cov], axis=-1)
            x, state = self.decoder_cell(decoder_input, states=state, training=training)
            prediction = self._project(x, training=training)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        return tf.transpose(predictions, [1, 0, 2])


def configure_runtime(cfg: Config) -> None:
    tf.keras.utils.set_random_seed(cfg.seed)
    if cfg.use_xla:
        tf.config.optimizer.set_jit(True)
    if cfg.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if cfg.enable_gpu_memory_growth:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def load_data(csv_path: str, datetime_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df[datetime_col] = pd.to_datetime(df[datetime_col], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(datetime_col).set_index(datetime_col)
    return df


def split_by_date(cfg: Config, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "train": df.loc[cfg.train_start : cfg.train_end].copy(),
        "test": df.loc[cfg.test_start : cfg.test_end].copy(),
        "val": df.loc[cfg.val_start : cfg.val_end].copy(),
    }


def build_preprocessor(cfg: Config) -> AirQualityPreprocessor:
    return AirQualityPreprocessor(
        target_col=cfg.target_col,
        auxiliary_stations=cfg.auxiliary_stations,
        quality_col=None,
        datetime_col=cfg.datetime_col,
        freq=cfg.freq,
        horizon=0,
        time_features_at=cfg.time_features_at,
        lag_other_cols=cfg.lag_other_cols,
        missing_col_threshold=cfg.missing_col_threshold,
        rolling_windows=cfg.rolling_windows,
        lags=cfg.lags,
        use_cyclical_time=cfg.use_cyclical_time,
    )


def prepare_split(
    preproc: AirQualityPreprocessor, df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray]:
    df_feat, y, valid_mask, _ = preproc.transform(df)
    return df_feat, valid_mask


def impute_and_scale_all(
    cfg: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    imputer = SimpleImputer(strategy=cfg.impute_strategy)
    train_imp = imputer.fit_transform(train_df).astype(np.float32)
    test_imp = imputer.transform(test_df).astype(np.float32)
    val_imp = imputer.transform(val_df).astype(np.float32)

    scaler = None
    if cfg.scale_features or cfg.scale_target:
        # Keep inputs and labels on the same scale for autoregressive decoding.
        scaler = StandardScaler()
        train_imp = scaler.fit_transform(train_imp).astype(np.float32)
        test_imp = scaler.transform(test_imp).astype(np.float32)
        val_imp = scaler.transform(val_imp).astype(np.float32)

    return train_imp, test_imp, val_imp, scaler


def append_valid_mask(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    mask_col = valid_mask.astype(np.float32).reshape(-1, 1)
    return np.concatenate([data, mask_col], axis=1)


def build_seasonal_profiles(
    train_series: pd.Series, freq: str | None, use_month: bool
) -> tuple[pd.Series, pd.Series | None, float]:
    series = train_series.copy()
    if freq is not None:
        series = series.asfreq(freq)
    idx = series.index
    how = idx.dayofweek * 24 + idx.hour
    temp = pd.DataFrame({"y": series.values, "how": how}, index=idx)
    overall = temp["y"].mean()

    how_mean = temp.groupby("how")["y"].mean().reindex(range(168))
    how_mean = how_mean.interpolate(limit_direction="both")

    month_mean = None
    if use_month:
        month = idx.month
        temp["month"] = month
        month_mean = temp.groupby("month")["y"].mean().reindex(range(1, 13))
        month_mean = month_mean.interpolate(limit_direction="both")

    return how_mean, month_mean, overall


def seasonal_component(
    index: pd.DatetimeIndex,
    how_mean: pd.Series,
    month_mean: pd.Series | None,
    overall: float,
    use_month: bool,
) -> np.ndarray:
    how = (index.dayofweek * 24 + index.hour).to_numpy()
    comp = how_mean.values[how]
    if use_month and month_mean is not None:
        month = index.month.to_numpy()
        comp = comp + month_mean.values[month - 1] - overall
    return comp


def apply_seasonal_baseline(
    df: pd.DataFrame,
    target_col: str,
    how_mean: pd.Series,
    month_mean: pd.Series | None,
    overall: float,
    freq: str | None,
    use_month: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    if freq is not None:
        df = df.asfreq(freq)
    target = df[target_col].replace(-9999, np.nan).astype(float)
    comp = seasonal_component(df.index, how_mean, month_mean, overall, use_month)
    df[target_col] = target - comp
    return df, comp


def inverse_scale_target(
    arr: np.ndarray, scaler: StandardScaler | None, target_index: int
) -> np.ndarray:
    if scaler is None:
        return arr
    return arr * scaler.scale_[target_index] + scaler.mean_[target_index]


def dataset_to_numpy(
    ds: tf.data.Dataset,
) -> Tuple[np.ndarray | Tuple[np.ndarray, np.ndarray], np.ndarray]:
    xs = []
    futures = []
    ys = []
    for batch_x, batch_y in ds:
        if isinstance(batch_x, (tuple, list)):
            batch_inputs, batch_future = batch_x
            xs.append(batch_inputs.numpy())
            futures.append(batch_future.numpy())
        else:
            xs.append(batch_x.numpy())
        ys.append(batch_y.numpy())
    if not xs:
        return np.empty((0,)), np.empty((0,))
    x_out = np.concatenate(xs, axis=0)
    if futures:
        future_out = np.concatenate(futures, axis=0)
        return (x_out, future_out), np.concatenate(ys, axis=0)
    return x_out, np.concatenate(ys, axis=0)


def count_valid_windows(
    mask: np.ndarray, total_window_size: int, label_start: int
) -> int:
    if len(mask) < total_window_size:
        return 0
    windows = np.lib.stride_tricks.sliding_window_view(mask, total_window_size)
    label_windows = windows[:, label_start:]
    return int(np.sum(np.all(label_windows, axis=1)))


def build_label_seasonal_windows(
    component: np.ndarray,
    valid_mask: np.ndarray,
    total_window_size: int,
    label_start: int,
) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(component, total_window_size)
    label_windows = windows[:, label_start:]
    mask_windows = np.lib.stride_tricks.sliding_window_view(
        valid_mask, total_window_size
    )
    label_mask = mask_windows[:, label_start:]
    valid = np.all(label_mask, axis=1)
    return label_windows[valid][:, :, None]


def build_model(input_shape: Tuple[int, int], cfg: Config) -> tf.keras.Model:
    return AutoregressiveLSTM(cfg, out_steps=cfg.out_steps)


def compile_model(model: tf.keras.Model, cfg: Config) -> None:
    if cfg.optimizer.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    model.compile(
        optimizer=optimizer,
        loss=cfg.loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), R2Score()],
    )


def build_callbacks(cfg: Config, model_path: Path) -> list:
    callbacks = []
    if cfg.use_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=cfg.best_monitor,
                mode=cfg.best_mode,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                restore_best_weights=True,
            )
        )
    if cfg.use_reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=cfg.best_monitor,
                mode=cfg.best_mode,
                patience=cfg.reduce_lr_patience,
                factor=cfg.reduce_lr_factor,
                min_lr=cfg.reduce_lr_min_lr,
                verbose=1,
            )
        )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=cfg.best_monitor,
            mode=cfg.best_mode,
            save_best_only=True,
            save_weights_only=True,
        )
    )
    return callbacks


def evaluate_model(
    model: tf.keras.Model,
    X: np.ndarray | Tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    scaler: StandardScaler | None,
    target_index: int,
    seasonal_component: np.ndarray | None = None,
    batch_size: int | None = None,
) -> dict:
    preds = model.predict(X, verbose=0, batch_size=batch_size)
    y_true = inverse_scale_target(y, scaler, target_index)
    preds = inverse_scale_target(preds, scaler, target_index)
    if seasonal_component is not None:
        comp = seasonal_component
        if comp.ndim == 2:
            comp = comp[:, :, None]
        y_true = y_true + comp
        preds = preds + comp
    y_true_flat = y_true.reshape(-1)
    preds_flat = preds.reshape(-1)
    rmse = root_mean_squared_error(y_true_flat, preds_flat)
    r2 = r2_score(y_true_flat, preds_flat)
    return {"rmse": float(rmse), "r2": float(r2)}


def permutation_importance_seq(
    model: tf.keras.Model,
    X: np.ndarray | Tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    scaler: StandardScaler | None,
    target_index: int,
    seasonal_component: np.ndarray | None,
    feature_names: list,
    metric: str,
    n_repeats: int,
    max_features: int | None,
    seed: int,
    batch_size: int | None = None,
) -> Tuple[list, dict]:
    if metric not in {"r2", "rmse"}:
        raise ValueError("feature_importance_metric must be 'r2' or 'rmse'")

    if isinstance(X, (tuple, list)):
        X_base = X[0]
        X_future = X[1]
    else:
        X_base = X
        X_future = None

    num_features = X_base.shape[2]
    if len(feature_names) != num_features:
        if len(feature_names) > num_features:
            feature_names = feature_names[:num_features]
        else:
            feature_names = feature_names + [
                f"feature_{i}" for i in range(len(feature_names), num_features)
            ]

    baseline = evaluate_model(
        model,
        X,
        y,
        scaler,
        target_index,
        seasonal_component=seasonal_component,
        batch_size=batch_size,
    )
    baseline_value = baseline[metric]

    indices = np.arange(num_features)
    truncated = False
    if max_features is not None and len(indices) > max_features:
        indices = indices[:max_features]
        truncated = True

    rng = np.random.default_rng(seed)
    X_perm = X_base.copy()
    results = []

    for idx in indices:
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X_perm))
            original = X_perm[:, :, idx].copy()
            X_perm[:, :, idx] = X_perm[perm, :, idx]
            X_eval = (X_perm, X_future) if X_future is not None else X_perm
            shuffled = evaluate_model(
                model,
                X_eval,
                y,
                scaler,
                target_index,
                seasonal_component=seasonal_component,
                batch_size=batch_size,
            )
            if metric == "r2":
                delta = baseline_value - shuffled["r2"]
            else:
                delta = shuffled["rmse"] - baseline_value
            deltas.append(delta)
            X_perm[:, :, idx] = original
        results.append(
            {
                "feature": feature_names[idx],
                "importance": float(np.mean(deltas)),
                "std": float(np.std(deltas)) if n_repeats > 1 else 0.0,
            }
        )

    results = sorted(results, key=lambda x: x["importance"], reverse=True)
    meta = {
        "metric": metric,
        "baseline": baseline_value,
        "n_repeats": n_repeats,
        "truncated": truncated,
        "num_features": num_features,
        "num_used": len(indices),
    }
    return results, meta


def plot_history(history: tf.keras.callbacks.History, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance(
    results: list, meta: dict, json_path: Path, csv_path: Path
) -> None:
    payload = {"meta": meta, "results": results}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)


def log_mlflow(
    cfg: Config,
    history: tf.keras.callbacks.History,
    metrics: dict,
    artifacts_dir: Path,
    model_path: Path,
    model: tf.keras.Model | None = None,
    extra_params: dict | None = None,
    extra_metrics: dict | None = None,
    extra_artifacts: list | None = None,
) -> None:
    if not cfg.use_mlflow:
        return
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is not installed. Please `pip install mlflow`."
        ) from exc

    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        for key, value in asdict(cfg).items():
            mlflow.log_param(key, value)
        if extra_params:
            for key, value in extra_params.items():
                mlflow.log_param(key, value)

        for split, values in metrics.items():
            mlflow.log_metric(f"{split}_rmse", values["rmse"])
            mlflow.log_metric(f"{split}_r2", values["r2"])
        if extra_metrics:
            for key, value in extra_metrics.items():
                mlflow.log_metric(key, value)

        monitor_key = cfg.best_monitor
        if monitor_key in history.history:
            values = history.history[monitor_key]
            if cfg.best_mode == "min":
                best_epoch = int(np.argmin(values))
            else:
                best_epoch = int(np.argmax(values))
            mlflow.log_metric("best_epoch", best_epoch)

        history_path = artifacts_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)
        mlflow.log_artifact(str(history_path))

        plot_path = artifacts_dir / "loss_curve.png"
        plot_history(history, plot_path)
        mlflow.log_artifact(str(plot_path))

        if model is not None:
            summary_path = artifacts_dir / "model_summary.txt"
            with summary_path.open("w", encoding="utf-8") as f:
                model.summary(print_fn=lambda line: f.write(line + "\n"))
            mlflow.log_artifact(str(summary_path))

        if model_path.exists():
            mlflow.log_artifact(str(model_path))

        if extra_artifacts:
            for artifact in extra_artifacts:
                if artifact and os.path.exists(artifact):
                    mlflow.log_artifact(artifact)


def main() -> None:
    cfg = Config()
    configure_runtime(cfg)
    if cfg.seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if cfg.out_steps < 1:
        raise ValueError("out_steps must be >= 1")

    df = load_data(cfg.csv_path, cfg.datetime_col)
    splits = split_by_date(cfg, df)

    seasonal_components = None
    if cfg.use_seasonal_baseline:
        train_target = (
            splits["train"][cfg.target_col].replace(-9999, np.nan).astype(float)
        )
        how_mean, month_mean, overall = build_seasonal_profiles(
            train_target, cfg.freq, cfg.seasonal_use_month
        )
        seasonal_components = {}
        for split_name, split_df in splits.items():
            adjusted, comp = apply_seasonal_baseline(
                split_df,
                cfg.target_col,
                how_mean,
                month_mean,
                overall,
                cfg.freq,
                cfg.seasonal_use_month,
            )
            splits[split_name] = adjusted
            seasonal_components[split_name] = comp

    preproc = build_preprocessor(cfg)
    preproc.fit(splits["train"])

    train_df, mask_train = prepare_split(preproc, splits["train"])
    test_df, mask_test = prepare_split(preproc, splits["test"])
    val_df, mask_val = prepare_split(preproc, splits["val"])

    seasonal_labels = {}
    if cfg.use_seasonal_baseline and seasonal_components is not None:
        seasonal_labels["train"] = build_label_seasonal_windows(
            seasonal_components["train"],
            mask_train,
            cfg.seq_len + cfg.out_steps,
            cfg.seq_len,
        )
        seasonal_labels["test"] = build_label_seasonal_windows(
            seasonal_components["test"],
            mask_test,
            cfg.seq_len + cfg.out_steps,
            cfg.seq_len,
        )
        seasonal_labels["val"] = build_label_seasonal_windows(
            seasonal_components["val"],
            mask_val,
            cfg.seq_len + cfg.out_steps,
            cfg.seq_len,
        )
    else:
        seasonal_labels = {"train": None, "test": None, "val": None}

    feature_names = list(train_df.columns)
    if cfg.target_col not in feature_names:
        raise ValueError(f"Target column '{cfg.target_col}' not found in features.")
    target_index = feature_names.index(cfg.target_col)
    time_feature_columns = [c for c in TIME_FEATURE_COLUMNS if c in feature_names]

    train_scaled, test_scaled, val_scaled, scaler = impute_and_scale_all(
        cfg,
        train_df,
        test_df,
        val_df,
    )

    train_data = append_valid_mask(train_scaled, mask_train)
    test_data = append_valid_mask(test_scaled, mask_test)
    val_data = append_valid_mask(val_scaled, mask_val)

    if len(train_data) < cfg.seq_len + cfg.out_steps:
        raise ValueError(
            "Not enough rows to build windows. "
            f"Required >= {cfg.seq_len + cfg.out_steps}, got {len(train_data)}."
        )
    for split_name, split_data in (("test", test_data), ("val", val_data)):
        if len(split_data) < cfg.seq_len + cfg.out_steps:
            raise ValueError(
                f"Not enough rows in {split_name} split to build windows. "
                f"Required >= {cfg.seq_len + cfg.out_steps}, got {len(split_data)}."
            )

    column_names = feature_names + [VALID_MASK_COLUMN]
    window = WindowGenerator(
        input_width=cfg.seq_len,
        label_width=cfg.out_steps,
        shift=cfg.out_steps,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        column_names=column_names,
        label_columns=[cfg.target_col],
        time_feature_columns=time_feature_columns,
        batch_size=cfg.batch_size,
        shuffle_buffer=cfg.shuffle_buffer,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
        mask_column=VALID_MASK_COLUMN,
    )

    input_shape = (cfg.seq_len, window.num_features)
    model = build_model(input_shape, cfg)
    compile_model(model, cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"lstm_best_{cfg.target_col.replace('__', '_')}.weights.h5"

    callbacks = build_callbacks(cfg, model_path)
    start_time = time.time()
    if cfg.use_tf_dataset:
        train_steps = math.ceil(
            count_valid_windows(
                mask_train, window.total_window_size, window.label_start
            )
            / cfg.batch_size
        )
        val_steps = math.ceil(
            count_valid_windows(mask_val, window.total_window_size, window.label_start)
            / cfg.batch_size
        )
        if train_steps < 1 or val_steps < 1:
            raise ValueError("Not enough valid windows to train or validate.")
        history = model.fit(
            window.train.repeat(),
            validation_data=window.val.repeat(),
            epochs=cfg.epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        train_eval_ds = window.make_dataset(window.train_data, shuffle=False)
        val_eval_ds = window.make_dataset(window.val_data, shuffle=False)
        X_train_seq, y_train_seq = dataset_to_numpy(train_eval_ds)
        X_val_seq, y_val_seq = dataset_to_numpy(val_eval_ds)
        history = model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=cfg.shuffle,
        )
    train_time = time.time() - start_time

    if model_path.exists():
        model.load_weights(model_path)

    train_eval_ds = window.make_dataset(window.train_data, shuffle=False)
    test_eval_ds = window.make_dataset(window.test_data, shuffle=False)
    val_eval_ds = window.make_dataset(window.val_data, shuffle=False)

    X_train_seq, y_train_seq = dataset_to_numpy(train_eval_ds)
    X_test_seq, y_test_seq = dataset_to_numpy(test_eval_ds)
    X_val_seq, y_val_seq = dataset_to_numpy(val_eval_ds)

    num_features = window.num_features
    if isinstance(X_train_seq, (tuple, list)):
        num_features = X_train_seq[0].shape[-1]

    metrics = {
        "train": evaluate_model(
            model,
            X_train_seq,
            y_train_seq,
            scaler,
            target_index,
            seasonal_component=seasonal_labels["train"],
            batch_size=cfg.batch_size,
        ),
        "test": evaluate_model(
            model,
            X_test_seq,
            y_test_seq,
            scaler,
            target_index,
            seasonal_component=seasonal_labels["test"],
            batch_size=cfg.batch_size,
        ),
        "val": evaluate_model(
            model,
            X_val_seq,
            y_val_seq,
            scaler,
            target_index,
            seasonal_component=seasonal_labels["val"],
            batch_size=cfg.batch_size,
        ),
    }

    results = {
        "metrics": metrics,
        "train_time_sec": float(train_time),
        "num_features": int(num_features),
        "num_train_samples": int(len(y_train_seq)),
        "num_test_samples": int(len(y_test_seq)),
        "num_val_samples": int(len(y_val_seq)),
        "feature_names": feature_names,
        "config": asdict(cfg),
    }

    results_path = output_dir / f"lstm_results_{cfg.target_col.replace('__', '_')}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    artifacts_dir = output_dir / f"artifacts_{cfg.target_col.replace('__', '_')}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    feature_importance_info = None
    extra_artifacts = []
    if cfg.compute_feature_importance:
        split_name = cfg.feature_importance_split.lower()
        split_map = {
            "train": (X_train_seq, y_train_seq),
            "test": (X_test_seq, y_test_seq),
            "val": (X_val_seq, y_val_seq),
        }
        if split_name not in split_map:
            raise ValueError("feature_importance_split must be train/test/val")
        X_imp, y_imp = split_map[split_name]
        fi_results, fi_meta = permutation_importance_seq(
            model=model,
            X=X_imp,
            y=y_imp,
            scaler=scaler,
            target_index=target_index,
            seasonal_component=seasonal_labels[split_name],
            feature_names=feature_names,
            metric=cfg.feature_importance_metric,
            n_repeats=cfg.feature_importance_n_repeats,
            max_features=cfg.feature_importance_max_features,
            seed=cfg.feature_importance_seed,
            batch_size=cfg.batch_size,
        )
        fi_json = artifacts_dir / f"feature_importance_{split_name}.json"
        fi_csv = artifacts_dir / f"feature_importance_{split_name}.csv"
        save_feature_importance(fi_results, fi_meta, fi_json, fi_csv)
        feature_importance_info = {
            "path_json": str(fi_json),
            "path_csv": str(fi_csv),
        }
        extra_artifacts.extend([str(fi_json), str(fi_csv)])

    extra_params = {
        "num_features": int(num_features),
        "num_train_samples": int(len(y_train_seq)),
        "num_test_samples": int(len(y_test_seq)),
        "num_val_samples": int(len(y_val_seq)),
    }
    extra_metrics = {"train_time_sec": float(train_time)}

    log_mlflow(
        cfg,
        history,
        metrics,
        artifacts_dir,
        model_path,
        model=model,
        extra_params=extra_params,
        extra_metrics=extra_metrics,
        extra_artifacts=extra_artifacts,
    )

    if feature_importance_info:
        results["feature_importance"] = feature_importance_info

    print("Training complete.")
    print(f"Results saved to {results_path}")
    print(metrics)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
