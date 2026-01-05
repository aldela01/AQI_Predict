
from __future__ import annotations

import logging
import os
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import tensorflow as tf
except Exception:
    tf = None  # type: ignore


@dataclass(frozen=True)
class TFConfig:
    seed: int = 42
    enable_gpu_memory_growth: bool = True
    mixed_precision: bool = False
    xla: bool = False

    # threading (CPU)
    intra_op_threads: int = 0  # 0 = TF default
    inter_op_threads: int = 0  # 0 = TF default


def configure_tf_runtime(cfg: TFConfig) -> Dict[str, object]:
    """
    Configure TF runtime for reproducibility and performance.

    Returns info dict (e.g., gpu devices).
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Install it before running this script.")

    tf.keras.utils.set_random_seed(cfg.seed)

    if cfg.intra_op_threads and cfg.intra_op_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(cfg.intra_op_threads)
    if cfg.inter_op_threads and cfg.inter_op_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(cfg.inter_op_threads)

    if cfg.xla:
        tf.config.optimizer.set_jit(True)

    if cfg.enable_gpu_memory_growth:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    if cfg.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception as e:
            LOGGER.warning("Could not enable mixed precision: %s", e)

    gpus = tf.config.list_physical_devices("GPU")
    return {"gpus": [d.name for d in gpus], "num_gpus": len(gpus)}


class R2Metric(tf.keras.metrics.Metric):
    def __init__(self, name: str = "r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            y_true = y_true * sw
            y_pred = y_pred * sw
        err = y_true - y_pred
        self.sse.assign_add(tf.reduce_sum(tf.square(err)))
        mean = tf.reduce_mean(y_true)
        self.sst.assign_add(tf.reduce_sum(tf.square(y_true - mean)))

    def result(self):
        eps = tf.keras.backend.epsilon()
        return 1.0 - (self.sse / (self.sst + eps))

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


def build_mlp(
    n_numeric: int,
    n_stations: int,
    emb_dim: int = 8,
    hidden: Sequence[int] = (256, 256, 128),
    dropout: float = 0.1,
) -> "tf.keras.Model":
    if tf is None:
        raise RuntimeError("TensorFlow not available")

    x_num = tf.keras.Input(shape=(n_numeric,), name="x_num")
    st = tf.keras.Input(shape=(), dtype=tf.int32, name="station_idx")

    emb = tf.keras.layers.Embedding(input_dim=n_stations + 1, output_dim=emb_dim, name="station_emb")(st)
    emb = tf.keras.layers.Flatten()(emb)

    x = tf.keras.layers.Concatenate()([x_num, emb])
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inputs=[x_num, st], outputs=out, name="mlp_tabular")
    return model


def build_lstm(
    seq_len: int,
    n_stations: int,
    emb_dim: int = 8,
    aux_dim: int = 5,
    lstm_units: int = 128,
    lstm_layers: int = 2,
    dropout: float = 0.1,
) -> "tf.keras.Model":
    if tf is None:
        raise RuntimeError("TensorFlow not available")

    seq = tf.keras.Input(shape=(seq_len, 1), name="seq")
    aux = tf.keras.Input(shape=(aux_dim,), name="aux")
    st = tf.keras.Input(shape=(), dtype=tf.int32, name="station_idx")

    x = seq
    for i in range(lstm_layers):
        return_seq = i < (lstm_layers - 1)
        x = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=return_seq,
            dropout=dropout,
            recurrent_dropout=0.0,
        )(x)

    emb = tf.keras.layers.Embedding(input_dim=n_stations + 1, output_dim=emb_dim, name="station_emb")(st)
    emb = tf.keras.layers.Flatten()(emb)

    x = tf.keras.layers.Concatenate()([x, aux, emb])
    x = tf.keras.layers.Dense(lstm_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, dtype="float32")(x)
    return tf.keras.Model(inputs=[seq, aux, st], outputs=out, name="lstm_seq2one")


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :], dtype=tf.float32)

    def call(self, x):
        L = tf.shape(x)[1]
        pe = tf.cast(self.pe[:, :L, :], x.dtype)
        return x + pe


def _encoder_block(x, d_model: int, nhead: int, dropout: float, ff_mult: int = 4):
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead, dropout=dropout)(x, x)
    x = tf.keras.layers.Add()([x, attn_out])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ff = tf.keras.layers.Dense(ff_mult * d_model, activation="gelu")(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(d_model)(ff)
    x = tf.keras.layers.Add()([x, ff])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer(
    seq_len: int,
    n_stations: int,
    emb_dim: int = 8,
    aux_dim: int = 5,
    d_model: int = 128,
    nhead: int = 8,
    layers: int = 3,
    dropout: float = 0.1,
) -> "tf.keras.Model":
    if tf is None:
        raise RuntimeError("TensorFlow not available")

    seq = tf.keras.Input(shape=(seq_len, 1), name="seq")
    aux = tf.keras.Input(shape=(aux_dim,), name="aux")
    st = tf.keras.Input(shape=(), dtype=tf.int32, name="station_idx")

    x = tf.keras.layers.Dense(d_model)(seq)
    x = PositionalEncoding(d_model, max_len=seq_len)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    for _ in range(layers):
        x = _encoder_block(x, d_model=d_model, nhead=nhead, dropout=dropout)

    # take last token
    x = x[:, -1, :]

    emb = tf.keras.layers.Embedding(input_dim=n_stations + 1, output_dim=emb_dim, name="station_emb")(st)
    emb = tf.keras.layers.Flatten()(emb)

    x = tf.keras.layers.Concatenate()([x, aux, emb])
    x = tf.keras.layers.Dense(d_model, activation="gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, dtype="float32")(x)

    return tf.keras.Model(inputs=[seq, aux, st], outputs=out, name="transformer_seq2one")


def compile_and_fit(
    model: "tf.keras.Model",
    train_ds: "tf.data.Dataset",
    val_ds: "tf.data.Dataset",
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    use_reduce_lr: bool = True,
) -> Dict[str, float]:
    if tf is None:
        raise RuntimeError("TensorFlow not available")

    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), R2Metric()],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    if use_reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_rmse",
                mode="min",
                patience=max(1, patience // 2),
                factor=0.5,
                min_lr=1e-6,
                verbose=0,
            )
        )

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    # Return best val rmse seen
    v = hist.history.get("val_rmse", [])
    best = float(np.min(v)) if v else float("nan")
    return {"best_val_rmse": best}


def predict_numpy(model: "tf.keras.Model", ds: "tf.data.Dataset") -> np.ndarray:
    pred = model.predict(ds, verbose=0)
    return pred.reshape(-1).astype(float)
