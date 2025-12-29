import json
import time

import numpy as np
import pandas as pd

from cuml.ensemble import RandomForestRegressor
from cuml.preprocessing import SimpleImputer as cuSimpleImputer

try:
    import cupy as cp
except ImportError:
    cp = None

from sklearn.impute import SimpleImputer as SkSimpleImputer

from air_quality_preprocessor_forecasting import AirQualityPreprocessor

USE_GPU = True
USE_RMM_POOL = True
USE_FAST_IMPORTANCE = True
RUN_FULL_MODEL = False
DTYPE = np.float32

TOP_N_FEATURES = 20
TOP_K_FEATURES = 5

RF_IMPORTANCE_PARAMS = {
    "n_estimators": 200,
    "max_depth": 50,
    "random_state": 42,
    "criterion": "squared_error",
}

RF_FULL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 150,
    "random_state": 42,
    "criterion": "squared_error",
}

RF_TOP5_PARAMS = dict(RF_FULL_PARAMS)

if USE_GPU and cp is None:
    print("cupy is not available; preprocessing will stay on CPU.")
    USE_GPU = False


def _init_rmm_pool():
    if not USE_GPU or not USE_RMM_POOL:
        return False
    try:
        import rmm

        rmm.reinitialize(pool_allocator=True)
        if hasattr(rmm, "mr") and hasattr(rmm.mr, "PoolMemoryResource"):
            device_mr = rmm.mr.CudaMemoryResource()
            pool_mr = rmm.mr.PoolMemoryResource(device_mr)
            rmm.mr.set_current_device_resource(pool_mr)
        elif hasattr(rmm, "rmm_cupy_allocator"):
            cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
        print("RMM pool enabled.")
        return True
    except Exception as exc:
        print(f"RMM pool not enabled: {exc}")
        return False


def _to_device_array(x, xp):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy(dtype=DTYPE, copy=False)
    else:
        x = np.asarray(x, dtype=DTYPE)
    return xp.asarray(x) if xp is cp else x


def _to_device_vector(x, xp):
    x = np.asarray(x, dtype=DTYPE)
    return xp.asarray(x) if xp is cp else x


def _to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _rmse(y_true, y_pred, xp):
    diff = y_true - y_pred
    return xp.sqrt(xp.mean(diff * diff))


def _r2(y_true, y_pred, xp):
    ss_res = xp.sum((y_true - y_pred) ** 2)
    ss_tot = xp.sum((y_true - xp.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


_init_rmm_pool()

xp = cp if USE_GPU else np
imputer_cls = cuSimpleImputer if USE_GPU else SkSimpleImputer

df = pd.read_csv("siata_merged_data.csv", sep=";")
df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], format="%Y-%m-%d %H:%M:%S")
df = df.sort_values("Fecha_Hora").set_index("Fecha_Hora")

train_start = "2019-01-01"
train_end = "2021-12-31"

test_start = "2022-01-01"
test_end = "2023-12-31"

val_start = "2024-01-01"
val_end = "2024-12-31"

df_train = df.loc[train_start:train_end].copy()
df_test = df.loc[test_start:test_end].copy()
df_val = df.loc[val_start:val_end].copy()

pm25_columns = [col for col in df.columns if "__PM2.5" in col]

for col in pm25_columns:
    preproc = AirQualityPreprocessor(
        target_col=col,  # or any other station column
        quality_col=None,  # <-- no quality column in this dataset
        datetime_col="Fecha_Hora",  # the time column in your file
        freq="1H",  # or "1H" if you want to enforce hourly frequency
        horizon=0,  # pm25_lag1..24 and lags for other numeric vars
        lag_other_cols=None,  # None = lag all numeric columns except target
        missing_col_threshold=0.5,  # or e.g. 0.5 to drop very sparse columns
        rolling_windows=None,
        lags=None,
        use_cyclical_time=False,
    )

    # VERY IMPORTANT: fit ONLY on training data (2019–2021)
    preproc.fit(df_train)

    # Train
    X_train, y_train, mask_train_valid, idx_train = preproc.transform(df_train)
    X_train = X_train[mask_train_valid]
    y_train = y_train[mask_train_valid]

    # Test (2022–2023)
    X_test, y_test, mask_test_valid, idx_test = preproc.transform(df_test)
    X_test = X_test[mask_test_valid]
    y_test = y_test[mask_test_valid]

    # Validation (2024)
    X_val, y_val, mask_val_valid, idx_val = preproc.transform(df_val)
    X_val = X_val[mask_val_valid]
    y_val = y_val[mask_val_valid]

    features = preproc.feature_cols_
    X_train = X_train[features]
    X_test = X_test[features]
    X_val = X_val[features]

    # Check if data has NaNs
    print(f"NaN in X_train: {pd.isna(X_train).sum().sum()}")
    print(f"NaN in X_test: {pd.isna(X_test).sum().sum()}")
    print(f"NaN in X_val: {pd.isna(X_val).sum().sum()}")

    # Fit imputer on training data only
    imputer = imputer_cls(strategy="median")
    X_train_device = _to_device_array(X_train, xp)
    X_test_device = _to_device_array(X_test, xp)
    X_val_device = _to_device_array(X_val, xp)

    X_train_clean = imputer.fit_transform(X_train_device)
    X_test_clean = imputer.transform(X_test_device)
    X_val_clean = imputer.transform(X_val_device)

    y_train_device = _to_device_vector(y_train, xp).ravel()
    y_test_device = _to_device_vector(y_test, xp).ravel()
    y_val_device = _to_device_vector(y_val, xp).ravel()

    print(
        f"\nData shapes: X_train={X_train_clean.shape}, y_train={y_train_device.shape}"
    )

    # Train on 2019–2021 with timing
    print("\n🚀 Training RandomForest...")
    start_time = time.time()

    if USE_FAST_IMPORTANCE:
        print("Training importance model (fast)...")
        rf_importance = RandomForestRegressor(**RF_IMPORTANCE_PARAMS)
        rf_importance.fit(X_train_clean, y_train_device)
        importance_model = rf_importance
        if RUN_FULL_MODEL:
            print("Training full model...")
            rf_full = RandomForestRegressor(**RF_FULL_PARAMS)
            rf_full.fit(X_train_clean, y_train_device)
        else:
            rf_full = rf_importance
    else:
        rf_full = RandomForestRegressor(**RF_FULL_PARAMS)
        rf_full.fit(X_train_clean, y_train_device)
        importance_model = rf_full

    train_time = time.time() - start_time

    print(f"✓ Training completed for {col} in {train_time:.2f} seconds")

    y_train_pred = rf_full.predict(X_train_clean)
    rmse_train = _to_float(_rmse(y_train_device, y_train_pred, xp))
    y_test_pred = rf_full.predict(X_test_clean)

    rmse_test = _to_float(_rmse(y_test_device, y_test_pred, xp))

    # Evaluate on 2024
    y_val_pred = rf_full.predict(X_val_clean)
    rmse_val = _to_float(_rmse(y_val_device, y_val_pred, xp))

    # Evaluate R2
    r2_train = _to_float(_r2(y_train_device, y_train_pred, xp))
    r2_test = _to_float(_r2(y_test_device, y_test_pred, xp))
    r2_val = _to_float(_r2(y_val_device, y_val_pred, xp))

    print("\n📊 Results:")
    print(f"RMSE test (2022–2023): {rmse_test:.4f}")
    print(f"RMSE val  (2024): {rmse_val:.4f}")
    print(f"R2 test (2022–2023): {r2_test:.4f}")

    print(f"R2 val  (2024): {r2_val:.4f}")

    print(f"\n🔍 Model type: {type(rf_full).__module__}.{type(rf_full).__name__}")

    # Get importances and feature names
    importances = _to_numpy(importance_model.feature_importances_)
    feature_names = preproc.feature_cols_

    # DEBUG: Check for mismatch
    print(f"🔍 Debugging info:")
    print(f"  Number of features in model: {len(importances)}")
    print(f"  Number of feature names: {len(feature_names)}")
    print(f"  X_train shape: {X_train_clean.shape}")

    # Ensure arrays match
    if len(importances) != len(feature_names):
        print("⚠️ WARNING: Mismatch detected!")
        # Use the smaller length to be safe
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]
        print(f"✓ Adjusted to {min_len} features")

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    # Get top N features
    n_top = min(TOP_N_FEATURES, len(importances))
    top_idx = indices[:n_top]

    # Create plot with error handling
    try:
        top_importances = importances[top_idx]
        top_features = [feature_names[i] for i in top_idx]

    except IndexError as e:
        print(f"❌ Error: {e}")
        print(f"Max index in top_idx: {max(top_idx)}")
        print(f"Feature names length: {len(feature_names)}")

    five_most_important_features = [feature_names[i] for i in top_idx[:TOP_K_FEATURES]]

    print(f"\nTop {n_top} important features for {col}: {five_most_important_features}")

    # Retrain with top 5 features only
    top5_idx = top_idx[:TOP_K_FEATURES]
    top5_idx_device = xp.asarray(top5_idx) if xp is cp else top5_idx
    X_train_top5 = X_train_clean[:, top5_idx_device]
    X_test_top5 = X_test_clean[:, top5_idx_device]
    X_val_top5 = X_val_clean[:, top5_idx_device]

    rf_top5 = RandomForestRegressor(**RF_TOP5_PARAMS)

    rf_top5.fit(X_train_top5, y_train_device)

    y_train_top5_pred = rf_top5.predict(X_train_top5)
    rmse_train_top5 = _to_float(_rmse(y_train_device, y_train_top5_pred, xp))
    y_test_top5_pred = rf_top5.predict(X_test_top5)
    rmse_test_top5 = _to_float(_rmse(y_test_device, y_test_top5_pred, xp))
    y_val_top5_pred = rf_top5.predict(X_val_top5)
    rmse_val_top5 = _to_float(_rmse(y_val_device, y_val_top5_pred, xp))
    r2_train_top5 = _to_float(_r2(y_train_device, y_train_top5_pred, xp))
    r2_test_top5 = _to_float(_r2(y_test_device, y_test_top5_pred, xp))
    r2_val_top5 = _to_float(_r2(y_val_device, y_val_top5_pred, xp))

    # Save results to JSON
    results = {
        "full_model": {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "rmse_val": rmse_val,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "r2_val": r2_val,
        },
        "top5_model": {
            "rmse_train": rmse_train_top5,
            "rmse_test": rmse_test_top5,
            "rmse_val": rmse_val_top5,
            "r2_train": r2_train_top5,
            "r2_test": r2_test_top5,
            "r2_val": r2_val_top5,
        },
        "top_5_features": five_most_important_features,
        "feature_importances": {
            feature_names[i]: float(importances[i]) for i in range(len(feature_names))
        },
    }
    with open(
        f"feature_selection_results/rf_results_{col.replace('__', '_')}.json", "w"
    ) as f:
        json.dump(results, f, indent=4)
    print(
        f"✓ Results saved to feature_selection_results/rf_results_{col.replace('__', '_')}.json"
    )
