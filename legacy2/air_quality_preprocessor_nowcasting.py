import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def _infer_feature_columns(df, target_col):
    """Split columns into feature / numeric / categorical, excluding target."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


class AirQualityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Generic preprocessor for wide air-quality tables (time × stations).

    - `target_col` is the station you want to model (e.g. 'BAR-TORR').
    - `quality_col` is optional; use None when you don't have a quality column.
    - `datetime_col` is the time column (string in the raw CSV).
    - Adds time features, lag features and basic imputation, then builds an
      encoded feature matrix suitable for RF / MLP / LSTM / etc.

    transform() returns:
      X: 2D np.ndarray  (features, no NaNs)
      y: 1D np.ndarray  (target, NaN where missing)
      valid_mask: bool mask where y is not NaN
      index: DatetimeIndex aligned with X/y
    """

    def __init__(
        self,
        target_col="pm25",
        quality_col=None,
        datetime_col="timestamp",
        freq=None,
        use_cyclical_time=True,
        max_lag=24,
        lag_other_cols=None,
        missing_col_threshold=None,
        scale_numeric=True,
    ):
        self.target_col = target_col
        self.quality_col = quality_col
        self.datetime_col = datetime_col
        self.freq = freq
        self.use_cyclical_time = use_cyclical_time
        self.max_lag = max_lag
        self.lag_other_cols = lag_other_cols
        self.missing_col_threshold = missing_col_threshold
        self.scale_numeric = scale_numeric

        # set during fit()
        self.base_columns_ = None
        self.lag_base_cols_ = None
        self.feature_cols_ = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.numeric_medians_ = None
        self.column_transformer_ = None

    # ---------- core helpers ----------

    def _normalize_target_and_time(self, df):
        """Parse time, sort, enforce freq, and mark target missing if -9999 or bad quality."""
        df = df.copy()

        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.sort_values(self.datetime_col).set_index(self.datetime_col)

        if self.freq is not None:
            df = df.asfreq(self.freq)

        # ensure target is float & normalize -9999
        df[self.target_col] = df[self.target_col].astype(float)
        df.loc[df[self.target_col] == -9999, self.target_col] = np.nan

        # optional quality column (not present in your CSV)
        if self.quality_col is not None and self.quality_col in df.columns:
            bad = df[self.quality_col] != 1
            df.loc[bad, self.target_col] = np.nan

        return df

    def _drop_high_missing_columns(self, df, fit_mode=True):
        """Optionally drop columns with too many missing values."""
        if self.missing_col_threshold is None:
            return df

        if fit_mode:
            cols_to_drop = set()
            cols = df.columns

            for col in cols:
                if col == self.target_col or col == self.quality_col:
                    continue
                s = df[col]
                if pd.api.types.is_numeric_dtype(s):
                    missing = s.isna() | (s == -9999)
                else:
                    missing = s.isna()

                if missing.mean() > self.missing_col_threshold:
                    cols_to_drop.add(col)

            self.base_columns_ = [c for c in cols if c not in cols_to_drop]
            return df[self.base_columns_]
        else:
            if self.base_columns_ is None:
                return df
            existing = [c for c in self.base_columns_ if c in df.columns]
            return df[existing]

    def _remove_quality_columns(self, df):
        """Drop any quality-like columns from the feature space."""
        df = df.copy()
        q_cols = [
            c
            for c in df.columns
            if c == self.quality_col
            or c.startswith("calidad_")
            or c.endswith("_calidad")
        ]
        return df.drop(columns=q_cols, errors="ignore")

    def _add_time_features(self, df):
        df = df.copy()
        idx = df.index

        df["hour"] = idx.hour
        df["dayofweek"] = idx.dayofweek
        df["month"] = idx.month
        df["dayofyear"] = idx.dayofyear

        if self.use_cyclical_time:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
            df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

        return df

    def _decide_lag_columns(self, df_noq):
        """Choose which base columns will receive lag features."""
        if self.max_lag is None or self.max_lag <= 0:
            self.lag_base_cols_ = []
            return

        numeric_base = [
            c for c in df_noq.columns if pd.api.types.is_numeric_dtype(df_noq[c])
        ]

        if self.lag_other_cols is None:
            # default: all numeric except the target
            other = [c for c in numeric_base if c != self.target_col]
        else:
            # user-specified subset
            other = [
                c
                for c in self.lag_other_cols
                if c in df_noq.columns and c != self.target_col
            ]

        self.lag_base_cols_ = [self.target_col] + other

    def _add_lag_features(self, df):
        """Add lag columns for each base col in self.lag_base_cols_."""
        df = df.copy()

        if not self.lag_base_cols_ or self.max_lag is None or self.max_lag <= 0:
            return df

        for base_col in self.lag_base_cols_:
            for lag in [1, 2, 3, 21, 22, 23, 24, 47, 48]:
                col_name = f"{base_col}_lag{lag}"
                df[col_name] = df[base_col].shift(lag)

        return df

    # ---------- sklearn API ----------

    def fit(self, df, y=None):
        # 1) normalize time + target
        df_norm = self._normalize_target_and_time(df)

        # 2) optional: drop very sparse columns
        df_base = self._drop_high_missing_columns(df_norm, fit_mode=True)

        # 3) remove quality columns from features (none in your CSV, but safe)
        df_noq = self._remove_quality_columns(df_base)

        # 4) decide which columns get lags (target + other numeric)
        self._decide_lag_columns(df_noq)

        # 5) add time features
        df_feat = self._add_time_features(df_noq)

        # 6) add lag features
        df_feat = self._add_lag_features(df_feat)

        # 7) infer feature list and types
        self.feature_cols_, self.numeric_cols_, self.categorical_cols_ = (
            _infer_feature_columns(df_feat, target_col=self.target_col)
        )

        # 8) compute medians for numeric imputation
        self.numeric_medians_ = df_feat[self.numeric_cols_].median()

        df_imputed = df_feat.copy()
        df_imputed[self.numeric_cols_] = df_imputed[self.numeric_cols_].ffill().bfill()
        for col in self.numeric_cols_:
            df_imputed[col] = df_imputed[col].fillna(self.numeric_medians_[col])

        # 9) build ColumnTransformer
        if self.scale_numeric:
            numeric_transformer = Pipeline([("scaler", StandardScaler())])
        else:
            numeric_transformer = "passthrough"

        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols_),
                ("cat", categorical_transformer, self.categorical_cols_),
            ]
        )

        self.column_transformer_.fit(df_imputed[self.feature_cols_])

        return self

    def transform(self, df):
        # 1) normalize time + target
        df_norm = self._normalize_target_and_time(df)

        # 2) apply same sparse-column filter
        df_base = self._drop_high_missing_columns(df_norm, fit_mode=False)

        # 3) remove quality columns
        df_noq = self._remove_quality_columns(df_base)

        # 4) add time + lag features
        df_feat = self._add_time_features(df_noq)
        df_feat = self._add_lag_features(df_feat)

        # 5) build y and validity mask
        y = df_feat[self.target_col].values.astype(float)
        valid_mask = ~np.isnan(y)

        # 6) impute numeric features
        df_feat[self.numeric_cols_] = df_feat[self.numeric_cols_].ffill().bfill()
        for col in self.numeric_cols_:
            df_feat[col] = df_feat[col].fillna(self.numeric_medians_[col])

        # 7) transform to X
        X = self.column_transformer_.transform(df_feat[self.feature_cols_])

        return X, y, valid_mask, df_feat.index
