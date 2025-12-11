import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def _infer_feature_columns(df, target_col):
    """Split into feature / numeric / categorical columns, excluding target."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


class AirQualityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Model-agnostic preprocessor for wide air-quality tables.

    - target_col: station to predict (e.g. 'BAR-TORR').
    - quality_col: optional (None for your current CSV).
    - datetime_col: time column in the raw df.
    - freq: optional frequency for asfreq (e.g. '1H').
    - max_lag: number of lags (1..max_lag) to add for target and other numeric cols.
    - lag_other_cols: list of columns to lag; None = all numeric except target.
    - missing_col_threshold: if not None, drop columns whose missing ratio > threshold.

    NOTE: This class DOES NOT impute or scale.
    It returns a feature DataFrame with NaNs, plus y, mask, and index.

    transform() returns:
      df_feat: pd.DataFrame with features (including lags, time features)
      y:       1D np.array with target values (NaN where missing)
      valid_mask: boolean array (True where y is not NaN)
      index:   pd.DatetimeIndex aligned with rows of df_feat/y
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
    ):
        self.target_col = target_col
        self.quality_col = quality_col
        self.datetime_col = datetime_col
        self.freq = freq
        self.use_cyclical_time = use_cyclical_time
        self.max_lag = max_lag
        self.lag_other_cols = lag_other_cols
        self.missing_col_threshold = missing_col_threshold

        # learned in fit()
        self.base_columns_ = None  # after dropping very sparse columns
        self.lag_base_cols_ = None  # which columns we lag (target + others)
        self.feature_cols_ = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None

    # ---------- core helpers ----------

    def _normalize_target_and_time(self, df):
        """Parse time, sort, enforce freq, and normalize target missingness."""
        df = df.copy()

        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], format='%d/%m/%y %H:%M')
        df = df.sort_values(self.datetime_col).set_index(self.datetime_col)

        if self.freq is not None:
            df = df.asfreq(self.freq)

        # ensure target is float & normalize sentinel -9999 to NaN
        df[self.target_col] = df[self.target_col].astype(float)
        df.loc[df[self.target_col] == -9999, self.target_col] = np.nan

        # optional quality handling (not used in your current CSV, but safe)
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
                if col in [self.target_col, self.quality_col]:
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
        """Drop any obvious quality columns from the features."""
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
        """Decide which base columns will receive lag features."""
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
            for lag in [1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 47, 48]:
                df[f"{base_col}_lag{lag}"] = df[base_col].shift(lag)

        return df

    # ---------- sklearn API ----------

    def fit(self, df, y=None):
        # 1) normalize time + target
        df_norm = self._normalize_target_and_time(df)

        # 2) optional: drop very sparse columns
        df_base = self._drop_high_missing_columns(df_norm, fit_mode=True)

        # 3) remove quality columns
        df_noq = self._remove_quality_columns(df_base)

        # 4) decide which columns will have lags
        self._decide_lag_columns(df_noq)

        # 5) add time features
        df_feat = self._add_time_features(df_noq)

        # 6) add lag features
        df_feat = self._add_lag_features(df_feat)

        # 7) infer feature / numeric / categorical columns
        self.feature_cols_, self.numeric_cols_, self.categorical_cols_ = (
            _infer_feature_columns(df_feat, target_col=self.target_col)
        )

        return self

    def transform(self, df):
        """
        Returns:
          df_feat: DataFrame with features (including lags, time features), may contain NaNs
          y: 1D np.array, target values with NaNs where missing
          valid_mask: bool array, True where y is not NaN
          index: DatetimeIndex aligned with df_feat/y
        """
        # 1) normalize time + target
        df_norm = self._normalize_target_and_time(df)

        # 2) apply same sparse-column filter
        df_base = self._drop_high_missing_columns(df_norm, fit_mode=False)

        # 3) remove quality columns
        df_noq = self._remove_quality_columns(df_base)

        # 4) add time and lag features
        df_feat = self._add_time_features(df_noq)
        df_feat = self._add_lag_features(df_feat)

        # 5) build y and mask
        y = df_feat[self.target_col].values.astype(float)
        valid_mask = ~np.isnan(y)

        return df_feat, y, valid_mask, df_feat.index
