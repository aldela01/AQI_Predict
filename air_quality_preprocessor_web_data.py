import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def _infer_feature_columns(df, target_col):
    # datetime is in the index now; target excluded from X
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols


class AirQualityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing for pm25 forecasting / imputation.

    - Uses calidad_pm25 + -9999 to mark invalid pm25.
    - Optionally drops columns with too many missing values (+ their calidad_* or *_calidad).
    - Removes *all* quality columns from the feature set.
    - Adds time features.
    - Adds lag features for pm25 and optionally for other numeric variables.
    - Imputes numeric features (NOT pm25) with a time-aware scheme.
    - Scales numeric features (optional) and one-hot encodes categoricals.

    transform() returns:
      X: 2D array, features (no NaNs)
      y: 1D array, pm25 (NaN where invalid)
      valid_mask: bool array, True where y is valid
      index: DatetimeIndex
    """

    def __init__(
        self,
        target_col="pm25",
        quality_col="calidad_pm25",
        datetime_col="timestamp",
        freq=None,  # e.g. "1H"
        use_cyclical_time=True,
        max_lag=24,  # lags from 1..max_lag
        lag_other_cols=None,  # list of column names to lag; None = all numeric except target
        missing_col_threshold=0.5,  # drop cols with >50% missing
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

        # attributes learned in fit()
        self.base_columns_ = None
        self.lag_base_cols_ = None  # columns that get lags
        self.feature_cols_ = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.numeric_medians_ = None
        self.column_transformer_ = None

    # ---------- core helpers ----------

    def _normalize_pm25_and_time(self, df):
        """Parse time, enforce freq, and mark pm25 invalid using quality + -9999."""
        df = df.copy()

        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.sort_values(self.datetime_col).set_index(self.datetime_col)

        if self.freq is not None:
            df = df.asfreq(self.freq)

        # pm25 to float, -9999 -> NaN
        df[self.target_col] = df[self.target_col].astype(float)
        df.loc[df[self.target_col] == -9999, self.target_col] = np.nan

        # calidad_pm25 != 1 -> invalid pm25
        if self.quality_col in df.columns:
            bad = df[self.quality_col] != 1
            df.loc[bad, self.target_col] = np.nan

        return df

    def _drop_high_missing_columns(self, df, fit_mode=True):
        """Drop columns (except target and its quality) with too many missing values."""
        if self.missing_col_threshold is None:
            return df

        if fit_mode:
            df_eval = df.copy()
            cols_to_drop = set()
            cols = df_eval.columns

            for col in cols:
                if col in [self.target_col, self.quality_col]:
                    continue

                s = df_eval[col]
                if pd.api.types.is_numeric_dtype(s):
                    missing = s.isna() | (s == -9999)
                else:
                    missing = s.isna()

                if missing.mean() > self.missing_col_threshold:
                    cols_to_drop.add(col)
                    # also drop its quality column if present
                    q1 = f"calidad_{col}"
                    q2 = f"{col}_calidad"
                    if q1 in cols:
                        cols_to_drop.add(q1)
                    if q2 in cols:
                        cols_to_drop.add(q2)

            self.base_columns_ = [c for c in cols if c not in cols_to_drop]
            return df_eval[self.base_columns_]
        else:
            if self.base_columns_ is None:
                return df
            existing = [c for c in self.base_columns_ if c in df.columns]
            return df[existing]

    def _remove_quality_columns(self, df):
        """
        Remove all quality columns from the dataframe
        (anything starting with 'calidad_' or ending with '_calidad').
        """
        df = df.copy()
        q_cols = [
            c for c in df.columns if c.startswith("calidad_") or c.endswith("_calidad")
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
        """
        Decide which base columns will have lag features.
        Runs only in fit().
        """
        if self.max_lag is None or self.max_lag <= 0:
            self.lag_base_cols_ = []
            return

        # numeric columns (before time features)
        numeric_base = [
            c for c in df_noq.columns if pd.api.types.is_numeric_dtype(df_noq[c])
        ]

        if self.lag_other_cols is None:
            # auto: all numeric except target
            other = [c for c in numeric_base if c != self.target_col]
        else:
            # user-provided list (intersection)
            other = [
                c
                for c in self.lag_other_cols
                if c in df_noq.columns and c != self.target_col
            ]

        self.lag_base_cols_ = [self.target_col] + other

    def _add_lag_features(self, df):
        """Add lagged columns for each base column in self.lag_base_cols_."""
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
        # 1) normalize pm25 and time
        df_norm = self._normalize_pm25_and_time(df)

        # 2) drop columns with too many missing values
        df_base = self._drop_high_missing_columns(df_norm, fit_mode=True)

        # 3) remove all quality columns
        df_noq = self._remove_quality_columns(df_base)

        # 4) decide which base columns get lags (pm25 + others)
        self._decide_lag_columns(df_noq)

        # 5) add time features
        df_feat = self._add_time_features(df_noq)

        # 6) add lag features
        df_feat = self._add_lag_features(df_feat)

        # 7) infer feature, numeric, categorical columns
        self.feature_cols_, self.numeric_cols_, self.categorical_cols_ = (
            _infer_feature_columns(df_feat, target_col=self.target_col)
        )

        # 8) numeric medians for imputation
        self.numeric_medians_ = df_feat[self.numeric_cols_].median()

        # 9) impute numeric for fitting scaler/encoder
        df_imputed = df_feat.copy()
        df_imputed[self.numeric_cols_] = df_imputed[self.numeric_cols_].ffill().bfill()
        for col in self.numeric_cols_:
            df_imputed[col] = df_imputed[col].fillna(self.numeric_medians_[col])

        # 10) ColumnTransformer
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
        # 1) normalize pm25 and time
        df_norm = self._normalize_pm25_and_time(df)

        # 2) apply same base-column filter
        df_base = self._drop_high_missing_columns(df_norm, fit_mode=False)

        # 3) remove all quality columns
        df_noq = self._remove_quality_columns(df_base)

        # 4) add time features
        df_feat = self._add_time_features(df_noq)

        # 5) add lag features (using self.lag_base_cols_)
        df_feat = self._add_lag_features(df_feat)

        # 6) target and validity mask
        y = df_feat[self.target_col].values.astype(float)
        valid_mask = ~np.isnan(y)

        # 7) impute numeric features
        df_feat[self.numeric_cols_] = df_feat[self.numeric_cols_].ffill().bfill()
        for col in self.numeric_cols_:
            df_feat[col] = df_feat[col].fillna(self.numeric_medians_[col])

        # 8) build X
        X = self.column_transformer_.transform(df_feat[self.feature_cols_])

        return X, y, valid_mask, df_feat.index
