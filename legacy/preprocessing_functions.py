import pandas as pd
import numpy as np


def drop_serial_code(df_instance: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the 'codigoSerial' column from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the 'serial_code' column dropped.
    """
    df = df_instance.copy()
    return df.drop(columns=["codigoSerial"], axis=1)


def drop_nulls_columns(df_instance: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with at least 10% null values from the DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    Returns:
    pd.DataFrame: The DataFrame with columns containing at least 10% null values dropped.
    """

    df = df_instance.copy()

    # Calculate the threshold for null values
    threshold = len(df) * 0.25
    # Drop columns with at least 10% null values
    df = df.dropna(thresh=threshold, axis=1)
    return df


def drop_quality_columns(df_instance: pd.DataFrame) -> pd.DataFrame:
    """
    Drop specific quality-related columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with specific quality-related columns dropped.
    """

    df = df_instance.copy()

    # Get the columns with measurements
    columns_with_measurements = df.columns[~df.columns.str.startswith("calidad_")]

    # Get the columns with the quality of the measurements
    columns_with_quality = df.columns[df.columns.str.startswith("calidad_")]

    # Find the columns with quality of the measurements that have a corresponding column with measurements
    columns_with_quality_and_measurements = columns_with_quality[
        columns_with_quality.str.replace("calidad_", "").isin(columns_with_measurements)
    ]

    # Drop the columns with the quality of the measurements that do not have a corresponding column with measurements
    columns_with_quality_to_drop = columns_with_quality[
        ~columns_with_quality.isin(columns_with_quality_and_measurements)
    ]

    df.drop(columns_with_quality_to_drop, axis=1, inplace=True)
    return df


def fill_nulls_with_monthly_mean(df_instance, column_name, lower_limit, upper_limit):
    """
    Fill null or out-of-range values in a specified column with the monthly mean.
    If a monthly mean for (Año, Mes) is missing, fall back to per-month mean (across years),
    then global mean. If still missing, forward-fill remaining NaNs using previous valid value
    (and finally use global mean for any remaining head NaNs).
    """

    df = df_instance.copy()

    # Ensure Año/Mes exist
    if "Año" not in df.columns or "Mes" not in df.columns:
        if "Fecha_Hora" in df.columns:
            df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"])
            df["Año"] = df["Fecha_Hora"].dt.year
            df["Mes"] = df["Fecha_Hora"].dt.month
        else:
            raise KeyError(
                "Año/Mes columns missing and Fecha_Hora not present to derive them."
            )

    # Valid values used to compute means
    valid_mask = df[column_name].between(lower_limit, upper_limit)
    valid_values = df.loc[valid_mask, column_name]

    # If there are no valid values at all, just forward-fill existing non-nulls and return
    if valid_values.empty:
        df[column_name] = df[column_name].ffill()
        return df

    # Compute monthly means indexed by (Año, Mes)
    monthly_means = valid_values.groupby(
        [df.loc[valid_mask, "Año"], df.loc[valid_mask, "Mes"]]
    ).mean()
    monthly_means_dict = monthly_means.to_dict()

    # Fallback: mean per Mes across years, then global valid mean
    monthly_by_month = valid_values.groupby(df.loc[valid_mask, "Mes"]).mean()
    monthly_by_month_dict = monthly_by_month.to_dict()
    global_mean = valid_values.mean()

    # Build fill values for each row (use dict lookups with fallbacks)
    keys = list(zip(df["Año"], df["Mes"]))
    fill_values = [
        monthly_means_dict.get(k, monthly_by_month_dict.get(k[1], global_mean))
        for k in keys
    ]
    fill_array = np.array(fill_values, dtype="float64")

    # Mask rows that need replacement (out-of-range or NaN)
    to_replace = (~df[column_name].between(lower_limit, upper_limit)) | df[
        column_name
    ].isna()

    # Assign computed fills where needed
    if to_replace.any():
        df.loc[to_replace, column_name] = fill_array[to_replace.values]

    # Forward-fill any remaining NaNs (use previous valid value)
    df[column_name] = df[column_name].ffill()

    # If still NaN (e.g. head of series), fill with global mean if available
    if pd.isna(df[column_name].iloc[0]) and not pd.isna(global_mean):
        df[column_name].fillna(global_mean, inplace=True)

    return df
