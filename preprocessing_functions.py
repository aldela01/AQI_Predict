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
