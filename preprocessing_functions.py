import pandas as pd
import numpy as np


def drop_serial_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the 'codigoSerial' column from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the 'serial_code' column dropped.
    """
    return df.drop(columns=["codigoSerial"], axis=1)


def drop_nulls_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with at least 10% null values from the DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    Returns:
    pd.DataFrame: The DataFrame with columns containing at least 50% null values dropped.
    """

    # Calculate the threshold for null values
    threshold = len(df) * 0.1
    # Drop columns with at least 10% null values
    df = df.dropna(thresh=threshold, axis=1)
    return df
