import pandas as pd
import numpy as np
import os


def load_stock_data(filepath):
    """
    Load stock market CSV data with 'Date' as datetime index.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Cleaned DataFrame with datetime index
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if 'Date' not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def clean_missing_data(df, column='Close'):
    """
    Fill or drop missing values for the target column.

    Args:
        df (pd.DataFrame): DataFrame with stock data
        column (str): Column to clean (default: 'Close')

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    df = df.copy()
    df[column].fillna(method='ffill', inplace=True)
    df[column].fillna(method='bfill', inplace=True)
    return df


def resample_data(df, freq='D', column='Close'):
    """
    Resample stock data to a consistent frequency (daily, weekly, etc.).

    Args:
        df (pd.DataFrame): Original stock data
        freq (str): Frequency (e.g., 'D', 'W', 'M')
        column (str): Column to resample

    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    df_resampled = df[[column]].resample(freq).mean()
    df_resampled = clean_missing_data(df_resampled, column)
    return df_resampled
