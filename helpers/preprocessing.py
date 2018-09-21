    import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy import stats


def fill_missing(df: pd.DataFrame, col: str, mode: str = 'mean', value: float = None):
    """ Fills the numerical column provided
    with the supplied average type (mode or median) or a provided value.
    Args: dataframe, column name, mode, value
    """
    assert mode in ["mean", "median", "mode", "value"]

    if mode == "mean":
        mean = df[col].mean()
        df[col] = df[col].fillna(mean)

    if mode == "median":
        median = df[col].median()
        df[col] = df[col].fillna(median)

    if mode == "mode":
        df[col] = df[col].fillna(df[col].mode()[0])

    if mode == "value":
        df[col] = df[col].fillna(value)


def filter_zscore(df: pd.DataFrame, alpha: float = 3):
    """Drops all rows containing outliers as defined by the
    provided threshold (which multiplies the standard deviation).
    Equivalent to dropping rows for a given high Z-score.
    Args:
        - df: pandas dataframe object
        - alpha: coefficient on standard deviation
    """
    return df[(np.abs(stats.zscore(df)) < alpha).all(axis=1)]


def filter_iqr(df: pd.DataFrame, ignore: Tuple[str] = '', range_filter: Tuple[float] = (0.05, 0.95)) -> pd.DataFrame:
    """
    Encodes the numerical column provided
    as boolean
    :param df: The DataFrame to filter
    :param range_filter: The chosen lower/upper cutoffs
    :param ignore: list of columns to ignore when performing the filtering
    :return: a filtered dataframe object
    """
    filter_cols = [col for col in df.columns if df[col].dtype == np.float32 and col not in ignore]

    for col in filter_cols:
        df = df.loc[(df[col] < max(range_filter)) | (df[col] > min(range_filter))]

    return df
