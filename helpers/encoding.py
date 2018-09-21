# import modules
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from typing import List


def text_to_dummies(df: pd.DataFrame, col: str, drop_f: bool = False, drop_original: bool = True):
    """ Encodes a column containing n unique
    values into n binary indicator columns.

    Args:
        - dataframe
        - name of column to create dummies for
        - drop_f; whether or not to create n-1 dummies
        - drop_original: whether or not to drop the original column
    """
    dummies = pd.get_dummies(data=df[col], drop_first=drop_f)

    for column in dummies.columns:
        dummy_name = f"{name}-{column}"
        df[dummy_name] = dummies[column]
    if drop_original:
        df.drop(col, axis=1, inplace=True)


def encode_text_index(df: pd.DataFrame, col: str, ret: bool = True):
    """ Encodes a column containing n unique
    values into a single indicator column.
    Returns the lookup array.

    Args:
        - dataframe
        - name of column to encode
        - whether or not to return the lookup array
    """
    # define the encoder
    encoder = preprocessing.LabelEncoder()

    # fit and transform in place
    df[col] = encoder.fit_transform(df[col])

    if ret:
        return encoder.classes_


def encode_zscore(df: pd.DataFrame, col: str) -> None:
    """ Encodes the numerical column provided
    as a Z-score variable.

    Args: dataframe, column name
    """
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std


def encode_modified_zscore(df: pd.DataFrame, col: str) -> None:
    """
    Encodes a numerical column provided with a modified (robust)
    Z-score.

    :param df: The supplied dataframe
    :param col: The chosen column
    :return: Returns none, modifies inplace.
    """
    median = df[col].median()
    median_absolute_deviation = np.median(np.abs(df[col] - median))
    df[col] = 0.6745 * (df[col] - median) / median_absolute_deviation


def encode_min_max(df: pd.DataFrame, col: str) -> None:
    """ Encodes the numerical column provided
    as a min-max normalized variable

    Args: dataframe, column name
    """
    maximum = df[col].max()
    minimum = df[col].min()
    df[col] = df[col] / (maximum - minimum)


def df_convert_Xy(df: pd.DataFrame, label_col: str, mode=None) -> tuple:
    """ Converts a fully numeric dataframe with a specified label column
    into a matrix and vector suitable for classification
    Args:
        - Dataframe
        - Column name to treat as target
        - Mode: classification vs. regression
    Returns: X, y
    """
    if df.isna().sum().sum() > 0:
        raise ValueError("Null values encountered in dataframe.")

    if mode not in ["classification", "regression", None]:
        raise ValueError("Mode expected either 'classification' or 'regression', but got neither.")

    # empty list to store column names for independent variables
    X = []

    # append each non-target column to this list
    for col in df.columns:
        if col != label_col:
            X.append(col)

    # check the type of classification
    y_type = df[label_col].dtypes
    y_type = y_type[0] if hasattr(
        y_type, '__iter__') else y_type

    if mode:
        if mode == "classification":
            dummies = pd.get_dummies(df[label_col])
            return df[X].values.astype(np.float32), dummies.values.astype(np.float32)

        elif mode == "regression":
            return df[X].values.astype(np.float32), df.y.values.astype(np.float32)

    else:
        if y_type in (np.int64, np.int32):
            # for classification
            dummies = pd.get_dummies(df[label_col])

            return df[X].values.astype(np.float32), dummies.values.astype(np.float32)

            # for regression
        return df[X].values.astype(np.float32), df.y.values.astype(np.float32)
