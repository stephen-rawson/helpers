# import modules
import pandas as pd
import numpy as np
from srhelpers.utils import split_lists


def create_average_df(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """ Returns average dataframe derived from the input dataframe.
    Assumes trailing values are specified by integer endings i.e. data_1.

    :param df: dataframe to average over
    :param lags: the number of columns to average over for each average (i.e. 3 trailing columns = 3 to average)
    :return: average dataframe, excluding non-averaged columns from the original
    """
    # get list of columns to average over
    endings = tuple([str(i+1) for i in range(lags)])
    to_average = [col for col in df.columns if col.endswith(endings) and not "tariff" in col]

    # split list of columns into
    lol = split_lists(to_average, lags)

    # dataframe creation
    output_df = pd.DataFrame(index=df.index)

    for l in lol:
        col_name = str(l[0])[:-1] + "avg"
        output_df[col_name] = df[l].mean(axis=1)

    return output_df
