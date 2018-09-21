# import modules used
import pandas as pd
import time
import os
import calendar
from datetime import date
import numpy as np
from typing import List

# define and functions used classes used
def multigen(gen_func):
    """
    Returns a 'multigen' object, i.e. a generator that
    does not need to be redefined.

    :param gen_func: a function returning a generator
    :return:
    """
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)

        def next(self):
            return self.__iter__().__next__()

    return _multigen


@multigen
def dataframes(filenames: List(str), filepath: str):
    """
    A multigen-style iterator of dataframe objects.

    :param filenames: a list of filenames to load dataframes for
    :param filepath: a filepath from which to load dataframes
    :return:
    """
    i = 0
    while i < len(filenames):
        df = pd.read_pickle(os.path.join(filepath, filenames[i]))
        yield df
        i += 1


def check_duplicates(df_names: List(str), dataframes, colname: str) -> None:
    """
    Basic check, whether or not duplicates are found across the dataframes
    provided in the column specified.

    :param df_names: list of strings to zip to / match dataframes provided
    :param dataframes: iterable of dataframe objects
    :param colname: string column name to check for duplicates
    :return: None
    """
    # check for duplicates
    for df_name, df in zip(df_names, dataframes):
        no_dupes = df[colname].nunique() == df[colname].count()
        if no_dupes:
            no_dupes = "No Duplicates"
        else:
            no_dupes = "Duplicates Found"

        print(f"{df_name.capitalize()} : {no_dupes}")


def load_data(df_names: list, suffix: str, fpath: str, mode: str = "gen", ftype="pickle", verbose=True):
    """
    Returns either a list or resusable generator of dataframes

    :param df_names: list of df names to load
    :param suffix: suffix to apply to generate full filename e.g., "_data.csv"
    :param fpath: filepath to find the filenames in
    :param mode: lst or gen
    :param ftype: filetype, expects either pickle or csv
    :return: list or generator
    """
    assert mode in ["lst", "gen"]

    # if list is chosen, return a list of dataframes
    if mode == "lst":
        dfs = []
        for df_name in df_names:
            path = os.path.join(fpath, df_name + suffix)
            if ftype == "pickle":
                df = pd.read_pickle(path)
            elif ftype == "csv":
                df = pd.read_csv(path, low_memory=False)

            if verbose:
                print(f"{df_name.capitalize()}: {len(df)} subscriptions")

            dfs.append(df)

        return dfs

    # otherwise if generator is specified, return a multigen object of the dfs
    elif mode == "gen":
        fnames = [df_name + suffix for df_name in df_names]
        dfs = dataframes(filenames=fnames, filepath=fpath)
        return dfs


def mem_usage(pandas_obj) -> str:
    """ Returns the memory usage of a pandas object """
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes

    return f"{usage_mb:03.2f} mb"


def optimize_dataframes(dataframe_names: list, dataframes: list) -> list:
    """
    Takes a list of dataframes and returns a list of dataframes with
    optimized datatypes. Displays summary information regarding the relative
    memory usage of the initial and optimized versions.

    :param dataframe_names: list of names to associate with dataframes
    :param dataframes: a list of dataframes to optimize
    :return: a list of optimized dataframes
    """
    print("-" * 30)
    print("SUMMARIZING MEMORY USAGE")
    print("-" * 30)

    for df_name, df in zip(dataframe_names, dataframes):
        print(df_name.capitalize())
        print(df.info(memory_usage="deep"))
        print("-" * 30)

    for df, df_name in zip(dataframes, dataframe_names):
        print(df_name.capitalize())
        print("-" * 25)
        for dtype in ['float', 'int', 'object']:
            selected_dtype = df.select_dtypes(include=[dtype])
            mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
            mean_usage_mb = mean_usage_b / 1024 ** 2
            print("Average memory usage for {} columns: {:03.2f} MB".format(dtype, mean_usage_mb))
        print("")

    print("-" * 30)
    print("OPTIMIZING MEMORY USAGE")
    print("-" * 30)

    optimized_dfs = []

    for df, df_name in zip(dataframes, dataframe_names):
        print("-" * 25)
        print(df_name)
        print("-" * 25)

        try:
            print("ints")
            ints = df.select_dtypes(include=["int"])
            converted_ints = ints.apply(pd.to_numeric, downcast="unsigned")

            print("Original: ", mem_usage(ints))
            print("Optimized: ", mem_usage(converted_ints))

            print("")

        except Exception as e:
            print(e)

        try:
            print("floats")
            floats = df.select_dtypes(include=["float"])
            converted_floats = floats.apply(pd.to_numeric, downcast="float")

            print("Original: ", mem_usage(floats))
            print("Optimized: ", mem_usage(converted_floats))

            print("")

        except Exception as e:
            print(e)

        try:
            print("objects")
            objs = df.select_dtypes(include=["object"]).copy()

            converted_objs = pd.DataFrame()

            for col in objs.columns:
                num_unique_values = len(objs[col].unique())
                num_total_values = len(objs[col])
                if num_unique_values / num_total_values < 0.5:
                    converted_objs.loc[:, col] = objs[col].astype('category')
                else:
                    converted_objs.loc[:, col] = objs[col]

            print("Original: ", mem_usage(objs))
            print("Optimized: ", mem_usage(converted_objs))

        except Exception as e:
            print(e)

        optimized_df = df.copy()
        optimized_df[converted_ints.columns] = converted_ints
        optimized_df[converted_floats.columns] = converted_floats
        optimized_df[converted_objs.columns] = converted_objs

        print("")
        print("total")
        print("Original: ", mem_usage(df))
        print("Optimized: ", mem_usage(optimized_df))

        optimized_dfs.append(optimized_df)
        print("")

    print("-" * 30)
    print("SUMMARY OPTIMIZED")
    print("-" * 30)

    for df_name, optimized_df in zip(dataframe_names, optimized_dfs):
        print(df_name)
        print("-" * 30)
        print(optimized_df.info())
        print("")

    return optimized_dfs


def filtered_col_list(df: pd.DataFrame, starts: tuple, ends: tuple, contains: tuple) -> list:
    col_list = []

    for col in df.columns:
        for start in starts:
            if col.startswith(start):
                col_list.append(col)
            else:
                pass
        for end in ends:
            if col.endswith(end):
                col_list.append(col)
            else:
                pass
        for contain in contains:
            if contain in col:
                col_list.append(col)
            else:
                pass

    col_list = list(set(col_list))

    return col_list
