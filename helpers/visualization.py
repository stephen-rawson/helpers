# import modules
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import stats
from scipy.stats import norm, boxcox
from srhelpers.encoding import encode_zscore
from srhelpers.utils import filtered_col_list
from math import ceil


def null_values(dfs, figure_output_dir: str, subtitles: list, title: str = "Missing Values", rot: int = 0,
                save: bool = True) -> None:
    """Plots null values across an iterable of dataframes. The subtitles
    Args:
        - Iterable of dataframes
        - Output dir to save figure
        - List of strings to identify subplots
        - Title shared for all subplots
        - Rotation for axis, default = 0
        - Option to save figure to output dir, defaults to True
    """

    cols = []
    for df in dfs:
        na_cols = [col for col in df.columns if df[col].isnull().values.any()]
        for col in na_cols:
            cols.append(col)

    cols_chosen = list(set(cols))  # remove duplicates

    if type(dfs) == list:
        original_order = list(dfs[0].columns)
    else:
        original_order = list(dfs.next().columns)
    cols_chosen.sort(key=lambda col: original_order.index(col))  # sort to match df columns

    plt.figure(figsize=(30, 15))

    count = 0

    for df, subtitle in zip(dfs, subtitles):
        ax = plt.subplot2grid((len(subtitles), 1), (count, 0), rowspan=1, colspan=1)

        subtitle = title + f", {subtitle}"

        ax = sns.heatmap(df[cols_chosen].isnull(), cbar=False)
        ax.get_yaxis().set_visible(False)

        plt.xticks(rotation=rot)
        plt.title(subtitle)

        plt.tight_layout()

        count += 1

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    if save:
        plt.savefig(os.path.join(figure_output_dir, title + ".png"), format='png', dpi=100)
    plt.show()
    plt.close()


def countplots(colname: str, output_dir: str, subtitles: list, dataframes, title: str,
               ylabel: str, limit: int = None,
               rot: int = 45, save=False, show: bool = True) -> None:
    """Plots value counts for categorical columns for a list of dataframes
    Args:
        - Column name representing categorical column to plot counts across
        - Iterable of dataframes
        - Title for graphs
        - Limit to the top n values/categories
        - Rotation
        - Save the plot?
        - Show the plot?
    """
    plt.figure(figsize=(30, 20))

    count = 0
    for df, subtitle in zip(dataframes, subtitles):

        subtitle = title + f", {subtitle.capitalize() + ' 2018'}"

        ax = plt.subplot2grid((len(subtitles), 1), (count, 0), rowspan=1, colspan=1)

        if limit:
            ax = sns.countplot(x=df[colname], order=dataframes[-1][colname].value_counts().iloc[:limit].index)
            sns.despine()
            ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            ax = sns.countplot(x=df[colname], order=dataframes[-1][colname].value_counts().index)
            ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
            sns.despine()

        for p, label in zip(ax.patches, df[colname].value_counts()):
            ax.annotate("{:,}".format(label), (p.get_x() + 0.4, p.get_height() + 5), ha="center", size=20)

        plt.title(subtitle)
        plt.ylabel(ylabel)
        plt.ylim((0, max(df[colname].value_counts()) * 1.1))
        plt.xlabel("")
        plt.xticks(rotation=rot)
        plt.tight_layout()

        textstr = "Total: {:,}".format(int(len(df)))

        ax.text(0.8, 0.90, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox={"boxstyle": "round", "facecolor": "aqua"})

        count += 1

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    if save:
        plt.savefig(os.path.join(output_dir, title + ".png"), format='png', dpi=100)

    if show:
        plt.show()

    plt.close()


def user_counts(colname: str, output_dir: str, df: pd.DataFrame, month: str, limit: int = None, ylabel = "User Counts",
               rot: int = 0, save=False, show: bool = True) -> None:
    """Plots value counts for categorical columns for a dataframe
    Args:
        - Column name representing categorical column to plot counts across
        - Dataframe
        - Title
        - Limit to the top n values/categories
        - Rotation
        - Save the plot?
        - Show the plot?
    """
    plt.figure(figsize=(15, 8))

    title = colname.capitalize() + f", {month.capitalize() + ' 2018'}"

    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    if limit:
        ax = sns.countplot(x=df[colname], order=df[colname].value_counts().iloc[:limit].index)
        sns.despine()
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    else:
        ax = sns.countplot(x=df[colname], order=df[colname].value_counts().index)
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        sns.despine()

    for p, label in zip(ax.patches, df[colname].value_counts()):
        ax.annotate("{:,}".format(label), (p.get_x() + 0.4, p.get_height() + 5), ha="center", size=14)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim((0, max(df[colname].value_counts()) * 1.1))
    plt.xlabel("")
    plt.xticks(rotation=rot)
    plt.tight_layout()

    textstr = "Total: {:,}".format(int(len(df)))

    ax.text(0.8, 0.90, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox={"boxstyle": "round", "facecolor": "aqua"})

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    if save:
        plt.savefig(os.path.join(output_dir, title + ".png"), format='png', dpi=100)

    if show:
        plt.show()

    plt.close()


def subscription_counts(dfs, output_path, month_names, save=False, show=False):
    """
    Saves a plot of total subscription counts.
    :param dfs: a list of dataframes
    :param output_path: the output dir
    :param month_names: the month names to use as xlabels
    :param save: whether to save
    :param show: whether to show or not
    :return:
    """
    subscription_numbers = []

    months = [name.capitalize() + " 2018" for name in month_names]

    for df in dfs:
        sub_count = len(df)
        subscription_numbers.append(sub_count)

    data = {"Month": months, "Sub Count": subscription_numbers}

    df_sub_count = pd.DataFrame.from_dict(data)

    plt.figure(figsize=(10, 8))

    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    ax = sns.barplot(x="Month", y="Sub Count", data=df_sub_count)

    for p, label in zip(ax.patches, subscription_numbers):
        ax.annotate("{:,}".format(label), (p.get_x() + 0.4, p.get_height() + 5), ha="center", size=14)

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.title(f"Total Subscriptions, {months[0]} - {months[-1]}")
    plt.ylim(0, max(subscription_numbers) * 1.2)
    plt.ylabel("Subscriptions")
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_path, "subscription_counts.png"), format='png', dpi=100)
    if show:
        plt.show()


def unique_subscribers(dfs, output_path, month_names, save=False, show=False):
    """
    Saves a plot of unique subscribers.
    :param dfs: list of dataframes
    :param output_path: output dir
    :param month_names: names to use a xlabels
    :param save: whether or not to save
    :param show: whether or not to show
    :return:
    """
    unique_subs = []

    months = [name.capitalize() + " 2018" for name in month_names]

    for df in dfs:
        unique_sub_count = df.msisdn.nunique()
        unique_subs.append(unique_sub_count)

    data = {"Month": months, "Unique Subs": unique_subs}

    df_sub_count = pd.DataFrame.from_dict(data)

    plt.figure(figsize=(10, 8))

    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    ax = sns.barplot(x="Month", y="Unique Subs", data=df_sub_count)

    for p, label in zip(ax.patches, unique_subs):
        ax.annotate("{:,}".format(label), (p.get_x() + 0.4, p.get_height() + 5), ha="center", size=14)

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.title(f"Unique Users, {months[0]} - {months[-1]}")
    plt.ylim(0, max(unique_subs) * 1.2)
    plt.ylabel("Unique Users")
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_path, "unique_subs.png"), format='png', dpi=100)

    if show:
        plt.show()


def plot_variable_over_time(colname: str, title: str, ylabel: str, periods: list, periods_dt: list, dfs,
                            figure_output_dir: str, save: bool = True,
                            small_numbers: bool = False, show: bool = True) -> None:
    """ Plots the average for the column over time from the supplied dataframes
    Args:
        - Column name to plot over time
        - Title of plot
        - Ylabel e.g., units
        - List of periods to use as xlabels
        - List of datetime dates to use as xticks
        - Iterable of dataframes, assumed to contain the relevant variable and be in time order
    """

    figname = title + f", {periods[0]} - {periods[-1]}"

    values = []

    for df in dfs:
        values.append(df[colname].mean())

    fig = plt.figure(figsize=(20, 8))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2, colspan=1)

    ax1.plot(periods_dt, values)

    if not small_numbers:
        textstr = """Mean: {:,}
    
    Range: {:,}
    Range as % of mean: {:.1%}
    
    Std. Deviation: {:,}
    Std. Deviation as % of Mean: {:.1%}""".format(
            int(np.mean(values)),
            int(max(values) - min(values)),
            float((max(values) - min(values)) / np.mean(values)),
            int(np.std(values)),
            float((np.std(values) / np.mean(values)))
        )

    if small_numbers:
        textstr = """Max: {:.2f}
    75th: {:.2f}
    50th: {:.2f}
    25th: {:.2f}
    Min: {:.2f}
    Mean: {:.2f}""".format(df[colname].max(),
                           df[colname].quantile(0.75),
                           df[colname].quantile(0.5),
                           df[colname].quantile(0.25),
                           df[colname].min(),
                           df[colname].mean()
                           )

    ax1.text(0.7, 0.90, textstr, transform=ax1.transAxes, fontsize=18,
             verticalalignment='top', bbox={"boxstyle": "round", "facecolor": "aqua"})

    if not small_numbers:
        ax1.get_yaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

    months_locs = mdates.MonthLocator()  # every month

    fmt = mdates.DateFormatter('%b-%Y')
    ax1.xaxis.set_major_locator(months_locs)
    ax1.xaxis.set_major_formatter(fmt)

    ax1.axhline(y=np.mean(values), c="g")

    ax1.legend([ylabel, "Period Average"], loc=2, fancybox=True)

    sns.despine()
    plt.title(figname)
    plt.ylabel(ylabel, labelpad=20)

    plt.ylim((min(values) * 0.8, max(values) * 1.2))

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(figure_output_dir, str(figname + ".png")), format='png')

    if show:
        plt.show()


def corr_plot(cols: list, dataframe: pd.DataFrame, title: str, figure_output_dir: str, periods: list,
              save: bool = True, show = False) -> None:
    """
    Args:
    :param figure_output_dir: where to save the file
    :param cols: list of column names to correlate
    :param dataframe: dataframe to find column names in
    :param title: title of plot
    """
    if len(periods) > 1:
        figname = title + f" (Averaged Trailing Months), {periods[0].capitalize()} - {periods[-1].capitalize()}"
    else:
        figname = title + "Averaged Trailing Months"

    # Subset the df by the columns supplied
    subset = dataframe[cols]

    # Generate correlations
    corr = subset.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(400, 400))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt='.2f')

    plt.title(figname)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(figure_output_dir, str(figname + ".png")), format='png')
    if show:
        plt.show()


def plot_dists(colname: str, figure_output_dir: str, periods: list, dataframes,
               title: str = "Negative Credit Users", xlimit=150000,
               bins=30, save=True, small_numbers: bool = False, show=True) -> None:
    """
    Plots histograms for all dataframes supplied for the given column name.
    :param colname: column to plot
    :param title: title for graph
    :param xlabel: variable name
    :param figure_output_dir: output directory for saving
    :param periods: list of strings representing periods over which distributions are plotted
    :param dataframes: iterable of dataframes
    :param xlimit: limit for plotting
    :param bins: number of bins
    :return: None, calls to plt.show()
    """

    plt.figure(figsize=(20, 25))
    xlabel = colname
    count = 0
    for df, period in zip(dataframes, periods):
        subtitle = title.capitalize() + f", {period}"

        ax1 = plt.subplot2grid((len(periods), 2), (count, 0), rowspan=1, colspan=2)
        if not small_numbers:
            ax1.get_xaxis().set_major_formatter(
                FuncFormatter(lambda x, p: format(int(x), ',')))

        ax1.get_yaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

        plt.title(subtitle)
        plt.ylabel("Count")
        plt.xlim((0, xlimit))

        textstr = """Max: {:,}
75th: {:,}
50th: {:,}
25th: {:,}
Min: {:,}
Mean: {:,}""".format(int(df[colname].max()),
                     int(df[colname].quantile(0.75)),
                     int(df[colname].quantile(0.5)),
                     int(df[colname].quantile(0.25)),
                     int(df[colname].min()),
                     int(df[colname].mean())
                     )

        if small_numbers:
            textstr = """Max: {:.2f}
75th: {:.2f}
50th: {:.2f}
25th: {:.2f}
Min: {:.2f}
Mean: {:.2f}""".format(df[colname].max(),
                       df[colname].quantile(0.75),
                       df[colname].quantile(0.5),
                       df[colname].quantile(0.25),
                       df[colname].min(),
                       df[colname].mean()
                       )

        ax1.text(0.8, 0.90, textstr, transform=ax1.transAxes, fontsize=18,
                 verticalalignment='top', bbox={"boxstyle": "round", "facecolor": "aqua"})
        sns.despine()

        sns.despine()

        sns.distplot(df[df[colname] < xlimit][colname].dropna(), hist=True, bins=30, ax=ax1, norm_hist=False, kde=False,
                     axlabel=xlabel)

        plt.tight_layout()

        count += 1

    plt.tight_layout()
    plt.subplots_adjust(hspace=1)

    if save:
        plt.savefig(os.path.join(figure_output_dir, title + ", " + colname + ".png"), format='png', dpi=100)

    if show:
        plt.show()
    plt.close()

def plot_dist(colname: str, figure_output_dir: str, df, period:str,
               title: str = "Negative Credit Users", xlimit=150000,
               bins=30, save=False, small_numbers: bool = False, show=True) -> None:
    """
    Plots histogram for dataframe for the given column name.
    :param colname: column to plot
    :param title: title for graph
    :param figure_output_dir: output directory for saving
    :param df: dataframe
    :param xlimit: limit for plotting
    :param bins: number of bins
    :return: None, calls to plt.show()
    """

    plt.figure(figsize=(10, 8))
    xlabel = colname
    subtitle = title + f", {period}"

    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    if not small_numbers:
        ax1.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

    ax1.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.title(subtitle)
    plt.ylabel("Count")
    plt.xlim((0, xlimit))

    textstr = """Max: {:,}
75th: {:,}
50th: {:,}
25th: {:,}
Min: {:,}
Mean: {:,}""".format(int(df[colname].max()),
                 int(df[colname].quantile(0.75)),
                 int(df[colname].quantile(0.5)),
                 int(df[colname].quantile(0.25)),
                 int(df[colname].min()),
                 int(df[colname].mean())
                 )

    if small_numbers:
        textstr = """Max: {:.2f}
75th: {:.2f}
50th: {:.2f}
25th: {:.2f}
Min: {:.2f}
Mean: {:.2f}""".format(df[colname].max(),
                   df[colname].quantile(0.75),
                   df[colname].quantile(0.5),
                   df[colname].quantile(0.25),
                   df[colname].min(),
                   df[colname].mean()
                   )

    ax1.text(0.8, 0.90, textstr, transform=ax1.transAxes, fontsize=18,
             verticalalignment='top', bbox={"boxstyle": "round", "facecolor": "aqua"})
    sns.despine()

    sns.despine()

    sns.distplot(df[df[colname] < xlimit][colname].dropna(), hist=True, bins=30, ax=ax1, norm_hist=False, kde=False,
                 axlabel=xlabel)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(figure_output_dir, subtitle + ", " + colname + ".png"), format='png', dpi=100)

    if show:
        plt.show()
    plt.close()


def plot_categorical_dependence(dataframe: pd.DataFrame, target: str, category: str) -> None:
    """
    Plots the continuous target variable on the y-axis against a categorical independent variable
    on the x-axis.
    :param dataframe: the dataframe containing the target and the feature
    :param target: the target/dependent variable
    :param category: the categorical independent variable
    :return: None, calls to plt.show()
    """
    data = pd.concat([dataframe[target], dataframe[category]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=category, y=category, data=data)
    fig.axis(ymin=0, ymax=800000);
    plt.show()


def focused_corr_plot(n: int, dataframe: pd.DataFrame, target: str) -> None:
    """
    Plots a correlation heatmap focused to the n highest correlations with the
    specified target column.
    :param n: the number of correlating variables to show
    :param dataframe: the dataframe to find correlations in
    :param target: the target column
    :return: None, calls to plt.show()
    """
    corrmatrix = dataframe.corr()
    f, ax = plt.subplots(figsize=(8, 6))
    cols = corrmatrix.nlargest(n, target)[target].index
    cm = np.corrcoef(dataframe[cols].values.T)
    fig = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                      annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


def plot_df_transformed(df: pd.DataFrame, filters: tuple, zero_values: str = "ignore") -> None:
    """
    Plots three graphs for each variable satisfying the filters; original distribution,
    transformed distribution and transformed probability plot.
    :param df: supplied dataframe
    :param zero_values: either "ignore" or "increment"
    :param filters: tuple containing filter strings for columns
    :return:
    """
    to_plot = filtered_col_list(df, strings=filters)

    height = ceil(len(to_plot) * 10 / 3)

    fig = plt.figure(figsize=(12, height))

    count = 0
    for var in to_plot:
        try:
            if zero_values == "ignore":
                series = pd.Series(df[np.abs(df[var]) > 0][var])
            elif zero_values == "increment":
                series = pd.Series(df[var] + 0.0001)

            ax1 = plt.subplot2grid((len(to_plot), 3), (count, 0), colspan=1, rowspan=1)
            ax1.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.yticks([], [])
            sns.distplot(series, fit=norm)
            plt.title("Distribution: Original Data")
            plt.ylabel("Density")

            series_transformed = boxcox(np.abs(np.asarray(series.values)))[0]

            ax2 = plt.subplot2grid((len(to_plot), 3), (count, 1), rowspan=1, colspan=1)
            ax2.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
            sns.distplot(series_transformed, fit=norm)
            plt.xlabel(var)
            plt.title("Distribution: Boxcox Data")
            plt.yticks([], [])
            plt.ylabel("Density")

            ax3 = plt.subplot2grid((len(to_plot), 3), (count, 2), colspan=1, rowspan=1)
            ax3.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
            stats.probplot(series_transformed, plot=ax3)
            plt.title("Probability Plot: Boxcox Data")
            plt.xlabel(var)
            plt.yticks([], [])

        except Exception as e:
            print(var, e)

        count += 1

    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()

    plt.show()


def plot_prediction_thresholds(predicted_probabilities: np.ndarray, y_test: np.ndarray,
                               thresholds: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) -> None:
    """
    Plots a grid of plots show confusion matrices for each of the supplied decision thresholds
    :param thresholds: a list of probability thresholds for classification i.e. [0.25, 0.5, 0.75]
    :param predicted_probabilities: an array of predicted probabilities
    :param y_test: the test values for the target variable
    :return: None, calls to plt.show()
    """

    plt.figure(figsize=(10, 10))

    j = 1
    for i in thresholds:
        preds = predicted_probabilities[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_test, preds)
        np.set_printoptions(precision=2)

        print(f"Recall metric in the testing dataset with threshold {i}: ",
              cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

        # Plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix
                              , classes=class_names
                              , title=f"Threshold >= {i}")

    plt.show()


def plot_confusion_matrix(cm: metrics.confusion_matrix, names: list,
                          title: str = 'Confusion matrix', cmap: plt.cm = plt.cm.Blues) -> None:
    """ Plots a confusion matrix for the given CM and names of classes."""
    cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_roc(pred, y):
    """ Plots a ROC curve for the provided predictions and target values"""
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def visualize_outliers(dataframes: list, string_filter: str = "avg",
                       to_drop: list = ["msisdn", "nps_score", "since_survey"], threshold: float = 3,
                       ret: bool = True, fig_output_path: str = None, save: bool = False) -> pd.DataFrame:
    """
    Plots bar plots for each dataframe showing the number of outliers.
    :param dataframes: list of dataframes to plot for
    :param string_filter: filter for columns to consider, default "avg"
    :param to_drop: columns to drop
    :param ret: should it return the dataframe containing the specified columns?
    :param fig_output_path: output path for figure saving
    :param save: should it save the figure?
    :return: if returns, returns the dataframe used to plot
    """
    dfs_numerical = []

    for dataframe in dataframes:
        filtered = dataframe.select_dtypes(exclude=["category"], include=[np.number])

        cols = [col for col in filtered.columns if string_filter in str(col)]

        filtered2 = filtered.drop(columns=to_drop)[cols]

        dfs_numerical.append(filtered2)

    plt.figure(figsize=(20, 25))

    looper = dfs_numerical.copy()
    count = 0
    for df in looper:
        for col in df.columns:
            encode_zscore(df, col)

        mask = (df > 3) | (df < -3)
        df = df[mask]

        ax1 = plt.subplot2grid((len(dfs_numerical), 1), (count, 0), rowspan=1, colspan=1)
        ax1 = df.sum(0).abs().plot.bar()
        ax1.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        count += 1

    plt.tight_layout()
    plt.show()

    if ret:
        return dfs_numerical
