# Helpers

## Overview
A set of classes, functions and visualisation tools to accelerate EDA and preprocessing with pandas, matplotlib and seaborn.

Composed of the following modules:
- Utils
    - Multigen class to allow continous iteration over dataframes without redeclaring the generator
    - Load data function to easily load multiple dataframes i.e. the same dataframe over multiple periods
    - Function to check for duplicate values in a given column across a list of dataframes
    - Function to check memory usage of a pandas object i.e. dataframe or series
    - Function to loop through dataframes and optimize memory usage by downcasting, categorization, etc.
    - Function to allow easier column filtering by start, end and contain string filters
    - Function to split lists into sublists based on a desired sublist length, useful for generating sublists of specified length
- Averages
    - Function to return an average dataframe from a dataframe containing many lagged values
- Encoding
    - Function to create dummy columns in place or in addition to a column containing categorical data in string form
    - Function to encode a text column into a numerical column, returning the lookup if required
    - Functions to encode numerical columns as their z-score, modified z-score and min-max standardized versions
    - Function to return X, y numpy arrays derived from a pandas df, suitable for direct implementation in tensorflow or sklearn, etc
- Preprocessing
    - Function to fill missing values in a dataframe's column with mean, mode, median or specified value methods
    - Function to return a filtered dataframe based on the specified z-score
    - Function to return a filtered dataframe based on the specified multiple of the IQR
- Visualization
    - A selection of visualization functions built on matplotlib and seaborn
        - Null values
        - Outlier visualization
        - Distributions across a single dataframe / multiple dataframes
        - Categorical countplots
        - Time series plotting of averages across multiple dataframes
        - Correlation plots
        - Comparisons between original and boxcox data
        - Prediction thresholds
        - ROC plotting
        - Classification report / confusion matrix