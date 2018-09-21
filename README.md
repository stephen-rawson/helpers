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
