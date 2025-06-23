#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:00:26 2025

@author: juanitagomez
"""
import sqlite3
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import string
import matplotlib
import random


matplotlib.rcParams['font.family'] = 'Lato'

def db_to_df(db_path, output_filename, db_type='sqlite', db_params=None):
    """
    Exports all data from 'repositories' to a DataFrame, including org URL and email.

    :param db_path: Path to the SQLite database.
    :param output_filename: Base name for the CSV file (unused here).
    :param db_type: Type of database ('sqlite' only for now).
    :param db_params: Dictionary with connection parameters for PostgreSQL (unused).
    :return: A pandas DataFrame with repository info and organization contact details.
    """
    if db_type == 'sqlite':
        conn = sqlite3.connect(db_path)
    else:
        raise ValueError("Unsupported database type. Use 'sqlite'.")

    # Load repositories and organizations
    repo_df = pd.read_sql_query("SELECT * FROM repositories", conn)
    org_df = pd.read_sql_query("SELECT login, url AS org_url, email AS org_email FROM organizations", conn)

    # Merge on owner == login
    merged_df = repo_df.merge(org_df, how='left', left_on='owner', right_on='login')

    conn.close()

    # Clean up
    merged_df['subscribers_count'] = merged_df['subscribers_count'].fillna(0).astype(int)
    merged_df['release_downloads'] = merged_df['release_downloads'].fillna(0).astype(int)
    merged_df.drop(columns=['login'], inplace=True)  # Remove redundant column after merge

    return merged_df

def filter_data(data, threshold):
    """
    Filters the input data to include only rows where the 'prediction' column is greater than the threshold.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing related data.

    Returns
    -------
    pandas.DataFrame
        The filtered dataset with 'prediction' greater than threshold.
    """
    data['ai_prediction'] = data['ai_prediction'].astype(float)
    return data[data['ai_prediction'] > threshold].reset_index(drop=True)

def build_shared_color_map(all_data_dict, column, threshold=0.02):
    """
    Build a shared color map for a specified column across multiple DataFrames.
    
    This function aggregates values from the specified column in all DataFrames 
    contained in `all_data_dict`, applies a frequency threshold to filter out 
    infrequent labels, and generates a color map for the remaining labels. 
    Common infrequent or missing labels are grouped under "Other" or "None".
    
    Parameters:
    ----------
    all_data_dict : dict of {str: pd.DataFrame}
        A dictionary mapping university acronyms to their corresponding DataFrames.
    column : str
        The column name for which the color map is to be generated (e.g., 'language', 'license').
    threshold : float, optional
        The minimum proportion (global frequency) a label must have to be assigned its own color.
        Labels below this threshold are grouped under "Other". Default is 0.02.
    
    Returns:
    -------
    dict
        A dictionary mapping labels to color values, suitable for consistent plotting.
    """
    LANGUAGE_LABEL_MAP = {
        "Jupyter Notebook": "Jupyter",
    }

    # Aggregate all column values across all universities
    combined_series = pd.concat([
        df[column].replace(LANGUAGE_LABEL_MAP)
        for df in all_data_dict.values()
    ])

    total = len(combined_series)
    value_counts = combined_series.value_counts(dropna=True)

    # Keep only labels that exceed the threshold globally
    major_labels = value_counts[value_counts / total >= threshold].index.tolist()

    # Sort and finalize label list
    unique_labels = sorted(label for label in major_labels if pd.notnull(label))
    unique_labels.append("Other")
    if column == "license":
        unique_labels.append("None")
    # Generate color map
    cmap = matplotlib.colormaps.get_cmap('tab10').resampled(len(unique_labels))
    return dict(zip(unique_labels, [cmap(i) for i in range(len(unique_labels))]))
