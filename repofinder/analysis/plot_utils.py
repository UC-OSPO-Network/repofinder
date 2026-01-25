#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import string
import matplotlib
import random
import json

acronyms = [
    "UCB", "UCD", "UCI", "UCLA", "UCM",
    "UCR", "UCSB", "UCSC", "UCSD", "UCSF"
]

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



def get_acronym_domain_map():
    acronym_domain = {}
    for acronym in acronyms:
        config_path = f"config/config_{acronym}.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            domain = config.get("UNIVERSITY_EMAIL_DOMAIN")
            if domain:
                acronym_domain[acronym] = domain
    return acronym_domain


def db_to_df_filtered(db_path, acronym, db_type='sqlite', db_params=None):
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

    acronym_domain = get_acronym_domain_map()

    # Load repositories and organizations
    repo_df = pd.read_sql_query("SELECT * FROM repositories", conn)
    org_df = pd.read_sql_query("SELECT login, url AS org_url, email AS org_email FROM organizations", conn)
    
    # Merge on owner == login
    merged_df = repo_df.merge(org_df, how='left', left_on='owner', right_on='login')

    mask = (
    merged_df['org_email'].fillna('').str.contains(acronym_domain[acronym], regex=False)
    | merged_df['org_url'].fillna('').str.contains(acronym_domain[acronym], regex=False)
    )
    merged_df = merged_df[mask]

    # Clean up
    merged_df['subscribers_count'] = merged_df['subscribers_count'].fillna(0).astype(int)
    merged_df['release_downloads'] = merged_df['release_downloads'].fillna(0).astype(int)
    merged_df.drop(columns=['login'], inplace=True)  # Remove redundant column after merge

    return merged_df




def filter_data(data, threshold):
    """
    Filter data based on prediction threshold and repository characteristics.

    Filters the input data to include only rows where:
    - The 'ai_prediction' column is greater than the threshold
    - Repository size is greater than 0
    - Repository is not archived
    - Repository is not a fork
    - Repository has at least one star

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing repository data.
    threshold : float
        Minimum prediction value to include (repositories with ai_prediction > threshold).

    Returns
    -------
    pandas.DataFrame
        The filtered dataset with valid repositories and predictions above threshold.
    """
    # Convert ai_prediction to numeric
    data['affiliation_prediction_gpt_5_mini'] = pd.to_numeric(
        data['affiliation_prediction_gpt_5_mini'],
        errors="coerce"
    )
    
    # Start with prediction threshold filter
    filtered = data[data['affiliation_prediction_gpt_5_mini'] > threshold].copy()
    
    # Filter out repositories with size=0 (but keep size=NULL, matching SQL: size > 0 OR size IS NULL)
    if 'size' in filtered.columns:
        filtered['size'] = pd.to_numeric(filtered['size'], errors="coerce")
        filtered = filtered[(filtered['size'] > 0) | (filtered['size'].isna())]
    
    # Filter out archived repositories (keep only non-archived: 0, False, or NaN)
    if 'archived' in filtered.columns:
        # Handle boolean, integer, and string representations
        # Convert to numeric, coercing errors (strings become NaN)
        archived_numeric = pd.to_numeric(filtered['archived'], errors="coerce")
        # Also check for boolean False directly
        archived_bool = filtered['archived'] == False
        # Keep rows where archived is 0, False, or NaN (missing)
        filtered = filtered[(archived_numeric == 0) | (archived_bool) | (archived_numeric.isna())]
    
    # Filter out fork repositories (keep only non-forks: 0, False, or NaN)
    if 'fork' in filtered.columns:
        # Handle boolean, integer, and string representations
        # Convert to numeric, coercing errors (strings become NaN)
        fork_numeric = pd.to_numeric(filtered['fork'], errors="coerce")
        # Also check for boolean False directly
        fork_bool = filtered['fork'] == False
        # Keep rows where fork is 0, False, or NaN (missing)
        filtered = filtered[(fork_numeric == 0) | (fork_bool) | (fork_numeric.isna())]
    
    # Filter out template repositories (keep only non-templates: 0, False, or NaN)
    if 'is_template' in filtered.columns:
        # Handle boolean, integer, and string representations
        # Convert to numeric, coercing errors (strings become NaN)
        template_numeric = pd.to_numeric(filtered['is_template'], errors="coerce")
        # Also check for boolean False directly
        template_bool = filtered['is_template'] == False
        # Keep rows where is_template is 0, False, or NaN (missing)
        filtered = filtered[(template_numeric == 0) | (template_bool) | (template_numeric.isna())]
    
    # Filter out repositories with no stars (keep only repositories with stars > 0)
    # if 'stargazers_count' in filtered.columns:
    #     filtered['stargazers_count'] = pd.to_numeric(filtered['stargazers_count'], errors="coerce")
    #     filtered = filtered[filtered['stargazers_count'] > 0]
    
    return filtered.reset_index(drop=True)


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

    if threshold > 0:
        unique_labels.append("Other")
    if column == "license":
        unique_labels.append("None")
    # Generate color map
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(unique_labels))
    return dict(zip(unique_labels, [cmap(i) for i in range(len(unique_labels))]))


