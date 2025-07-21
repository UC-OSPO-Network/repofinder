#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from repofinder.analysis.feature_counts_per_type import plot_feature_counts_per_type
from repofinder.analysis.feature_heatmap_per_stars import plot_feature_heatmap_by_star_bucket
from repofinder.analysis.language_distribution_by_type import plot_language_distribution_by_type
from repofinder.analysis.license_distribution_by_type import plot_license_distribution_by_type
from repofinder.analysis.plot_utils import filter_data, db_to_df
from repofinder.analysis.scatterplot_features_per_star import (
    plot_feature_presence_by_stars_grid,
    plot_avg_feature_presence_by_stars
)
from repofinder.analysis.type_distribution import plot_type_distribution

# Set default font
matplotlib.rcParams['font.family'] = 'Lato'

def plot_analysis_by_type_combined(all_data_dict):
    """
    Generate and save grouped bar plots for language usage, license types,
    and repository feature presence across multiple universities.

    Parameters
    ----------
    all_data_dict : dict of {str: pd.DataFrame}
        A dictionary mapping university acronyms (e.g., "UCSB", "UCSC", "UCSD") to their
        corresponding repository metadata DataFrames.
    
    Notes
    -----
    - Thresholds for filtering repositories differ by university and are hardcoded within the function.
    - Uses `filter_data`, `plot_language_distribution`, `plot_license_distribution`, 
      `plot_feature_counts`, and `build_shared_color_map` as helper functions.
    - Output files:
        - `language_distribution_grouped.png`
        - `license_distribution_grouped.png`
        - `feature_counts.png`
    """
    os.makedirs('plots/combined', exist_ok=True)

    thresholds = {"UCSB": 0.3, "UCSC": 0.6, "UCSD": 0.3}
    
    
    # TYPE DISTRIBUTION
    
    filtered_dfs = []
    common_cols = None
    
    for i, (acronym, df) in enumerate(all_data_dict.items()):
        if df is None or df.empty:
            print(f"Skipping {acronym} — empty DataFrame.")
            continue
    
        threshold = thresholds.get(acronym, 0.5)
    
        filtered = filter_data(df, threshold)
        filtered["University"] = acronym

        # Track common columns
        if common_cols is None:
            common_cols = set(filtered.columns)
        else:
            common_cols &= set(filtered.columns)

        filtered_dfs.append(filtered)
    
    # Align and combine
    if filtered_dfs and common_cols:
        common_cols = list(common_cols) + ["University"]
        merged_filtered_df = pd.concat([df[common_cols] for df in filtered_dfs], ignore_index=True)
    else:
        merged_filtered_df = pd.DataFrame()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_type_distribution(
        merged_filtered_df, 
        acronym="", 
        ax=ax, 
        label_size=22, 
        title_size=20)
    
    plt.savefig("plots/combined/all_type_distribution.png", dpi=300)
    plt.tight_layout()
    #plt.show()
    plt.close()
    
    # LICENSE DISTRIBITION
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_license_distribution_by_type(
        merged_filtered_df, 
        acronym="", 
        ax=ax, 
        label_size=22, 
        title_size=23,
        other_thres=0.009)
    plt.tight_layout()
    plt.savefig("plots/combined/all_license_distribution.png", dpi=300)
    #plt.show()
    plt.close()
    
    # LANGUAGE DISTRIBUTION
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_language_distribution_by_type(
        merged_filtered_df, 
        acronym="", 
        ax=ax, 
        label_size=22, 
        title_size=23,
        other_thres=0.02)
    plt.tight_layout()
    plt.savefig("plots/combined/all_language_distribution.png", dpi=300)
    #plt.show()
    plt.close()
    
    features = [
        'description', 'readme', 'license', 'code_of_conduct',
        'contributing', 'security_policy', 'issue_templates', 'pull_request_template'
    ]
    
    # FEATURE COUNT
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_feature_counts_per_type(
        merged_filtered_df, 
        features,
        acronym="", 
        ax=ax, 
        label_size=22, 
        title_size=23)
    plt.tight_layout()
    plt.savefig("plots/combined/all_features_distribution.png", dpi=300)
    #plt.show()
    plt.close()
    
    # FEATURE COUNT BY STARS
    
    dev_df = merged_filtered_df[merged_filtered_df['gpt_category'] == 'DEV']
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_feature_heatmap_by_star_bucket(
        dev_df,
        features,
        star_col='stargazers_count',
        ax=ax
    )
    plt.tight_layout()
    plt.savefig("plots/combined/all_heat_map_stars_features.png", dpi=300)
    #plt.show()
    plt.close()
    
    # FEATURE SCATTERPLOT 
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_avg_feature_presence_by_stars(
        dev_df,
        features,
        star_col='stargazers_count',
    )
    
    plt.tight_layout()
    plt.savefig("plots/combined/all_scatterplot_combined_features.png", dpi=300)
    #plt.show()
    plt.close()
    
    # FEATURE SCATTERPLOT GRID
    
    features = [
         'description', 'license', 'contributing','issue_templates'
    ]
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_feature_presence_by_stars_grid(
        dev_df,
        features,
        star_col='stargazers_count',
    )
    
    plt.tight_layout()
    plt.savefig("plots/combined/all_scatterplot_features.png", dpi=300)
    #plt.show()
    plt.close()




def get_all_data(acronyms = ['UCB', 'UCD', 'UCI', 'UCLA', 'UCM', 'UCR', 'UCSB', 'UCSC', 'UCSD', 'UCSF']):
    all_data = {}
    for acronym in acronyms:
        path = f"Data/db/repository_data_{acronym}_database.db"
        data = db_to_df(path, acronym)
        all_data[acronym] = data
    return all_data


if __name__ == "__main__":
    all_data = get_all_data()
    plot_analysis_by_type_combined(all_data)

