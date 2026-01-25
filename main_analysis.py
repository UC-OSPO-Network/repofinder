#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from repofinder.analysis.language_distribution import plot_language_distribution
from repofinder.analysis.license_distribution import plot_license_distribution
from repofinder.analysis.feature_counts import plot_feature_counts
from repofinder.analysis.plot_utils import build_shared_color_map, filter_data, db_to_df
import matplotlib.pyplot as plt
import os
import string
import matplotlib

matplotlib.rcParams['font.family'] = 'Lato'

def plot_analysis(all_data_dict):
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

    # LANGUAGE DISTRIBUTION
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lang_color_map = build_shared_color_map(all_data_dict, column='language')
    
    for i, (acronym, df) in enumerate(all_data_dict.items()):
        letter = string.ascii_lowercase[i]
        title_prefix = f"({letter}) "
        filtered = filter_data(df, threshold=thresholds[acronym])
        plot_language_distribution(filtered, acronym, ax=axes[i], color_map=lang_color_map, title_prefix=title_prefix)
    plt.tight_layout()
    plt.savefig("plots/combined/language_distribution_grouped.png", dpi=300)
    plt.show()
    plt.close()

    # LICENSE DISTRIBUTION
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lang_color_map = build_shared_color_map(all_data_dict, column='license')
    for i, (acronym, df) in enumerate(all_data_dict.items()):
        letter = string.ascii_lowercase[i]
        title_prefix = f"({letter}) "
        filtered = filter_data(df, threshold=thresholds[acronym])
        plot_license_distribution(filtered, acronym, ax=axes[i], color_map=lang_color_map, title_prefix=title_prefix)
    plt.tight_layout()
    plt.savefig("plots/combined/license_distribution_grouped.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # FEATURE COUNT
    order, feature_colors = None, None
    
    # Step 1: Determine maximum y-limit across all universities
    max_count = 0
    
    for i, (acronym, df) in enumerate(all_data_dict.items()):
        filtered = filter_data(df, threshold=thresholds[acronym])
        count = filtered[
            ['description', 'readme', 'license', 'code_of_conduct_file',
             'contributing', 'security_policy', 'issue_templates', 'pull_request_template']
        ].notna().sum().max()
        max_count = max(max_count, count)
    
    ylim = max_count + int(max_count * 0.09)  # Add 9% headroom for percentage labels
    
    # Step 2: Plot with shared scale and controlled y-axis labels
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    for i, (acronym, df) in enumerate(all_data_dict.items()):
        letter = string.ascii_lowercase[i]
        title_prefix = f"({letter})"
        filtered = filter_data(df, threshold=thresholds[acronym])
        order, feature_colors = plot_feature_counts(
            filtered,
            acronym,
            ax=axes[i],
            title_prefix=title_prefix,
            order=order,
            feature_colors=feature_colors,
            ylim=ylim,
            hide_ylabel=(i != 0)
        )
    plt.savefig("plots/combined/feature_counts.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def get_all_data(acronyms = ['UCSB', 'UCSC', 'UCSD']):
    all_data = {}
    for acronym in acronyms:
        path = f"Data/db/repository_data_{acronym}_database.db"
        data = db_to_df(path, acronym)
        all_data[acronym] = data
    return all_data


if __name__ == "__main__":
    all_data = get_all_data()
    plot_analysis(all_data)


