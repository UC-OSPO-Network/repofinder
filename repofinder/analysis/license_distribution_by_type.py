#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 20:08:24 2025

@author: juanitagomez
"""

import matplotlib.pyplot as plt
import pandas as pd
 
def plot_license_distribution_by_type(
    filtered_data, acronym="", ax=None, color_map=None,
    title_prefix="", hide_ylabel=False, license_order=None,
    ylim=None, label_size=25, title_size=24, other_thres=0.02
):
    """
    Plot a stacked bar chart showing license distribution across GPT-predicted project types.
    
    This function visualizes how software licenses are distributed among project types
    (as classified by a GPT model). Each license is represented as a stacked bar where
    each segment corresponds to a project type. An additional bar labeled "Project Type"
    on the right shows the total number of repositories per project type, regardless of license.
    
    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame containing at least the columns 'license' and 'gpt_category'.
    acronym : str, optional
        Acronym to include in the plot title, e.g., a university or organization abbreviation.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes will be created.
    color_map : dict, optional
        A mapping from project types to colors. If None, a default color palette is used.
    title_prefix : str, optional
        A prefix for the plot title (e.g., the institution name).
    hide_ylabel : bool, default=False
        Whether to hide the y-axis label (but keep tick marks).
    license_order : list of str, optional
        Order in which to display licenses on the x-axis. If None, licenses are ordered by total count.
    ylim : tuple of float, optional
        Y-axis limits. If None, the limits are set automatically.
    label_size : int, default=25
        Font size for axis labels.
    title_size : int, default=24
        Font size for the plot title.
    other_thres : float, default=0.02
        Threshold for filtering minor licenses. Licenses representing less than this
        fraction of the total are grouped into an "Other" category.
    
    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object with the rendered bar chart.
    """
    total_repositories = len(filtered_data)

    # Clean and standardize license names
    data = filtered_data.copy()
    data['license'] = data['license'].fillna('None')

    # Count licenses per project type
    grouped = data.groupby('license')['gpt_category'].value_counts().unstack(fill_value=0)

    # Filter major licenses (at least 2% globally)
    license_totals = data['license'].value_counts()
    total_licenses = license_totals.sum()
    major_licenses = license_totals[license_totals / total_licenses >= other_thres].index.tolist()

    # Combine minor ones into "Other"
    grouped = grouped.copy()
    
    # Drop rows that will be grouped as 'Other'
    minor_rows = grouped.drop(index=major_licenses, errors='ignore')
    grouped = grouped.loc[major_licenses]
    
    # Add minor rows under 'Other'
    if not minor_rows.empty:
        grouped.loc['Other'] = minor_rows.sum()
    
    # Merge an existing 'other' row (if present) into 'Other'
    if 'other' in grouped.index:
        if 'Other' in grouped.index:
            grouped.loc['Other'] += grouped.loc['other']
        else:
            grouped.loc['Other'] = grouped.loc['other']
        grouped = grouped.drop('other')
    
    # Fill NaNs just in case
    grouped = grouped.fillna(0)

    total_counts = grouped.sum(axis=1)
    grouped = grouped.loc[total_counts.sort_values().index]
    
    # Compute the project type totals (sum over all licenses)
    project_type_totals = filtered_data['gpt_category'].value_counts()
    project_type_totals.name = 'Project Type'

    # Append project type totals as a new "bar" on the right
    grouped_with_pt = pd.concat([grouped, project_type_totals.to_frame().T])

    license_labels = grouped_with_pt.index.tolist()

    # Project type list and colormap
    category_list = grouped_with_pt.columns.tolist()
    cmap = plt.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(category_list)}

    num_licenses = len(grouped)
    gap = 1.2
    x_positions = list(range(num_licenses)) + [num_licenses - 1 + gap]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Stack bars by category
    bottoms = [0] * len(grouped_with_pt)
    for category in category_list:
        values = grouped_with_pt[category].values
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=category,
            color=category_colors[category],
            width=0.8
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    # Add percentage labels on top of each full bar (including the project type bar)
    for i, license_label in enumerate(license_labels):
        count = grouped_with_pt.loc[license_label].values.sum()
        percent = (count / total_repositories) * 100
        ax.annotate(
            f'{percent:.1f}%',
            (x_positions[i], count + total_repositories * 0.02),
            ha='center', va='bottom',
            fontsize=14, color='black'
        )

    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{UC\ License\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
    
    ax.set_xlabel("License", fontsize=label_size)
    
    if not hide_ylabel:
        ax.set_ylabel("Repository Count", fontsize=label_size)
    else:
        ax.set_ylabel("")  # hide label only, keep ticks & labels visible
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(license_labels, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    
    # Set y-limit once based on data max + margin
    max_height = max(bottoms)
    ax.set_ylim(0, max_height + total_repositories * 0.11)
    
    # Legend inside top-right corner of the plot
    ax.legend(
        title="Project Type",
        fontsize=14,
        title_fontsize=16,
        loc='upper left',
        frameon=True,
        framealpha=0.9
    )
    
    # Optional: draw vertical line to separate 'Project Type' bar visually
    pt_index = license_labels.index('Project Type')
    ax.axvline(pt_index - 0.4, color='gray', linestyle='--', linewidth=1.5)

