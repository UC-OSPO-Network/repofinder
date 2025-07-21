#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import pandas as pd



def plot_feature_counts_per_type(
    filtered_data, features, acronym="", top=False, ax=None, title_prefix="",
    order=None, feature_colors=None, ylim=None, hide_ylabel=False,
    label_size=25, title_size=24
    ):
    """
    Plot a stacked bar chart of feature presence counts across GPT-predicted project types.

    This function visualizes how often specific repository features (e.g., README, license, contributing guide)
    appear across different project types predicted by a GPT model. Each feature is represented by a stacked bar,
    segmented by project type. A special bar labeled "Project Type" can be included to show the total number of
    repositories per project type. Percentage labels are displayed above each bar, and a legend identifies
    the project type segments.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame that includes columns for the specified `features` and a 'gpt_category' column
        representing predicted project types.
    features : list of str
        List of feature column names to include in the plot (e.g., 'readme', 'license').
    acronym : str, optional
        Acronym to include in the saved file name and title (e.g., institution name).
    top : bool, default=False
        If True, saves the plot as a PNG file in the 'plots/{acronym}' directory.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.
    title_prefix : str, optional
        Custom text to prepend to the plot title.
    order : list of str, optional
        Custom order for the features on the x-axis. If None, features are ordered by total count.
    feature_colors : dict, optional
        (Currently unused) Dictionary mapping features to specific colors.
    ylim : tuple of float, optional
        (Currently unused) Custom y-axis limits.
    hide_ylabel : bool, default=False
        Whether to hide the y-axis label and ticks.
    label_size : int, default=25
        Font size for axis labels.
    title_size : int, default=24
        Font size for the plot title.

    Returns
    -------
    tuple
        order : list of str
            The order of feature labels used in the x-axis.
        category_colors : dict
            Mapping from GPT-predicted project types to color values used in the plot.
    """

    feature_display_names = {
        'description': 'Description',
        'readme': 'README',
        'license': 'License',
        'code_of_conduct': 'Code of Conduct',
        'contributing': 'Contributing Guide',
        'security_policy': 'Security Policy',
        'issue_templates': 'Issue Templates',
        'pull_request_template': 'PR Template',
        'gpt_category': "Project Type"
        
    }

    # Validate the gpt_category column
    if "gpt_category" not in filtered_data.columns:
        raise ValueError("filtered_data must contain a 'gpt_category' column")

    # Map original feature names to display names
    data = filtered_data.copy()
    data = data[features + ['gpt_category']]
    data = data.rename(columns=feature_display_names)

    # Compute grouped counts
    grouped = data.groupby('Project Type').apply(lambda g: g.notna().sum())
    grouped = grouped.T  # Features as rows, categories as columns

    if order is None:
        total_counts = grouped.sum(axis=1)
        order = total_counts.sort_values().index.tolist()

    grouped = grouped.loc[order]

    # Default colors for each gpt_category
    category_list = grouped.columns.tolist()
    cmap = plt.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(category_list)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Compute x positions with spacing before 'Project Type'
    gap = 0.6
    x_positions = []
    for i, feature in enumerate(order):
        if feature == "Project Type":
            x_positions.append(i + gap)
        else:
            x_positions.append(len(x_positions))

    # Initialize bottom values aligned with x_positions
    bottoms_array = [0] * len(order)
    used_labels = set()
    
    for category in category_list:
        for i, feature in enumerate(order):
            value = grouped.loc[feature, category]
            if value > 0:
                ax.bar(
                    x_positions[i], value,
                    bottom=bottoms_array[i],
                    label=category if category not in used_labels else "",
                    color=category_colors[category],
                    width=0.8
                )
                bottoms_array[i] += value
                used_labels.add(category)

    total_repositories = len(filtered_data)

    # Add percentage labels on top of each full bar
    for i, feature in enumerate(order):
        count = grouped.loc[feature].sum()
        percent = (count / total_repositories) * 100
        ax.annotate(f'{percent:.1f}%',
                    (x_positions[i], count + total_repositories * 0.03),
                    ha='center', va='bottom', fontsize=14, color='black')


    # Draw a vertical dashed line to separate 'Project Type'
    if "Project Type" in order:
        pt_index = order.index("Project Type")
        pt_x = x_positions[pt_index] - 0.8
        ax.axvline(x=pt_x, linestyle='--', color='gray', linewidth=2)

    # Formatting
    
    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{UC\ Community\ Files}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
    
    ax.set_xlabel("Community File", fontsize=label_size)
    
    if not hide_ylabel:
        ax.set_ylabel("Repository Count", fontsize=label_size)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    max_height = max(bottoms_array)
    ax.set_ylim(0, max_height + total_repositories * 0.13)
    ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.8)
    ax.legend(title="Project Type", fontsize=14, title_fontsize=16)

    if top:
        plt.savefig(f'plots/{acronym}/CountPerFeatureTop_Stacked.png', dpi=300)

    return order, category_colors
