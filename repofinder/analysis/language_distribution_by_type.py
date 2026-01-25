#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
 
def plot_language_distribution_by_type(
    filtered_data, acronym="", ax=None, color_map=None,
    title_prefix="", hide_ylabel=False, language_order=None,
    ylim=None, label_size=25, title_size=24, props=12, other_thres=0.02,
    legend_size=None
    ):
    """
    Plot a stacked bar chart showing programming language distribution across GPT-predicted project types.

    This function visualizes how programming languages are distributed among project types
    (as predicted by a GPT model). Each language is represented as a stacked bar where
    each segment corresponds to a project type. An additional bar labeled "Project Type"
    on the right shows the total number of repositories per project type, independent of language.
    Languages that appear in fewer than a specified percentage of repositories are grouped
    into an "Other" category.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame containing at least the columns 'language' and 'type_prediction_gpt_5_mini'.
    acronym : str, optional
        Acronym to include in the plot title (e.g., a university or organizational abbreviation).
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure and axes are created.
    color_map : dict, optional
        Mapping from project types to colors. If None, a default colormap is used.
    title_prefix : str, optional
        Custom prefix to display before the title (e.g., institution name).
    hide_ylabel : bool, default=False
        Whether to hide the y-axis label while keeping the ticks and labels visible.
    language_order : list of str, optional
        Custom order for displaying languages on the x-axis. If None, languages are sorted by total count.
    ylim : tuple of float, optional
        Manual setting for y-axis limits. If None, limits are determined automatically.
    label_size : int, default=25
        Font size for axis labels.
    title_size : int, default=24
        Font size for the plot title.
    other_thres : float, default=0.02
        Minimum threshold (as a proportion of total repositories) for a language to be shown as its own bar.
        Languages below this threshold are grouped into the "Other" category.
    legend_size : int, optional
        Font size for legend text and title. If None, uses label_size.

    Returns
    -------
    list of str
        List of language labels in the order they were plotted, including "Project Type" and potentially "Other".

    Notes
    -----
    - Minor language labels are grouped under "Other" if they fall below the `other_thres` threshold.
    - The function modifies 'Jupyter Notebook' to 'Jupyter NB' and standardizes 'other' to 'Other'.
    - Percentage labels are added above each full bar to show the relative contribution to the dataset.
    """
    total_repositories = len(filtered_data)

    # Clean and standardize language names
    data = filtered_data.copy()
    data['language'] = data['language'].fillna('None')
    data['language'] = data['language'].replace({
        'other': 'Other',
        'Jupyter Notebook': 'Jupyter NB'})

    # Count languages per project type
    grouped = data.groupby('language')['type_prediction_gpt_5_mini'].value_counts().unstack(fill_value=0)

    # Filter major languages (at least 2% globally)
    language_totals = data['language'].value_counts()
    total_languages = language_totals.sum()
    major_languages = language_totals[language_totals / total_languages >= other_thres].index.tolist()

    # Combine minor ones into "Other"
    grouped = grouped.copy()
    
    # Drop rows that will be grouped as 'Other'
    minor_rows = grouped.drop(index=major_languages, errors='ignore')
    grouped = grouped.loc[major_languages]
    
    # Add 'Other' if there are any minor rows
    if not minor_rows.empty:
        grouped.loc['Other'] = minor_rows.sum()
    grouped = grouped.fillna(0)

    # If a consistent language order is provided, use it
    if language_order is not None:
        grouped = grouped.reindex(language_order, fill_value=0)
    else:
        # Sort languages by ascending total counts (sum of all categories)
        total_counts = grouped.sum(axis=1)
        grouped = grouped.loc[total_counts.sort_values().index]
    
    # Compute the project type totals (sum over all languages)
    project_type_totals = filtered_data['type_prediction_gpt_5_mini'].value_counts()
    project_type_totals.name = 'Project Type'

    # Append project type totals as a new "bar" on the right
    grouped_with_pt = pd.concat([grouped, project_type_totals.to_frame().T])

    language_labels = grouped_with_pt.index.tolist()

    # Project type list and colormap
    category_list = grouped_with_pt.columns.tolist()
    cmap = plt.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(category_list)}

    num_languages = len(grouped)
    gap = 1.2
    x_positions = list(range(num_languages)) + [num_languages - 1 + gap]

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
    for i, language_label in enumerate(language_labels):
        count = grouped_with_pt.loc[language_label].values.sum()
        percent = (count / total_repositories) * 100
        ax.annotate(
            f'{percent:.1f}%',
            (x_positions[i], count + total_repositories * 0.02),
            ha='center', va='bottom',
            fontsize=props, color='black'
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
            rf"$\bf{{UC\ Language\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )

    ax.set_xlabel("Language", fontsize=label_size)
    
    if not hide_ylabel:
        ax.set_ylabel("Repository Count", fontsize=label_size)
    else:
        ax.set_ylabel("")  # hide label only, keep ticks & labels visible
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(language_labels, rotation=45, ha='right', fontsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.tick_params(axis='x', labelsize=label_size)
    
    # Set y-limit once based on data max + margin
    max_height = max(bottoms)
    ax.set_ylim(0, max_height + total_repositories * 0.11)
    
    # Legend inside top-right corner of the plot
    legend_fontsize = legend_size if legend_size is not None else label_size
    ax.legend(
        title="Project Type",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc='upper left',
        frameon=True,
        framealpha=0.9
    )
    
    # Optional: draw vertical line to separate 'Project Type' bar visually
    pt_index = language_labels.index('Project Type')
    ax.axvline(pt_index - 0.4, color='gray', linestyle='--', linewidth=1.5)
    
    return language_labels  # Return order for consistent reuse
