#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
import pandas as pd
import matplotlib.pyplot as plt
import random

def plot_feature_counts(
    filtered_data, acronym, top=False, ax=None, title_prefix=None,
    order=None, feature_colors=None, ylim=None, hide_ylabel=False
    ):
    """
    Plot the presence of key repository features (e.g., README, license) as a bar chart.
    
    This function counts how many repositories in the given DataFrame contain each of a 
    predefined set of features.
    
    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame containing repository metadata with boolean or non-null indicators for features.
    acronym : str
        Acronym for the university or institution (used in the plot title and file name).
    top : bool, optional
        If True, saves the figure to a file in the `plots/{acronym}/` directory. Default is False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to draw the plot on. If None, a new figure and axes are created.
    title_prefix : str, optional
        Optional text prefix for the plot title (e.g., "UC").
    order : list of str, optional
        List specifying the order in which features should appear on the x-axis. If None, features 
        are sorted by count.
    feature_colors : dict, optional
        Dictionary mapping feature display names to colors. If None, a shuffled colormap is used.
    ylim : float, optional
        Y-axis upper limit to enforce consistent scale across plots. If None, axis scales automatically.
    hide_ylabel : bool, optional
        If True, hides the y-axis label and ticks (useful for subplots). Default is False.
    
    Returns
    -------
    order : list of str
        The order of features used in the plot.
    feature_colors : dict
        The mapping of feature display names to their corresponding colors used in the plot.
    """
    features = [
        'description', 'readme', 'license', 'code_of_conduct',
        'contributing', 'security_policy', 'issue_templates', 'pull_request_template'
    ]
    
    feature_display_names = {
        'description': 'Description',
        'readme': 'README',
        'license': 'License',
        'code_of_conduct': 'Code of Conduct',
        'contributing': 'Contributing Guide',
        'security_policy': 'Security Policy',
        'issue_templates': 'Issue Templates',
        'pull_request_template': 'PR Template'
    }

    # Count non-null values
    feature_counts_raw = filtered_data[features].notna().sum()
    total_repositories = len(filtered_data)

    # Map to display names
    feature_counts = pd.Series({feature_display_names[k]: v for k, v in feature_counts_raw.items()})

    # Use order from first university if provided
    if order is None:
        order = feature_counts.sort_values().index.tolist()

    # Set consistent colors across subplots
    if feature_colors is None:
        cmap = plt.get_cmap('tab10')
        palette = [cmap(i) for i in range(10)]
        random.seed("39")
        random.shuffle(palette) 
        palette = palette[0:8]
        feature_colors = dict(zip(order, palette))

    # Sort by defined order
    feature_counts = feature_counts[order]
    colors = [feature_colors[feat] for feat in order]

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    ax.grid(True)
    ax.set_axisbelow(True)
    # Plot bars with matplotlib
    bars = ax.bar(feature_counts.index, feature_counts.values, color=colors)

    # Add percentage labels on top of bars
    for bar in bars:
        count = bar.get_height()
        percent = (count / total_repositories) * 100
        ax.annotate(f'{percent:.1f}%', 
                    (bar.get_x() + bar.get_width() / 2, count + total_repositories * 0.03),
                    ha='center', va='bottom', fontsize=16, color='black')

    # Formatting
    ax.set_title(rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
                 fontsize=28, loc="left", pad=20)
    ax.set_xlabel("Feature", fontsize=25)
    if not hide_ylabel:
        ax.set_ylabel("Repository Count", fontsize=25)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)

    ax.set_xticks(range(len(feature_counts.index)))
    ax.set_xticklabels(feature_counts.index, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Set consistent ylim across plots
    if ylim:
        ax.set_ylim(0, ylim)
    ax.set_xlim(-0.5, len(features)-0.3)

    # Save
    if top:
        plt.savefig(f'plots/{acronym}/CountPerFeatureTop.png', dpi=300)

    return order, feature_colors
