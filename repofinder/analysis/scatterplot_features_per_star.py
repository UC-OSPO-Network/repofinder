#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def plot_feature_presence_by_stars_grid(
    df, features, star_col='stargazers_count', max_stars=1000,
    bins=5, figsize=(18, 5), tick_size=16,
    label_size=20, title_size=24, annotations_size=16
    ):
        
    """
    Plot the percentage of repositories with specific features across star count bins.
    
    This function creates a grid of scatter plots, one for each feature, showing the
    percentage of repositories containing that feature within predefined star count bins.
    A linear regression line is included to visualize trends.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing repository metadata, including star counts and feature presence.
    features : list of str
        List of column names corresponding to features (e.g., community files) to evaluate.
    star_col : str, default='stargazers_count'
        Column in `df` representing the number of stars.
    max_stars : int, default=1000
        Maximum number of stars to consider; repositories with more stars are filtered out.
    bins : int, default=5
        Number of bins to divide the star count range into.
    figsize : tuple of int, default=(18, 5)
        Size of the entire figure.
    tick_size : int, default=16
        Font size for tick labels.
    label_size : int, default=20
        Font size for axis labels and subplot titles.
    title_size : int, default=24
        Font size for the overall figure title.
    annotations_size : int, default=16
        (Currently unused) Size for annotations on the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    """
    df = df.copy()
    df = df[df[star_col] <= max_stars]
    total_repositories = len(df)

    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    # Bin all repos to get total counts per bin
    df['star_bin'] = pd.cut(df[star_col], bins=bins, right=False)
    total_counts = df.groupby('star_bin', observed=True).size()

    # Precompute bin midpoints once
    bin_centers = np.array([interval.left + (interval.right - interval.left) / 2 for interval in total_counts.index])

    for i, feature in enumerate(features):
        ax = axes[i]

        # Select repos with the feature present
        df_feature = df[df[feature].notna()]

        # Count repos with feature per star bin
        feature_counts = df_feature.groupby('star_bin', observed=True).size()

        # Compute percentage (handle bins with zero total count)
        percentages = (feature_counts / total_counts * 100).reindex(total_counts.index, fill_value=0)

        ax.scatter(bin_centers, percentages, alpha=0.7)

        # Linear regression line
        slope, intercept, r_value, p_value, std_err = linregress(bin_centers, percentages)
        line_x = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        line_y = intercept + slope * line_x
        ax.plot(line_x, line_y, color='red', linestyle='--')

        ax.set_title(feature.replace("_", " ").title(), fontsize=label_size)
        ax.set_xlabel("# Stars", fontsize=label_size)
        ax.set_ylabel("Percentage with Feature", fontsize=label_size)
        tick_interval = max_stars // 5  # adjust granularity here
        xticks = np.arange(0, max_stars + 1, tick_interval)
        ax.set_xticks(xticks)
        ax.set_xlim(0, max_stars)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.grid(True)
        
    suptitle = (
        r"$\bf{Percentage\ of\ Community\ Files\ by\ Number\ of\ Stars\ }$" +
        r"$\bf{DEV\ Repositories}$" + f" (Total: {total_repositories})"
    )
    fig.suptitle(suptitle, fontsize=title_size)


    return fig


def plot_avg_feature_presence_by_stars(
    df, features, star_col='stargazers_count', max_stars=1000,
    bins=20, figsize=(8, 5), tick_size=16,
    label_size=20, title_size=22
    ):
    
    """
    Plot the average percentage of repositories with given features across star count bins.
    
    This function computes the average presence of several features across star bins
    and visualizes the trend in a single scatter plot with a linear regression line.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing repository metadata, including star counts and feature presence.
    features : list of str
        List of column names corresponding to features (e.g., community files) to average.
    star_col : str, default='stargazers_count'
        Column in `df` representing the number of stars.
    max_stars : int, default=1000
        Maximum number of stars to consider; repositories with more stars are filtered out.
    bins : int, default=20
        Number of bins to divide the star count range into.
    figsize : tuple of int, default=(8, 5)
        Size of the figure.
    tick_size : int, default=16
        Font size for tick labels.
    label_size : int, default=20
        Font size for axis labels.
    title_size : int, default=22
        Font size for the plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    """
    df = df.copy()
    df = df[df[star_col] <= max_stars]
    total_repositories = len(df)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Bin all repos to get total counts per bin
    df['star_bin'] = pd.cut(df[star_col], bins=bins, right=False)
    total_counts = df.groupby('star_bin', observed=True).size()

    # Precompute bin midpoints once
    bin_centers = np.array([
        interval.left + (interval.right - interval.left) / 2
        for interval in total_counts.index
    ])

    # Initialize a DataFrame to hold percentage values per feature
    percentages_per_feature = []

    for feature in features:
        df_feature = df[df[feature].notna()]
        feature_counts = df_feature.groupby('star_bin', observed=True).size()
        percentages = (feature_counts / total_counts * 100).reindex(total_counts.index, fill_value=0)
        percentages_per_feature.append(percentages)

    # Compute average percentage across features
    avg_percentages = pd.concat(percentages_per_feature, axis=1).mean(axis=1)

    # Scatter plot
    ax.scatter(bin_centers, avg_percentages, alpha=0.7)

    # Linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(bin_centers, avg_percentages)
    line_x = np.linspace(bin_centers.min(), bin_centers.max(), 100)
    line_y = intercept + slope * line_x
    ax.plot(line_x, line_y, color='red', linestyle='--')

    # Styling
    
    title = (
        r"$\bf{UC\ Average\ Community\ File\ Presence\ }$" + "\n" +
        r"$\bf{DEV\ Repos}$" + f" (Total: {total_repositories})"
    )
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("# Stars", fontsize=label_size)
    ax.set_ylabel("Average % with Feature", fontsize=label_size)
    tick_interval = max_stars // 5
    xticks = np.arange(0, max_stars + 1, tick_interval)
    ax.set_xticks(xticks)
    ax.set_xlim(0, max_stars)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(True)

    return fig
