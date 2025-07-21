#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_feature_heatmap_by_star_bucket(
    df, features, star_col='stargazers_count', title=None, ax=None,
    cmap="RdYlGn", annotate=True, fmt=".0f",
    label_size=22, title_size=23, annotations_size=16
    ):
    """
    Plot a heatmap showing the percentage of repositories that include specific features,
    grouped by star count buckets.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing repository data.
        features (list of str): List of feature column names to include in the heatmap.
        star_col (str): Column name representing the number of stars. Default is 'stargazers_count'.
        title (str, optional): Custom plot title. If None, a default title is generated.
        ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, a new figure is created.
        cmap (str): Color map for the heatmap. Default is 'RdYlGn'.
        annotate (bool): Whether to annotate heatmap cells with values. Default is True.
        fmt (str): Format for annotation values. Default is '.0f'.
        label_size (int): Font size for axis labels. Default is 22.
        title_size (int): Font size for the plot title. Default is 23.
        annotations_size (int): Font size for heatmap annotations. Default is 16.
    
    Returns:
        matplotlib.axes.Axes: The axis object with the heatmap.
    """

    total_repositories = len(df)

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

    # Define star buckets
    def get_star_bucket(stars):
        if stars <= 10:
            return '0–10'
        elif stars <= 50:
            return '11–50'
        elif stars <= 100:
            return '51–100'
        elif stars <= 200:
            return '101–200'
        else:
            return '>200'

    df = df.copy()
    df["star_bucket"] = df[star_col].apply(get_star_bucket)
    star_buckets = ['0–10', '11–50', '51–100', '101–200', '>200']

    results = []
    for feature in features:
        row = {"Feature": feature_display_names.get(feature, feature)}
        for bucket in star_buckets:
            subset = df[df["star_bucket"] == bucket]
            total = len(subset)
            count = subset[feature].notna().sum()
            pct = (count / total * 100) if total > 0 else 0
            row[bucket] = pct
        results.append(row)

    # Optional: Add row for average across features
    sum_row = {"Feature": "Average"}
    for bucket in star_buckets:
        bucket_avg = sum(row[bucket] for row in results) / len(results)
        sum_row[bucket] = bucket_avg
    results.append(sum_row)

    # Build and plot heatmap
    heatmap_df = pd.DataFrame(results).set_index("Feature")
    heatmap_df = heatmap_df[star_buckets]  # ensure column order

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    heatmap = sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=fmt,
        cbar_kws={'label': '% with feature', 'extend': 'neither'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={"fontsize": annotations_size} 
    )

    ax.set_xlabel("# Star Bucket", fontsize=label_size, labelpad=15)
    ax.set_ylabel("Community File", fontsize=label_size)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    
    title = (
        rf"$\bf{{UC\ Community\ Files\ by\ \#\ Stars}}$" + "\n" +
        rf"$\bf{{DEV\ Repos}}$" + f" (Total: {total_repositories})"
    )
    ax.set_title(
        title,
        fontsize=title_size,
        loc="center",
        pad=20
    )
    ax.tick_params(axis='both', which='both', length=0)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=annotations_size)
    cbar.set_label('% with feature', fontsize=label_size)

    return ax
