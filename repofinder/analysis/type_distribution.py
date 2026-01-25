#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_type_distribution(
    filtered_data, acronym, ax=None, color_map=None, 
    title_prefix=None, label_size=25, title_size=24, textprops=15):
    """
    Plots a pie chart representing the distribution of repository types based on GPT-predicted categories.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        The filtered dataset containing the 'type_prediction_gpt_5_mini' column.

    acronym : str
        Acronym for the institution or group being plotted.

    ax : matplotlib.axes.Axes, optional
        An existing matplotlib axis to plot on. If None, creates a new figure.

    color_map : dict, optional
        A dictionary mapping category labels to colors.

    title_prefix : str, optional
        Text to prefix the plot title with.

    Returns
    -------
    None
        This function generates a pie chart but does not return any values.
    """
    total_repositories = len(filtered_data)
    category_counts = filtered_data['type_prediction_gpt_5_mini'].value_counts()

    labels = category_counts.index.tolist()

    cmap = cm.get_cmap('tab20')
    category_colors = {cat: cmap(i) for i, cat in enumerate(labels)}
    colors = [category_colors[cat] for cat in labels]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        category_counts,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': textprops}
    )
        
    for i, text in enumerate(texts):
        label = text.get_text().strip().upper()
    
        if label == "DOCS":
            # Move label lower
            x, y = text.get_position()
            text.set_position((x, y - 0.1))
    
            # Move percentage text slightly lower too
            x_pct, y_pct = autotexts[i].get_position()
            autotexts[i].set_position((x_pct, y_pct - 0.08))


    if title_prefix:
        ax.set_title(
            rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="left",
            pad=20
        )
    else:
        ax.set_title(
            rf"$\bf{{UC\ Project\ Type\ Distribution}}$ (Total: {total_repositories})",
            fontsize=title_size,
            loc="center",
            pad=20
        )
