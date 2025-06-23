#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:01:21 2025

@author: juanitagomez
"""

import matplotlib.pyplot as plt

def plot_language_distribution(filtered_data, acronym, ax=None, color_map=None, title_prefix=None):
    """
    Plots pie charts representing the distribution of programming languages used in the projects:
    - Grouped language distribution (major languages with at least 2% usage).
    - Minor language distribution (languages with less than 2% usage).

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        The filtered dataset containing the project languages.

    Returns
    -------
    None
        This function generates and saves pie charts but does not return any values.
    """
    total_repositories = len(filtered_data)
    # Shorten long language names
    LANGUAGE_LABEL_MAP = {
        "Jupyter Notebook": "Jupyter",
    }

    # Apply replacements
    filtered_data['language'] = filtered_data['language'].replace(LANGUAGE_LABEL_MAP)
    language_counts = filtered_data['language'].value_counts()
    total_languages = language_counts.sum()

    lang_major = language_counts[language_counts / total_languages >= 0.05].copy()
    lang_minor = language_counts[language_counts / total_languages < 0.05].copy()

    lang_grouped = lang_major.copy()
    if not lang_minor.empty:
        lang_grouped['Other'] = lang_minor.sum()

    labels = lang_grouped.index.tolist()
    colors = [color_map.get(label, "#FF00FF") for label in labels]  # hot pink if missing

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        lang_grouped,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 20}
    )

    for text in texts + autotexts:
        text.set_fontsize(17)

    #ax.set_title(f"{title_prefix} {acronym} (Total: {total_repositories})", fontsize=25, weight='bold', loc="left")    
    
    ax.set_title(
    rf"$\bf{{{title_prefix}\ {acronym}}}$ (Total: {total_repositories})",
    fontsize=28,
    loc="left"
    )
    
    # # Language Distribution - Minor Plot
    # if not lang_minor.empty:
    #     total_repositories = lang_minor.sum()
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     wedges, texts, autotexts = ax.pie(lang_minor, labels=lang_minor.index, autopct='%1.1f%%', startangle=140)
    #     for text in texts:
    #         text.set_fontsize(8)
    #     for i, autotext in enumerate(autotexts):
    #         autotext.set_fontsize(8)
    #         percentage = (lang_minor.iloc[i] / total_languages) * 100
    #         autotext.set_text(f'{percentage:.1f}%')
    #     ax.set_title(f"{acronym.upper()} Language Distribution (Minor Categories) — Total Repositories: {total_repositories}")
    #     plt.savefig(f'plots/{acronym}/language_distribution_minor.png', dpi=300, bbox_inches='tight')
    #     plt.close()
