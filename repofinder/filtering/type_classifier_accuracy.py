#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def type_classifier_accuracy(test_file, predictions_file, acronym, output_prefix="classification_results"):
    """
    Evaluate GPT-based project type classification against manually labeled data.

    Parameters
    ----------
    test_file : str
        Path to the CSV file with manually labeled repository data (must contain 'html_url' and 'project_type').
    predictions_file : str
        Path to the CSV file with GPT predictions (must contain 'html_url', 'gpt_category', and optionally 'gpt_explanation').
    output_prefix : str
        Prefix for the output CSV files. Default is 'classification_results'.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-category and overall accuracy.
    """

    # Load and filter test data
    df_repos = pd.read_csv(test_file)
    df_repos = df_repos.drop(columns=["gpt_category"], errors="ignore")
    df_repos = df_repos[df_repos["project_type"].notnull()]

    # Load GPT predictions
    df_gpt = pd.read_csv(predictions_file)
    # Remove project_type column from predictions to avoid suffix conflicts during merge
    df_gpt = df_gpt.drop(columns=["project_type"], errors="ignore")

    # Debug: check for missing URLs
    missing_urls = set(df_repos["html_url"]) - set(df_gpt["html_url"])
    if missing_urls:
        print("Warning: Some repositories in the test set are missing in the predictions:")
        for url in sorted(missing_urls):
            print(f"  {url}")

    # Merge on html_url
    df_merged = pd.merge(df_repos, df_gpt, on="html_url", how="inner")
    df_merged["match"] = df_merged["project_type"] == df_merged["gpt_category"]

    # Export detailed results
    df_output = df_merged[["html_url", "gpt_category", "project_type", "gpt_explanation", "match"]]
    output_csv = f"results/{acronym}/{output_prefix}.csv"
    df_output.to_csv(output_csv, index=False)

    # Per-category statistics
    stats = {}
    for label in df_merged["project_type"].unique():
        subset = df_merged[df_merged["project_type"] == label]
        acc = round(accuracy_score(subset["project_type"], subset["gpt_category"]), 2)
        f1 = round(f1_score(subset["project_type"], subset["gpt_category"], average='macro'), 2)
        stats[label] = {"count": len(subset), "accuracy": acc, "f1": f1}

    # Overall statistics
    total_count = len(df_merged)
    total_acc = round(accuracy_score(df_merged["project_type"], df_merged["gpt_category"]), 2)
    total_f1 = round(f1_score(df_merged["project_type"], df_merged["gpt_category"], average='macro'), 2)
    stats["Total"] = {"count": total_count, "accuracy": total_acc, "f1": total_f1}

    # Convert stats to DataFrame and return
    df_stats = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    df_stats = df_stats.rename(columns={"index": "project_type"})
    df_stats.to_csv(f"results/{acronym}/{output_prefix}_summary.csv", index=False)

    return df_stats

    


