#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:20:46 2025

@author: juanitagomez
"""

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, f1_score
import matplotlib.pyplot as plt
import re
import copy
import string
import matplotlib as mpl

# Set Lato as the default font globally
mpl.rcParams['font.family'] = 'Lato'

def merge_labels_with_predictions(labels_file, predictions_file, output_file, models):
    """
    Merges manual labels with predictions from multiple models.

    Parameters:
    - labels_file: path to the CSV file containing 'html_url' and 'manual_label'.
    - predictions_file: path to the CSV file containing prediction columns per model.
    - output_file: path to save the merged CSV file.
    - models: list of model names (default includes 6 common models).

    Returns:
    - The merged DataFrame.
    """
    # Load label data
    df_labels = pd.read_csv(labels_file, usecols=["html_url", "manual_label"])

    # Load prediction data
    df_predictions = pd.read_csv(predictions_file)

    # Merge prediction columns one by one
    for model in models:
        col_name = f"Predictions with {model}"
        if col_name in df_predictions.columns:
            df_labels = df_labels.merge(
                df_predictions[["html_url", col_name]],
                on="html_url",
                how="left"
            )
        else:
            print(f"Warning: Column '{col_name}' not found in predictions file.")

    # Save merged DataFrame
    df_labels.to_csv(output_file, index=False)

    return df_labels


def plot_university_roc_curves(ax, file_path, models, label_map,
                               ai_file=None, ai2_file=None, weights_file=None,
                               title_prefix=""):
    df = pd.read_csv(f"{file_path}")
    models_local = copy.deepcopy(models)

    # Extract acronym
    match = re.search(r'(uc[^_]+)', file_path)
    acronym = match.group(1).upper() if match else "Unknown"
    acronym = acronym.split('/')[0]

    # Clean and filter labels
    df = df[df['manual_label'].notna() & (df['manual_label'] != '')]
    df = df[df['manual_label'].isin([0, 1])]
    df['manual_label'] = df['manual_label'].astype(int)

    # Merge predictions
    def safe_merge(csv_file, col_in, col_out, model_key):
        nonlocal df
        if csv_file:
            try:
                temp = pd.read_csv(csv_file, usecols=['html_url', col_in])
                temp = temp.rename(columns={col_in: col_out})
                df = df.merge(temp, on='html_url', how='left')
                models_local.append(model_key)
            except Exception as e:
                print(f"[{acronym}] Error loading {model_key}: {e}")

    safe_merge(ai_file, 'gpt_belonging', 'Predictions with ai', 'ai')
    safe_merge(ai2_file, 'gpt_belonging', 'Predictions with ai2', 'ai2')
    safe_merge(weights_file, 'total_score', 'Predictions with weights', 'weights')

    # Plot each model
    y_true = df['manual_label']
    #with open(report_output_file, 'w') as report_file:
    report_rows = []    
    for model in models_local:
        col_name = f"Predictions with {model}"
        if col_name not in df.columns or df[col_name].isnull().all():
            continue
        y_prob = df[col_name]
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        label = label_map.get(model, model)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})", linewidth=3)
        
        # Report
        # j_scores = tpr - fpr
        # best_threshold = thresholds[j_scores.argmax()]
        # y_pred = (y_prob >= best_threshold).astype(int)
        
        
        
        best_f1 = -1
        best_threshold = 0.5  # default fallback
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            score = f1_score(y_true, y_pred)
            if score > best_f1:
                best_f1 = score
                best_threshold = thresh
        
        y_pred = (y_prob >= best_threshold).astype(int)
        #accuracy = accuracy_score(y_true, y_pred)
        #class_report = classification_report(y_true, y_pred, digits=3)
        # print(f'{acronym} {model}: {best_threshold}')
        # print(class_report)


        # Inside model loop:
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        accuracy = report_dict['accuracy']
        report_rows.append({
            "Model": label,
            "Threshold": round(best_threshold, 2),
            "Precision 0": round(report_dict['0']['precision'], 2),
            "Recall 0": round(report_dict['0']['recall'], 2),
            "F1 0": round(report_dict['0']['f1-score'], 2),
            "Precision 1": round(report_dict['1']['precision'], 2),
            "Recall 1": round(report_dict['1']['recall'], 2),
            "F1 1": round(report_dict['1']['f1-score'], 2),
            "Accuracy": round(accuracy, 2),
        })
    
    # Plot random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=3)
    #ax.set_title(f"{title_prefix}{acronym}", fontsize=32, loc="left", pad=20, fontweight="bold")
    ax.set_title(rf"$\bf{{{title_prefix}\ {acronym}}}$",
                 fontsize=32, loc="left", pad=20)
    
    ax.set_xlabel("False Positive Rate", fontsize=32, labelpad=15)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=25)
    
    # Write report
    report_df = pd.DataFrame(report_rows)
    csv_path = f"classification_report_{acronym}.csv"
    report_df.to_csv(csv_path, index=False)


    return ax



def roc_multi(path, file_paths, models, ai_predictions_files, 
              ai_predictions_files2, weights_files):

    label_map = {
        "svm": "SVM",
        "ai2": "gpt-4o",
        "ai": "gpt-3.5",
        "weights": "SBC"
    }

    fig, axes = plt.subplots(ncols=3, figsize=(21, 7), sharey=True)

    for idx, file_path in enumerate(file_paths):
        ax = axes[idx]
        ai_file = ai_predictions_files[idx] if ai_predictions_files else None
        ai2_file = ai_predictions_files2[idx] if ai_predictions_files2 else None
        weights_file = weights_files[idx] if weights_files else None
        letter = string.ascii_lowercase[idx]
        title_prefix = f"({letter}) "

        plot_university_roc_curves(
            ax, file_path, models, label_map,
            ai_file=ai_file,
            ai2_file=ai2_file,
            weights_file=weights_file,
            title_prefix=title_prefix
        )

        if idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=32, labelpad=15)

        ax.legend(loc="lower right", fontsize=22)

    plt.tight_layout()
    plt.savefig(f"{path}/roc_combined.png", dpi=300, bbox_inches="tight")
    plt.show()




models = [
    "least_squares",
    "random_forest", 
    "neural_network", 
    "svm", 
    "grid_search",
    "logistic_regression"
]


acronyms =["UCSB", "UCSC", "UCSD"]

for acronym in acronyms:
    path = f"/Users/juanitagomez/Documents/Research/RepositoryFinder/repofinder/filtering/results/{acronym}"  
    
    merge_labels_with_predictions(
        labels_file=f"{path}/{acronym}_new_labels.csv",
        predictions_file=f"results/{acronym}/repository_embeddings_{acronym}_predictions.csv",
        output_file=f"{path}/{acronym}_labeled_test_set_with_predictions_embeddings.csv",
        models = models
    )
    
    merge_labels_with_predictions(
        labels_file=f"{path}/{acronym}_new_labels.csv",
        predictions_file=f"results/{acronym}/repository_university_matrix_{acronym}_predictions.csv",
        output_file=f"{path}/{acronym}_labeled_test_set_with_predictions_university_matrix.csv",
        models = models
    )


path = f"/Users/juanitagomez/Documents/Research/RepositoryFinder/repofinder/filtering/results"  

def build_paths_for(path, method, data, acronyms=["ucsb", "ucsc", "ucsd"]):
    file_paths = []
    ai_files_35 = []
    ai_files_4o = []
    weights_files = []

    for acronym in acronyms:
        if data == "all_data":
            file_path = f"{path}/{acronym}/repository_{method}_{acronym}_predictions.csv"
            ai_file_35 = f"{path}/{acronym}/{acronym}_ai_predictions.csv"
            ai_file_4o = f"{path}/{acronym}/{acronym}_ai_predictions(4o).csv"
            weights_file = f"{path}/{acronym}/{acronym}_weight_scores.csv"

        elif data == "new_labels":
            file_path = f"{path}/{acronym}/{acronym}_labeled_test_set_with_predictions_{method}.csv"
            ai_file_35 = f"{path}/{acronym}/{acronym}_ai_predictions_new_labels.csv"
            ai_file_4o = f"{path}/{acronym}/{acronym}_ai_predictions_new_labels(4o).csv"
            weights_file = f"{path}/{acronym}/{acronym}_new_labels_weight_scores.csv"
            
        file_paths.append(file_path)
        ai_files_35.append(ai_file_35)
        ai_files_4o.append(ai_file_4o)
        weights_files.append(weights_file)

    return file_paths, ai_files_35, ai_files_4o, weights_files



file_paths, ai_files_35, ai_files_4o, weights_files= build_paths_for(path, "embeddings", "new_labels")

roc_multi(path, file_paths, models=["svm"],
          ai_predictions_files=ai_files_35,
          ai_predictions_files2=ai_files_4o,
          weights_files=weights_files)


# def merge_uc_data(acronym, method="embeddings"):
#     """
#     Merge labeled test set with AI predictions (3.5, 4o) and weighted scores based on html_url.
    
#     Parameters:
#         acronym (str): University acronym (e.g., "UCSB")
#         path (str): Base path to data folders
#         method (str): Embedding method (default is "embeddings")
    
#     Returns:
#         pd.DataFrame: Merged DataFrame
#     """
#     path = f"/Users/juanitagomez/Documents/Research/RepositoryFinder/repofinder/filtering/results/"  
#     # File paths
#     file_path = f"{path}/{acronym}/{acronym}_labeled_test_set_with_predictions_{method}.csv"
#     ai_file_35 = f"{path}/{acronym}/{acronym}_ai_predictions_new_labels.csv"
#     ai_file_4o = f"{path}/{acronym}/{acronym}_ai_predictions_new_labels(4o).csv"
#     weights_file = f"{path}/{acronym}/{acronym}_new_labels_weight_scores.csv"

#     # Load CSVs
#     df_main = pd.read_csv(file_path)
#     df_35 = pd.read_csv(ai_file_35)[["html_url", "gpt_belonging"]].rename(columns={"gpt_belonging": "gpt_belonging_3_5"})
#     df_4o = pd.read_csv(ai_file_4o)[["html_url", "gpt_belonging"]].rename(columns={"gpt_belonging": "gpt_belonging_4o"})
#     df_weights = pd.read_csv(weights_file)[["html_url", "total_score"]]

#     # Merge all on html_url
#     df_merged = df_main.merge(df_35, on="html_url", how="left")
#     df_merged = df_merged.merge(df_4o, on="html_url", how="left")
#     df_merged = df_merged.merge(df_weights, on="html_url", how="left")

#     # Save to file
#     output_file = f"{path}/{acronym}_merged_dataset.csv"
#     df_merged.to_csv(output_file, index=False)
#     print(f"Merged dataset saved to {output_file}")
    
#     return df_merged


# df_ucsb = merge_uc_data("UCSD")



#embeddings
#roc(path, file_path_embeddings_all_data, models, ai_file_35_all_data, ai_file_4o_all_data, weights_file_all_data )
#roc(path, file_path_embeddings_new_labels, models, ai_file_35_new_labels, ai_file_4o_new_labels, weights_file_new_labels)


