#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:20:46 2025

@author: juanitagomez
"""

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import re
import copy
import string
import matplotlib as mpl

# Set Lato as the default font globally
mpl.rcParams['font.family'] = 'Lato'

def plot_university_roc_curves(acronym, ax, file_path, models, label_map, test_set,
                               ai_file=None, ai2_file=None, weights_file=None,
                               title_prefix=""):
    df = pd.read_csv(f"{file_path}")
    test_df = pd.read_csv(test_set, usecols=['html_url', 'manual_label'])
    models_local = copy.deepcopy(models)

    # Merge predictions
    def safe_merge(csv_file, col_in, col_out, model_key):
        nonlocal df
        if csv_file:
            try:
                temp = pd.read_csv(csv_file, usecols=['html_url', col_in], lineterminator='\n')
                temp = temp.rename(columns={col_in: col_out})
                temp = temp[temp[col_out] != 'error']
                temp[col_out] = temp[col_out].astype(float)  # Ensure float type
                df = df.merge(temp, on='html_url', how='left')
                models_local.append(model_key)
            except Exception as e:
                print(f"[{acronym}] Error loading {model_key}: {e}")


    # Merge predictions
    safe_merge(ai_file, 'gpt_belonging', 'Predictions with ai', 'ai')
    safe_merge(ai2_file, 'gpt_belonging', 'Predictions with ai2', 'ai2')
    safe_merge(weights_file, 'total_score', 'Predictions with weights', 'weights')
    
    # Load test set and preserve its manual_label
    test_df = pd.read_csv(test_set, usecols=['html_url', 'manual_label'])
    test_df = test_df.rename(columns={'manual_label': 'test_manual_label'})
    
    # Merge with test set to filter only test samples
    df = df.merge(test_df, on='html_url', how='inner')
    
    # Keep only the test set label
    df = df[df['test_manual_label'].isin([0, 1])]
    df['manual_label'] = df['test_manual_label'].astype(int)
    df = df.drop(columns=['test_manual_label'])

    
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
        
        # Classification Report
        j_scores = tpr - fpr
        best_threshold = thresholds[j_scores.argmax()]
        y_pred = (y_prob >= best_threshold).astype(int)
    
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
    csv_path = f"results/{acronym}/classification_report_{acronym}.csv"
    report_df.to_csv(csv_path, index=False)

    return ax



def roc_multi(acronyms, file_paths, models, ai_predictions_files, 
              ai_predictions_files2, weights_files):

    label_map = {
        "svm": "SVM",
        "ai2": "gpt-4o",
        "ai": "gpt-3.5",
        "weights": "SBC"
    }

    fig, axes = plt.subplots(ncols=len(acronyms), figsize=(7 * len(acronyms), 7), sharey=True)

    for idx, (acronym, file_path) in enumerate(zip(acronyms, file_paths)):
        ax = axes[idx]
        ai_file = ai_predictions_files[idx] if ai_predictions_files else None
        ai2_file = ai_predictions_files2[idx] if ai_predictions_files2 else None
        weights_file = weights_files[idx] if weights_files else None
        letter = string.ascii_lowercase[idx]
        title_prefix = f"({letter}) "


        test_set = f"Data/test_data/test_set_{acronym}.csv"
        plot_university_roc_curves(
            acronym, ax, file_path, models, 
            label_map, test_set,
            ai_file=ai_file,
            ai2_file=ai2_file,
            weights_file=weights_file,
            title_prefix=title_prefix
        )

        if idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=32, labelpad=15)

        ax.legend(loc="lower right", fontsize=22)

    plt.tight_layout()
    plt.savefig("results/roc_combined.png", dpi=300, bbox_inches="tight")
    plt.show()



def build_paths_for(method, acronyms):
    file_paths = []
    ai_files_35 = []
    ai_files_4o = []
    sbc_files = []

    for acronym in acronyms:
        file_path = f"results/{acronym}/predictions_{method}_{acronym}.csv"
        ai_file_35 = f"results/{acronym}/predictions_ai_gpt-3.5-turbo_{acronym}.csv"
        ai_file_4o = f'results/{acronym}/predictions_ai_gpt-4o_{acronym}.csv'
        sbc_file = f"results/{acronym}/predictions_sbc_{acronym}.csv"

        file_paths.append(file_path)
        ai_files_35.append(ai_file_35)
        ai_files_4o.append(ai_file_4o)
        sbc_files.append(sbc_file)

    return file_paths, ai_files_35, ai_files_4o, sbc_files

def create_roc_curves(acronyms):
    models = [
        # "least_squares",
        # "random_forest", 
        # "neural_network", 
        "svm", 
        # "grid_search",
        # "logistic_regression"
    ]
    file_paths, ai_files_35, ai_files_4o, weights_files= build_paths_for("embeddings", acronyms)
    
    roc_multi(acronyms, file_paths, models=models,
              ai_predictions_files=ai_files_35,
              ai_predictions_files2=ai_files_4o,
              weights_files=weights_files)
    

