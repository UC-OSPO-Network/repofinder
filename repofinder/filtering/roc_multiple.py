#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os
import copy
import string
import matplotlib as mpl

# Set Lato as the default font globally
mpl.rcParams['font.family'] = 'Lato'

def plot_university_roc_curves(acronym, ax, base_file, models, label_map, test_set,
                               ml_files=None, ai_files=None, weights_file=None,
                               title_prefix=""):
    """
    Plot ROC curves for a university.

    This function generates ROC curves for multiple classification models
    (ML, AI, and SBC) for a single university. It merges predictions from
    various sources, filters to test set samples, and generates classification
    reports.

    Parameters
    ----------
    acronym : str
        University acronym (e.g., "UCSD", "UCSC").
    ax : matplotlib.axes.Axes
        Matplotlib axis object to plot on.
    base_file : str
        Path to base prediction file to load initial dataframe. Can be any
        prediction file (ML, AI, or SBC).
    models : list of str
        List of ML model names to plot (e.g., ["svm"]).
    label_map : dict
        Dictionary mapping model keys to display names for legends.
    test_set : str
        Path to test set CSV file containing html_url and manual_label columns.
    ml_files : list of str, optional
        List of ML prediction file paths (default: None).
    ai_files : list of tuple, optional
        List of tuples (file_path, model_key) for AI models (default: None).
    weights_file : str, optional
        Path to SBC (score-based classifier) weights file (default: None).
    title_prefix : str, optional
        Prefix for plot title (default: "").

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axis object with plotted ROC curves.

    Notes
    -----
    - Filters predictions to only include test set samples with valid labels (0 or 1).
    - Generates a classification report CSV file saved to
      `results/{acronym}/classification_report_{acronym}.csv`.
    - Skips models that are not found in the merged dataframe.
    - Uses Youden's J statistic to determine optimal threshold for classification.
    """

    # Create minimal dataframe from test set
    test_df = pd.read_csv(test_set, usecols=['html_url', 'manual_label'])
    df = test_df.copy()
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

    # Merge ML predictions
    if ml_files:
        for ml_file in ml_files:
            try:
                ml_df = pd.read_csv(ml_file, lineterminator='\n')
                # Find prediction columns (columns that look like "Predictions with {model}")
                prediction_cols = [col for col in ml_df.columns if col.startswith('Predictions with ')]
                for col in prediction_cols:
                    model_name = col.replace('Predictions with ', '')
                    # Merge this specific column
                    temp = ml_df[['html_url', col]].copy()
                    temp = temp[temp[col] != 'error']
                    temp[col] = pd.to_numeric(temp[col], errors='coerce')
                    if model_name in models_local:
                        df = df.merge(temp, on='html_url', how='left')
            except Exception as e:
                print(f"[{acronym}] Error loading ML file {ml_file}: {e}")
    
    # Merge AI predictions (support multiple AI models)
    if ai_files:
        for ai_file, model_key in ai_files:
            safe_merge(ai_file, 'gpt_belonging', f'Predictions with {model_key}', model_key)
    
    # Merge SBC predictions
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
        
        # Filter out NaN values - only keep rows where both y_true and y_prob are valid
        valid_mask = y_true.notna() & y_prob.notna()
        y_true_clean = y_true[valid_mask]
        y_prob_clean = y_prob[valid_mask]
        
        if len(y_true_clean) == 0:
            continue
            
        fpr, tpr, thresholds = roc_curve(y_true_clean, y_prob_clean)
        auc = roc_auc_score(y_true_clean, y_prob_clean)
        label = label_map.get(model, model)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})", linewidth=3)
        
        # Classification Report
        # Use Youden's J statistic (TPR - FPR) for threshold selection,
        # with a minimum threshold constraint to avoid very small thresholds.
        min_threshold = 0.3
        j_scores = tpr - fpr

        valid_mask_thr = thresholds >= min_threshold
        if valid_mask_thr.any():
            valid_thresholds = thresholds[valid_mask_thr]
            valid_scores = j_scores[valid_mask_thr]
            best_threshold = valid_thresholds[valid_scores.argmax()]
        else:
            # Fallback if ROC thresholds don't include anything >= min_threshold
            best_threshold = min_threshold
        y_pred_clean = (y_prob_clean >= best_threshold).astype(int)
        
        # Compute confusion counts using the *selected* threshold (best_threshold)
        # True Negatives (TN): y=0, pred=0
        # False Positives (FP): y=0, pred=1
        # False Negatives (FN): y=1, pred=0
        # True Positives (TP): y=1, pred=1
        tn = int(((y_true_clean == 0) & (y_pred_clean == 0)).sum())
        fp = int(((y_true_clean == 0) & (y_pred_clean == 1)).sum())
        fn = int(((y_true_clean == 1) & (y_pred_clean == 0)).sum())
        tp = int(((y_true_clean == 1) & (y_pred_clean == 1)).sum())

        n_total = int(len(y_true_clean))
        n_neg = int((y_true_clean == 0).sum())
        n_pos = int((y_true_clean == 1).sum())

        # Rates by class (these are what people usually mean by FPR/FNR)
        # FPR% = FP / (# actual negatives) * 100
        # FNR% = FN / (# actual positives) * 100
        false_positive_rate_pct = (fp / n_neg * 100) if n_neg > 0 else 0.0
        false_negative_rate_pct = (fn / n_pos * 100) if n_pos > 0 else 0.0

        # Also include % of the full evaluated set (sometimes more intuitive)
        false_positive_pct_total = (fp / n_total * 100) if n_total > 0 else 0.0
        false_negative_pct_total = (fn / n_total * 100) if n_total > 0 else 0.0
    
        # Inside model loop:
        report_dict = classification_report(y_true_clean, y_pred_clean, output_dict=True)
        accuracy = report_dict['accuracy']
        
        # Handle missing classes in classification report
        precision_0 = round(report_dict.get('0', {}).get('precision', 0.0), 2) if '0' in report_dict else 0.0
        recall_0 = round(report_dict.get('0', {}).get('recall', 0.0), 2) if '0' in report_dict else 0.0
        f1_0 = round(report_dict.get('0', {}).get('f1-score', 0.0), 2) if '0' in report_dict else 0.0
        precision_1 = round(report_dict.get('1', {}).get('precision', 0.0), 2) if '1' in report_dict else 0.0
        recall_1 = round(report_dict.get('1', {}).get('recall', 0.0), 2) if '1' in report_dict else 0.0
        f1_1 = round(report_dict.get('1', {}).get('f1-score', 0.0), 2) if '1' in report_dict else 0.0
        
        report_rows.append({
            "Model": label,
            "Threshold": round(best_threshold, 2),
            "N": n_total,
            "N pos": n_pos,
            "N neg": n_neg,
            "FP": fp,
            "FN": fn,
            "Precision 0": precision_0,
            "Recall 0": recall_0,
            "F1 0": f1_0,
            "Precision 1": precision_1,
            "Recall 1": recall_1,
            "F1 1": f1_1,
            "Accuracy": round(accuracy, 2),
            "False Positive %": round(false_positive_rate_pct, 2),
            "False Negative %": round(false_negative_rate_pct, 2),
            "False Positive % (total)": round(false_positive_pct_total, 2),
            "False Negative % (total)": round(false_negative_pct_total, 2),
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
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    report_df.to_csv(csv_path, index=False)

    return ax



def roc_multi(acronyms, base_files, models, ml_files_list=None, 
              ai_files_list=None, weights_files=None, label_map=None):
    """
    Generate ROC curves for multiple universities.

    This function creates a multi-panel figure with ROC curves for each
    university, plotting predictions from ML models, AI models, and SBC.

    Parameters
    ----------
    acronyms : list of str
        List of university acronyms to plot (e.g., ["UCSD", "UCSC"]).
    base_files : list of str
        List of base prediction file paths, one per acronym. Used as the
        initial dataframe for merging other predictions.
    models : list of str
        List of ML model names to plot (e.g., ["svm"]).
    ml_files_list : list of list of str, optional
        List of ML prediction file lists (one list per ML model type).
        Each inner list contains one file per acronym (default: None).
    ai_files_list : list of tuple, optional
        List of tuples (file_list, model_key) where file_list contains
        one file per acronym for each AI model (default: None).
    weights_files : list of str, optional
        List of SBC weights file paths, one per acronym (default: None).
    label_map : dict, optional
        Dictionary mapping model keys to display names. If None, uses
        default mapping with "svm" -> "SVM" and "weights" -> "SBC"
        (default: None).

    Returns
    -------
    None
        Saves the plot to "results/roc_combined_4&5.png" and displays it.

    Notes
    -----
    - Creates a subplot for each university with letter labels (a), (b), etc.
    - Saves the combined plot as a high-resolution PNG (300 DPI).
    - Only the first subplot has a y-axis label.
    """

    if label_map is None:
        label_map = {
            "svm": "SVM",
            "weights": "SBC"
        }

    # Handle single acronym case (axes will be a single Axes object, not an array)
    if len(acronyms) == 0:
        raise ValueError("At least one acronym must be provided")
    
    fig, axes = plt.subplots(ncols=len(acronyms), figsize=(7 * len(acronyms), 7), sharey=True)
    
    # Convert single Axes to list for consistent indexing
    if len(acronyms) == 1:
        axes = [axes]

    for idx, acronym in enumerate(acronyms):
        ax = axes[idx]
        base_file = base_files[idx] if base_files else None
        
        # Get ML file for this acronym
        # ml_files_list is a list of file lists (one list per ML model type)
        # Since one ML file contains all ML model predictions, we use the first list
        ml_files = None
        if ml_files_list and len(ml_files_list) > 0 and idx < len(ml_files_list[0]):
            ml_files = [ml_files_list[0][idx]]  # Single file in a list for plot_university_roc_curves
        
        # Get AI files for this acronym (list of (file_path, model_key) tuples)
        ai_files = None
        if ai_files_list:
            ai_files = [(files[idx], model_key) for files, model_key in ai_files_list if idx < len(files)]
        
        weights_file = weights_files[idx] if weights_files and idx < len(weights_files) else None
        letter = string.ascii_lowercase[idx]
        title_prefix = f"({letter}) "

        test_set = f"Data/test_data/test_set_{acronym}.csv"
        plot_university_roc_curves(
            acronym, ax, base_file, models, 
            label_map, test_set,
            ml_files=ml_files,
            ai_files=ai_files,
            weights_file=weights_file,
            title_prefix=title_prefix
        )

        if idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=32, labelpad=15)

        ax.legend(loc="lower right", fontsize=17)

    plt.tight_layout()
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/roc_combined.png", dpi=300, bbox_inches="tight")
    plt.show()



def build_paths_for(method, acronyms):
    """
    Build file paths for prediction files.

    This function constructs file paths for ML, AI, and SBC prediction files
    based on the method and university acronyms.

    Parameters
    ----------
    method : str
        ML method name (e.g., "embeddings", "matrix").
    acronyms : list of str
        List of university acronyms (e.g., ["UCSD", "UCSC"]).

    Returns
    -------
    tuple
        Three-element tuple containing:
        - ml_files (list of str): List of ML prediction file paths,
          one per acronym. Format: `results/{acronym}/predictions_{method}_{acronym}_subset.csv`
        - ai_files_dict (dict): Dictionary mapping model keys (e.g., "gpt-4o")
          to lists of AI file paths, one per acronym. Format:
          `results/{acronym}/predictions_ai_gpt-{version}_{acronym}_subset.csv`
        - sbc_files (list of str): List of SBC prediction file paths,
          one per acronym. Format: `results/{acronym}/predictions_sbc_{acronym}_subset.csv`

    Notes
    -----
    - Supports AI models: gpt-3.5-turbo, gpt-4o, gpt-5-mini, gpt-5
    - All paths are relative to the project root directory.
    """
    ml_files = []
    ai_files_dict = {}  # {model_key: [file1, file2, ...]}
    sbc_files = []

    for acronym in acronyms:
        # ML files: predictions_{method}_{acronym}_subset.csv
        ml_file = f"results/{acronym}/predictions_{method}_{acronym}_subset.csv"
        ml_files.append(ml_file)
        
        # AI files: predictions_ai_gpt-{version}_{acronym}_subset.csv
        ai_versions = ['3.5-turbo', '4o', '5-mini', '5']
        for version in ai_versions:
            model_key = f"gpt-{version}"
            if model_key not in ai_files_dict:
                ai_files_dict[model_key] = []
            # Convert version for file path: 3.5-turbo -> 3-5-turbo
            path_version = version.replace('.', '-')
            ai_file = f"results/{acronym}/predictions_ai_gpt-{path_version}_{acronym}_subset.csv"
            ai_files_dict[model_key].append(ai_file)
        
        # SBC files
        sbc_file = f"results/{acronym}/predictions_sbc_{acronym}_subset.csv"
        sbc_files.append(sbc_file)

    return ml_files, ai_files_dict, sbc_files

def create_roc_curves(acronyms, curves_to_plot=['gpt-4o', 'gpt-5-mini'], method='embeddings'):
    """
    Create ROC curves for specified models across multiple universities.

    This function orchestrates the generation of ROC curves by determining
    which prediction files to load based on the requested curves, building
    file paths, and calling the plotting functions.

    Parameters
    ----------
    acronyms : list of str
        List of university acronyms to plot (e.g., ["UCSD", "UCSC"]).
    curves_to_plot : list of str, optional
        List of curve types to plot. Valid options (case-insensitive):
        - 'sbc' or 'scorebased' or 'weights': Score-based classifier
        - 'ml' or 'machinelearning': Machine learning models (e.g., SVM)
        - 'gpt-4o' or 'gpt4o' or 'gpt4': GPT-4o predictions
        - 'gpt-5-mini' or 'gpt-mini-5' or 'gpt5mini': GPT-5-mini predictions
        - 'gpt-3.5-turbo' or 'gpt-35' or 'gpt35': GPT-3.5-turbo predictions
        - 'gpt-5' or 'gpt5': GPT-5 predictions
        (default: ['sbc', 'ml', 'gpt-4o', 'gpt-5-mini'])
    method : str, optional
        ML method name for building ML file paths (default: 'embeddings').

    Returns
    -------
    None
        Generates and saves ROC curve plots.

    Notes
    -----
    - Automatically determines the base file from available prediction files
      (prioritizes ML, then AI, then SBC).
    - Normalizes curve names to handle variations (e.g., 'gpt-5-mini' and
      'gpt5mini' are treated the same).
    - Only includes models that have corresponding prediction files.
    - Creates a label map for all requested models with formatted display names.
    """
    # Normalize curve names (case-insensitive, handle variations)
    curves_set = {c.lower().replace('-', '').replace('_', '').replace('.', '') for c in curves_to_plot}
    
    # Build all paths
    ml_files, ai_files_dict, sbc_files = build_paths_for(method, acronyms)
    
    # Determine which files to use based on curves_to_plot
    # Normalize check values to match the normalization of curves_set
    use_ml = any(c.lower().replace('-', '').replace('_', '').replace('.', '') in curves_set 
                 for c in ['ml', 'machinelearning', 'svc', 'svm', 'embeddings', 'matrix'])
    use_sbc = any(c.lower().replace('-', '').replace('_', '').replace('.', '') in curves_set 
                  for c in ['sbc', 'scorebased', 'weights'])
    
    # ML models list (only used if ML is requested)
    models = [
        # "least_squares",
        # "random_forest", 
        # "neural_network", 
        "svm", 
        # "grid_search",
        # "logistic_regression"
    ] if use_ml else []
    
    # Determine which AI models to include
    ai_files_list = []  # List of (file_list, model_key) tuples
    base_files = []
    
    # Find requested AI models (directly match against available model keys)
    requested_ai_models = []
    for curve in curves_to_plot:
        # Normalize the curve name for comparison (case-insensitive, ignore dashes/underscores/dots)
        normalized_curve = curve.lower().replace('-', '').replace('_', '').replace('.', '')
        # Check each available model key
        for model_key in ai_files_dict.keys():
            normalized_key = model_key.lower().replace('-', '').replace('_', '').replace('.', '')
            if normalized_curve == normalized_key:
                if model_key not in requested_ai_models:
                    requested_ai_models.append(model_key)
                break
    
    # Build AI files list
    for model_key in requested_ai_models:
        if model_key in ai_files_dict:
            ai_files_list.append((ai_files_dict[model_key], model_key))
    
    # Determine base file (first available file, one per acronym)
    if use_ml and ml_files:
        base_files = ml_files
    elif ai_files_list and len(ai_files_list) > 0:
        # Use first AI file as base
        base_files = ai_files_list[0][0]  # This is already a list of files (one per acronym)
    elif use_sbc and sbc_files:
        base_files = sbc_files
    else:
        # Fallback: use ML files even if not requested
        base_files = ml_files
    
    # Prepare ML files list (only if requested)
    # ml_files_list should be a list of file lists (one list per ML model type)
    # Since we have one ML file per acronym, we wrap it in a list
    ml_files_list = None
    if use_ml and ml_files:
        ml_files_list = [ml_files]  # List containing one list of ML files
    
    # Prepare weights files (only if requested)
    weights_files_param = sbc_files if use_sbc else None
    
    # Build label map with AI models
    label_map = {
        "svm": "SVM",
        "weights": "SBC"
    }
    for model_key in requested_ai_models:
        # Use model_key directly for AI models (e.g., "gpt-5-mini", "gpt-4o")
        label_map[model_key] = model_key
    
    roc_multi(acronyms, base_files, models=models,
              ml_files_list=ml_files_list,
              ai_files_list=ai_files_list,
              weights_files=weights_files_param,
              label_map=label_map)
    

