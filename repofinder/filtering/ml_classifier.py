#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:55:56 2025

@author: juanitagomez
"""

from repofinder.filtering.classification_models import least_squares, random_forest, svm,  grid_search, logistic_regression, neural_network
from repofinder.filtering.university_matrix import build_matrix_with_attributes
from repofinder.filtering.embeddings import build_matrix_with_embeddings
import pandas as pd
import os 
import time

def prepare_data(filename):
    """
    Loads data from a CSV file and prepares feature matrices and labels.
    
    Args:
        filename (str): Path to the CSV file containing repository data.
    
    Returns:
        tuple: 
            - matrix (np.ndarray): Features of labeled data (excluding first and last columns).
            - labels (np.ndarray): Manual labels corresponding to the labeled data.
            - all_data (np.ndarray): Feature matrix for all rows (excluding first and last columns).
    """
    
    # Read CSV as DataFrame
    df = pd.read_csv(filename)
    columns = df.columns[1:-1]  # Exclude first (repository_html) and last (manual_label) columns
    all_data = df[columns].to_numpy(dtype=float)
    labeled_df = df.dropna(subset=['manual_label'])
    labeled_df = labeled_df[labeled_df['manual_label'] != -1]
    
    # Convert matrix and labels to NumPy arrays
    matrix = labeled_df[columns].to_numpy(dtype=float)
    labels = labeled_df.iloc[:, -1].to_numpy(dtype=float)
    
    return matrix, labels, all_data
    

def run_models(filename, weights_output, acronym, method, all_models=True, client=None):
    """
    Runs classification models on prepared data and saves prediction results.
    
    Args:
        filename (str): CSV file path with input data.
        weights_output (str): Path to save model weights (currently commented out).
        acronym (str): University or dataset acronym used for naming output files.
        embeddings (bool, optional): Whether the data contains embeddings. Defaults to True.
        all_models (bool, optional): Whether to run all models or only a selected one. Defaults to True.
        client (object, optional): Optional client for embedding generation (unused in this snippet).
    
    Returns:
        str: Path to the CSV file containing model predictions.
    """

    # Load original CSV file
    df = pd.read_csv(filename)
    A, M, all_data = prepare_data(filename)
    filename = filename.split('.')[0]
    if all_models:
        # List of model functions with their names
        models = [
           ("least_squares", least_squares),
           ("random_forest", random_forest),
           ("neural_network", neural_network),
           ("svm", svm),
           ("grid_search", grid_search),
           ("logistic_regression", logistic_regression),
        ]

        output_filename = f"results/{acronym}/predictions_{method}_{acronym}.csv"
    
    else: 
        # This is just the chosen model. In this case random forest
        models = [
           ("svm", svm),
        ]
        output_filename = f"results/{acronym}/final_predictions_{method}_{acronym}.csv"
        
        
    # Initialize empty lists to store weights, predictions, and accuracies
    weights = []
    predictions = []
    accuracies = []
    
    # Iterate through the models, call each one, and store the results
    for model_name, model_func in models:
        start_time = time.time()
        weight, pred, accuracy = model_func(A, M, all_data, acronym, method)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{acronym} - {model_name} Model time: {elapsed:.2f} seconds")
        weights.append(weight)
        predictions.append(pred)
        accuracies.append(accuracy)
    
    # Creating predictions DataFrame
    if all_models:
        predictions_df = pd.DataFrame({
            f"Predictions with {model_name}": pred 
            for (model_name, pred, accuracy) in zip([name for name, _ in models], predictions, accuracies)
        })
    
    else: 
        predictions_df = pd.DataFrame({
            f"Predictions with {model_name}": pred 
            for (model_name, pred, accuracy) in zip([name for name, _ in models], predictions, accuracies)
        })
        
    # Creating weights DataFrame
    # weights_df = pd.DataFrame({
    #     f"{model_name}, accuracy={accuracy}": weight 
    #     for (model_name, weight, accuracy) in zip([name for name, _ in models], weights, accuracies)
    # })

    # Add predictions as new columns to the original dataframe
    df = pd.concat([df, predictions_df], axis=1)
    
    last_7_cols = df.columns[-7:].tolist()  # Use .tolist() for clarity
    cols_to_keep = ['html_url'] + last_7_cols


    df= df[cols_to_keep]
    # Save the modified DataFrame with added predictions
    df.to_csv(output_filename, index=False)

    # Saving to CSV files
    # weights_df.to_csv(weights_output, index=False)
    return output_filename
   


def train_models(acronym, config_file, db_file, method="embeddings", build_matrix=False, client=None):
    """
    trains the machine learning models on university repository data.
    
    Depending on the method, either builds attribute or embedding matrices (optionally),
    then runs models on the generated matrices.
    
    Args:
        acronym (str): University acronym (e.g., 'UCSC').
        config_file (str): Path to the JSON configuration file with university metadata.
        db_file (str): Path to the SQLite database file.
        method (str, optional): 'matrix' or 'embeddings' to select training data type. Defaults to 'embeddings'.
        build_matrix (bool, optional): Whether to build the matrix before training. Defaults to False.
        client (object, optional): Client instance for embedding generation (used if method is 'embeddings').
    
    Returns:
        str: Path to the CSV file with model predictions.
    """
    # Trains the models using the manual labels from the university (Currently UCSC, UCSB, UCSD is the only one that has manual labels)
    
    #matrix
    if method == "matrix":
        # Attributes
        if build_matrix:
            print("Building repository matrix")
            csv_matrix = build_matrix_with_attributes(config_file, db_file, acronym)  # Unless the manual labels are updated, this matrix doesn't change
        else: 
            csv_matrix = f'results/{acronym}/repository_university_matrix_{acronym}.csv'
        weights_output1 = f'results/{acronym}/weights_matrix_{acronym}.csv'
        print("Running models")
        output_filename = run_models(csv_matrix, weights_output1, acronym, method, all_models=True)
    
    # Embeddings 
    if method == "embeddings":
        
        if build_matrix: 
            print("Building embeddings matrix")
            csv_embeddings = build_matrix_with_embeddings(config_file, db_file, acronym, client=client)
        else:
            csv_embeddings = f'results/{acronym}/repository_embeddings_{acronym}.csv'
        weights_output2 = f'results/{acronym}/weights_embeddings_{acronym}.csv'
        print(f"Running models with {method}")
        output_filename= run_models(csv_embeddings, weights_output2, acronym, method, all_models=True, client=client)
    return output_filename


def train_big_model(acronyms, output_acronym="combined"):
    """
    Trains models on combined data from multiple universities and saves the results.
    
    Loads attribute and embedding matrices for each university in `acronyms`,
    concatenates them, saves combined CSVs, and runs models on combined data.
    
    Args:
        acronyms (list of str): List of university acronyms to combine (e.g., ['UCSC', 'UCSD', 'UCSB']).
        output_acronym (str, optional): Acronym to use for combined output files. Defaults to 'combined'.
    
    Returns:
        None
    """
    
    # Attributes
    attribute_dfs = []
    for acronym in acronyms:
        df = pd.read_csv(f"results/{acronym}/repository_university_matrix_{acronym}.csv")
        attribute_dfs.append(df)
    combined_attributes_df = pd.concat(attribute_dfs, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    attr_output_path = f"results/{output_acronym}/repository_university_matrix_combined.csv"
    os.makedirs(os.path.dirname(attr_output_path), exist_ok=True)
    combined_attributes_df.to_csv(attr_output_path, index=False)
    
    # Train on university matrix
    run_models(
        attr_output_path,
        f"weights_attributes_{output_acronym}.csv",
        output_acronym,
        embeddings=False,
        all_models=False
    )
    
    # Load and combine embedding matrices
    embedding_dfs = []
    for acronym in acronyms:
        df = pd.read_csv(f"results/{acronym}/repository_embeddings_{acronym}.csv")
        embedding_dfs.append(df)
    combined_embeddings_df = pd.concat(embedding_dfs, ignore_index=True)
    
    # Save combined embeddings CSV
    embed_output_path = f"results/{output_acronym}/repository_embeddings_combined.csv"
    os.makedirs(os.path.dirname(embed_output_path), exist_ok=True)
    combined_embeddings_df.to_csv(embed_output_path, index=False)
    
    # Train on embeddings
    run_models(
        embed_output_path,
        f"weights_embeddings_{output_acronym}.csv",
        output_acronym,
        embeddings=True,
        all_models=False
    )
