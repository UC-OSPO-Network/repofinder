#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:40:26 2025

@author: juanitagomez
"""

from repofinder.filtering.ml_classifier import train_models
from repofinder.filtering.score_based_classifier import compute_predictions_sbc
from repofinder.filtering.ai_classifier import compute_ai_predictions
from dotenv import load_dotenv
import os

#from predictions_to_db import update_predictions

from openai import OpenAI 


DOTENV = ".env" 
load_dotenv(DOTENV)
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
client = OpenAI(api_key=OPENAI_TOKEN)

university_acronyms = ['UCSC'] # Only these three have manual labels for now


for acronym in university_acronyms:

    config_file= f"config/config_{acronym}.json"
    db_file = f"Data/db/repository_data_{acronym}_database.db"
    
    # sbc_predictions = compute_predictions_sbc(acronym, config_file, db_file)
    # ml_university_matrix_predictions = train_models(acronym, config_file, db_file, method="matrix", build_matrix=True)
    # ml_university_embeddings_predictions = train_models(acronym, config_file, db_file, method="embeddings", build_matrix=True, client=client)
    # ai_predictions_4o = compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False)
    ai_predictions_35 = compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-3.5-turbo", subset=False)
    
    
    
    
    
    
    
    
    # # Embeddings
    # csv_embeddings = build_matrix_with_embeddings(config_file, db_file, acronym, path) # Unless the manual labels are updated, this matrix doesn't change
    # # #csv_embeddings = f'{path}/results/ucsb/repository_embeddings_UCSB.csv'
    # # output_predictions = predict_rf(csv_embeddings, path, acronym)
    # csv_file_predictions = f'{path}/results/{acronym}/repository_embeddings_{acronym}_predictions_random_forest_saved.csv'
    # update_predictions(csv_file_predictions, db_file)
    
    # print(f'{acronym} Okay')
   
    
    
university_acronyms = ['UCSC']
path = "/Users/juanitagomez/Documents/Research/RepositoryFinder/repofinder"


# times_log = []
# for acronym in university_acronyms:
    
#     start_time = time.time()
#     db_file = f'{path}/Data/DB/repository_data_{acronym}_database.db'
#     config_file= f"{path}/config/config_{acronym}.json"
#     df_scores = compute_scores(db_file, config_file)
#     df_scores.to_csv(f"{path}/filtering/results/{acronym}/{acronym}_weight_scores_all_data.csv", index=False)
#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(f"{acronym} - Elapsed time Score based classifier (SBC): {elapsed:.2f} seconds")
    
    
#     # filtered data
#     db_file = f'{path}/Data/DB/repository_data_{acronym}_database.db'
#     config_file= f"{path}/config/config_{acronym}.json"
#     df_scores = compute_scores(db_file, config_file)
#     # filter_df = pd.read_csv(f"{path}/filtering/results/{acronym}/{acronym}_ai_predictions.csv")
#     # filtered_scores = df_scores[df_scores["html_url"].isin(filter_df["html_url"])]  
#     # #  Save the result to a CSV file
#     # filtered_scores.to_csv(f"{path}/filtering/results/{acronym}/{acronym}_weight_scores.csv", index=False)
        
    
#     #new labels
#     filter_df = pd.read_csv(f"{path}/filtering/results/{acronym}/{acronym}_ai_predictions_new_labels.csv")
#     filtered_scores = df_scores[df_scores["html_url"].isin(filter_df["html_url"])]  
#     # Save the result to a CSV file
#     filtered_scores.to_csv(f"{path}/filtering/results/{acronym}/{acronym}_new_labels_weight_scores.csv", index=False)
    
    
    
    