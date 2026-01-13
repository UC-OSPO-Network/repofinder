#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv
from openai import OpenAI

from repofinder.filtering.ai_classifier import compute_ai_predictions
from repofinder.filtering.ai_type_classifier import compute_ai_type_predictions
from repofinder.filtering.filter_utils import update_predictions_in_db
from repofinder.filtering.manual_labels import update_manual_labels
from repofinder.filtering.ml_classifier import train_models
from repofinder.filtering.roc_multiple import create_roc_curves
from repofinder.filtering.score_based_classifier import compute_predictions_sbc
from repofinder.filtering.type_classifier_accuracy import type_classifier_accuracy


DOTENV = ".env" 
load_dotenv(DOTENV)
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
client = OpenAI(api_key=OPENAI_TOKEN)

def filter_repos(university_acronyms = ['UCSB', 'UCSC', 'UCSD']): # Only 'UCSB', 'UCSC', 'UCSD' have manual labels for now
    for acronym in university_acronyms:

        config_file= f"config/config_{acronym}.json"
        db_file = f"Data/db/repository_data_{acronym}_database.db"


        manual_labels = f"Data/manual_labels/{acronym}_Random200.csv"
        test_set = f"Data/test_data/test_set_{acronym}.csv"

        # Affiliation Classification
        sbc_predictions = compute_predictions_sbc(acronym, config_file, db_file)
        update_manual_labels(manual_labels, db_file) # Needed for supervised ML models
        ml_university_matrix_predictions = train_models(acronym, config_file, db_file, method="matrix", build_matrix=True)
        ml_university_embeddings_predictions = train_models(acronym, config_file, db_file, method="embeddings", build_matrix=True, client=client)
        ai_predictions_4o = compute_ai_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-4o", 
            #subset=test_set,  # Use test_set to only run classification on a subset 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000
        )
        ai_predictions_5 = compute_ai_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-5-mini", 
            #subset=test_set,  # Use test_set to only run classification on a subset 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000
        )

        # Project Type classification (only for gpt-5-mini)
        ai_type_predictions_5 = compute_ai_type_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-5-mini", 
            #subset=test_set,  # Use test_set to only run classification on a subset 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000
        )

        # Paths where the predictions are saved
        ai_predictions_4o = f"results/{acronym}/predictions_ai_gpt-4o_{acronym}.csv"
        ai_predictions_5 = f"results/{acronym}/predictions_ai_gpt-5-mini_{acronym}.csv"
        ai_type_predictions_5 = f"results/{acronym}/repo_type_gpt-5-mini_{acronym}.csv"
        ai_type_predictions_4o = f"results/{acronym}/repo_type_gpt-4o_{acronym}.csv"


        # Update the predictions in the database
        update_predictions_in_db(ai_predictions_4o, db_file, "affiliation_prediction_gpt_4o")   
        update_predictions_in_db(ai_predictions_5, db_file, "affiliation_prediction_gpt_5_mini")       
        update_predictions_in_db(ai_type_predictions_4o, db_file, "type_prediction_gpt_4o")
        update_predictions_in_db(ai_type_predictions_5, db_file, "type_prediction_gpt_5_mini")
        
        type_classifier_accuracy(f"Data/test_data/type_test_set_{acronym}", ai_type_predictions_5, acronym)
        
    create_roc_curves(university_acronyms, curves_to_plot=['gpt-4o', 'gpt-5-mini'])
        
if __name__ == "__main__":
    filter_repos()

