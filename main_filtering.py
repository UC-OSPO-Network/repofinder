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

        # Affiliation Classification
        sbc_predictions = compute_predictions_sbc(acronym, config_file, db_file)
        update_manual_labels(manual_labels, db_file) # Needed for supervised ML models
        ml_university_matrix_predictions = train_models(acronym, config_file, db_file, method="matrix", build_matrix=True)
        ml_university_embeddings_predictions = train_models(acronym, config_file, db_file, method="embeddings", build_matrix=True, client=client)
        ai_predictions_4o = compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False)
        ai_predictions_35 = compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-3.5-turbo", subset=False)
        update_predictions_in_db(ai_predictions_4o, db_file, "ai_prediction")
        
        # Project Type classification
        ai_type_predictions_4o = compute_ai_type_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False)  
        update_predictions_in_db(ai_type_predictions_4o, db_file, "gpt_category")
        type_classifier_accuracy(f"Data/test_data/type_test_set_{acronym}", ai_type_predictions_4o, acronym)
        
    create_roc_curves(university_acronyms)
        
if __name__ == "__main__":
    filter_repos()

