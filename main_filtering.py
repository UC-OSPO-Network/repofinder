#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:40:26 2025

@author: juanitagomez
"""

from repofinder.filtering.ml_classifier import train_models
from repofinder.filtering.score_based_classifier import compute_predictions_sbc
from repofinder.filtering.ai_classifier import compute_ai_predictions
from repofinder.filtering.manual_labels import update_manual_labels
from dotenv import load_dotenv
from repofinder.filtering.roc_multiple import create_roc_curves
import os
from openai import OpenAI 


DOTENV = ".env" 
load_dotenv(DOTENV)
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
client = OpenAI(api_key=OPENAI_TOKEN)

def filter_repos(university_acronyms = ['UCSB', 'UCSC', 'UCSD']): # Only 'UCSB', 'UCSC', 'UCSD' have manual labels for now
    for acronym in university_acronyms:
    
        config_file= f"config/config_{acronym}.json"
        db_file = f"Data/db/repository_data_{acronym}_database.db"
        manual_labels = f"Data/manual_labels/{acronym}_Random200.csv"
    
        sbc_predictions = compute_predictions_sbc(acronym, config_file, db_file)
        update_manual_labels(manual_labels, db_file) # Needed for supervised ML models
        ml_university_matrix_predictions = train_models(acronym, config_file, db_file, method="matrix", build_matrix=True)
        ml_university_embeddings_predictions = train_models(acronym, config_file, db_file, method="embeddings", build_matrix=True, client=client)
        ai_predictions_4o = compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False)
        ai_predictions_35 = compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-3.5-turbo", subset=False)
        
        
    create_roc_curves(university_acronyms)   
        
if __name__ == "__main__":
    filter_repos()

