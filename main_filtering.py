#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
import time

from repofinder.filtering.ai_classifier import compute_ai_predictions
from repofinder.filtering.ai_type_classifier import compute_ai_type_predictions
from repofinder.filtering.filter_utils import update_predictions_in_db
from repofinder.filtering.manual_labels import update_manual_labels
from repofinder.filtering.ml_classifier import train_models
from repofinder.filtering.roc_multiple import create_roc_curves
from repofinder.filtering.score_based_classifier import compute_predictions_sbc
from repofinder.filtering.type_classifier_accuracy import type_classifier_accuracy


# Load .env from project root (directory containing this file)
_project_root = os.path.dirname(os.path.abspath(__file__))
DOTENV = os.path.join(_project_root, ".env")
load_dotenv(DOTENV)

# Use Azure OpenAI if Azure env vars are set; otherwise OpenAI (Option A: same code path for both)
AZURE_API_KEY = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
AZURE_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
if AZURE_ENDPOINT and AZURE_API_KEY:
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )
    # Azure uses deployment names. Set in .env: AZURE_OPENAI_DEPLOYMENT (one for all), or
    # AZURE_OPENAI_EMBEDDING_DEPLOYMENT and AZURE_OPENAI_CHAT_DEPLOYMENT for separate deployments.
    print("Using Azure OpenAI (embeddings and chat will use your Azure deployment names)")
    using_azure = True
else:
    # Support both OPENAI_TOKEN and OPENAI_API_KEY (common in docs)
    OPENAI_TOKEN = (os.getenv("OPENAI_TOKEN") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not OPENAI_TOKEN:
        raise ValueError(
            "No OpenAI API key found. In .env set OPENAI_TOKEN=sk-... (or OPENAI_API_KEY=sk-...). "
            "Get a key at https://platform.openai.com/account/api-keys"
        )
    client = OpenAI(api_key=OPENAI_TOKEN)
    print("Using OpenAI (platform.openai.com)")
    using_azure = False

# When using Azure, embeddings need AZURE_OPENAI_EMBEDDING_DEPLOYMENT (embedding model, not chat).
has_embedding_deployment = bool((os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "").strip())

def filter_repos(university_acronyms = ['GT']):
    for acronym in university_acronyms:

        config_file= f"config/config_{acronym}.json"
        db_file = f"Data/db/repository_data_{acronym}_database.db"


        manual_labels = f"Data/manual_labels/{acronym}_Random200.csv"
        test_set = f"Data/test_data/test_set_{acronym}.csv"
        type_test_set = f"Data/test_data/type_test_set_{acronym}.csv"

        # Affiliation Classification
        sbc_predictions = compute_predictions_sbc(acronym, config_file, db_file, subset=test_set)
        update_manual_labels(manual_labels, db_file) # Needed for supervised ML models
        ml_university_matrix_predictions = train_models(acronym, config_file, db_file, method="matrix", build_matrix=True)
        if not using_azure or has_embedding_deployment:
            ml_university_embeddings_predictions = train_models(acronym, config_file, db_file, method="embeddings", build_matrix=True, client=client, subset=test_set)
        else:
            print("Skipping embeddings (Azure with no AZURE_OPENAI_EMBEDDING_DEPLOYMENT). Set it in .env to enable.")

        # Limit AI predictions to this many repos (set to None to process all)
        max_repos = 100
        ai_predictions_35 = compute_ai_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-3.5-turbo", 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000,
            max_repos=max_repos
        )
        ai_predictions_4o = compute_ai_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-4o", 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000,
            max_repos=max_repos
        )
        ai_predictions_5 = compute_ai_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-5-mini", 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000,
            max_repos=max_repos
        )
        update_predictions_in_db(ai_predictions_35, db_file, "affiliation_prediction_gpt_35_turbo")
        update_predictions_in_db(ai_predictions_4o, db_file, "affiliation_prediction_gpt_4o")
        update_predictions_in_db(ai_predictions_5, db_file, "affiliation_prediction_gpt_5_mini")

        # Project Type classification (only for gpt-5-mini)
        ai_type_predictions_5 = compute_ai_type_predictions(
            acronym, config_file, db_file, client, 
            model="gpt-5-mini", 
            truncation_type="start_end",
            truncate=20000,
            start_length=15000,
            end_length=5000,
            affiliated_only=True,
            affiliation_threshold=0.4,
            max_repos=max_repos
        )
        update_predictions_in_db(ai_type_predictions_5, db_file, "type_prediction_gpt_5_mini")

        type_classifier_accuracy(type_test_set, ai_type_predictions_5, acronym)

    create_roc_curves(university_acronyms, curves_to_plot=['SBC', 'embeddings', 'gpt-35-turbo', 'gpt-4o', 'gpt-5-mini'])
        
if __name__ == "__main__":
    filter_repos()

