#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:50:46 2025

@author: juanitagomez
"""

from repofinder.scraping.get_repositories import get_repositories
from repofinder.scraping.json_to_db import create_and_populate_database
from repofinder.scraping.get_contributors import get_contributor_data
from repofinder.scraping.get_organizations import get_organization_data
from repofinder.scraping.get_repo_extras import get_features_data
from dotenv import load_dotenv
import os
import time

DOTENV = ".env" 
load_dotenv(DOTENV)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'Juanis2112'
}

FEATURES = [
    "release_downloads",
    "readme",
    "code_of_conduct",
    "contributing",
    "security_policy",
    "issue_templates",
    "pull_request_template",
    "subscribers_count",
]

def scrape(university_acronyms=["UCM"]):
    for acronym in university_acronyms:
    
        config_file= f"config/config_{acronym}.json"
        repo_file=f"Data/json/repository_data_{acronym}.json"
        db_file = f"Data/db/repository_data_{acronym}_database.db"
    
        # Scraping
        times = []  
        
        start_time = time.time()
        print('Finding repositories')
        get_repositories(config_file, HEADERS)
        end_time = time.time()
        times.append(end_time - start_time)
        print('Repositories collected')
        
        # start_time = time.time()
        create_and_populate_database(repo_file, db_file)
        end_time = time.time()
        times.append(end_time - start_time)
        print('Database populated done')
        
        start_time = time.time()
        get_organization_data(repo_file, db_file, HEADERS)
        end_time = time.time()
        times.append(end_time - start_time)
        print('Organizations done')
        
        start_time = time.time()
        get_features_data(repo_file, db_file, HEADERS, FEATURES)
        end_time = time.time()
        times.append(end_time - start_time)
        print('Extra features done')
        
        start_time = time.time()
        get_contributor_data(repo_file, db_file, HEADERS)
        end_time = time.time()
        times.append(end_time - start_time)
        print('Contributors done')
    
        # Print execution times
        print("Execution times (seconds):", times)

if __name__ == "__main__":
    scrape()









