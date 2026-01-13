#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from dotenv import load_dotenv

from repofinder.scraping.get_repositories import get_repositories
from repofinder.scraping.json_to_db import create_and_populate_database
from repofinder.scraping.get_contributors import get_contributor_data
from repofinder.scraping.get_organizations import get_organization_data
from repofinder.scraping.get_repo_extras import get_features_data

DOTENV = ".env"
load_dotenv(DOTENV)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
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

def scrape(university_acronyms):
    for acronym in university_acronyms:
        config_file = f"config/config_{acronym}.json"
        repo_file = f"Data/json/repository_data_{acronym}.json"
        db_file = f"Data/db/repository_data_{acronym}_database.db"

        # Check if config exists to prevent crash
        if not os.path.exists(config_file):
            print(f"Error: Configuration file {config_file} not found. Skipping {acronym}.")
            continue

        print(f"--- Processing {acronym} ---")
        times = []  
        
        start_time = time.time()
        print('Finding repositories')
        get_repositories(config_file, HEADERS)
        times.append(time.time() - start_time)
        print('Repositories collected')
        
        start_time = time.time()
        print('Populating database')
        create_and_populate_database(repo_file, db_file)
        times.append(time.time() - start_time)
        print('Database populated')
        
        start_time = time.time()
        print('Fetching organization data')
        get_organization_data(repo_file, db_file, HEADERS)
        times.append(time.time() - start_time)
        print('Organizations done')
        
        start_time = time.time()
        print('Fetching extra features')
        get_features_data(repo_file, db_file, HEADERS, FEATURES)
        times.append(time.time() - start_time)
        print('Extra features done')
        
        start_time = time.time()
        print('Fetching contributor data')
        get_contributor_data(repo_file, db_file, HEADERS)
        times.append(time.time() - start_time)
        print('Contributors done')
    
        print("Execution times (seconds):", times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape GitHub repositories for University OSPOs.")
    parser.add_argument(
        "--universities", 
        nargs="+", 
        default=["UCSD"], 
        help="List of university acronyms to scrape (e.g., --universities UCSD UCSC)"
    )
    args = parser.parse_args()
    
    scrape(args.universities)








