#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:50:46 2025

@author: juanitagomez
"""

from dotenv import load_dotenv
from scraping.repofinder import repo_finder
from scraping.jason_to_db import create_and_populate_database
from scraping.get_contributors import get_contributor_data
from scraping.get_organizations import get_organization_data
from scraping.get_repo_extras import get_readme_data
import os
import time

path = os.getenv('PWD')
DOTENV = ".env" 
load_dotenv(DOTENV)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'Juanis2112'
}
university_acronyms = ['UCSD']

for acronym in university_acronyms:

    config_file= f"{path}/config/config_{acronym}.json"
    repo_file=f"{path}/Data/json/repository_data_{acronym}.json"
    db_file = f"{path}/Data/db/repository_data_{acronym}_database.db"

    # Scraping
    times = []  
    
    start_time = time.time()
    repo_finder(config_file, path, headers)
    end_time = time.time()
    times.append(end_time - start_time)
    print('Repo finder done')
    
    start_time = time.time()
    create_and_populate_database(repo_file, db_file)
    end_time = time.time()
    times.append(end_time - start_time)
    print('Database populated done')
    
    start_time = time.time()
    get_organization_data(repo_file, db_file, headers)
    end_time = time.time()
    times.append(end_time - start_time)
    print('Organizations done')
    
    start_time = time.time()
    get_readme_data(repo_file, db_file, headers)
    end_time = time.time()
    times.append(end_time - start_time)
    print('Readme done')
    
    start_time = time.time()
    get_contributor_data(repo_file, db_file, headers)
    end_time = time.time()
    times.append(end_time - start_time)
    print('Contributors done')
    
    # Print execution times
    print("Execution times (seconds):", times)


    
    
    