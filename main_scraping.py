#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from repofinder.scraping.search_repositories import search_repositories
from repofinder.scraping.search_organizations import search_organizations
from repofinder.scraping.search_users import search_users
from repofinder.scraping.repo_scraping_utils import get_repositories_from_organizations, get_repositories_from_users
from repofinder.scraping.json_to_db import create_and_populate_database, populate_organizations, populate_users
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
}

FEATURES = [
    "readme",
    "subscribers_count",
    "release_downloads",
    "code_of_conduct_file",
    "contributing",
    "issue_templates",
    "pull_request_template",
    "security_policy",
]

def scrape(university_acronyms=["UCSB", "UCSC", "UCSD"]):
    for acronym in university_acronyms:
    
        config_file= f"config/config_{acronym}.json"
        repo_file=f"Data/json/repository_data_{acronym}.json"
        org_file=f"Data/json/organization_data_{acronym}.json"
        repo_from_orgs_file=f"Data/json/repository_data_org_scraping_{acronym}.json"
        user_file=f"Data/json/user_data_{acronym}.json"
        repo_from_users_file=f"Data/json/repository_data_user_scraping_{acronym}.json"
        db_file = f"Data/db/repository_data_{acronym}_database.db"
    
        print('Finding repositories')
        search_repositories(config_file, HEADERS)
        print('Repositories collected')

        create_and_populate_database(repo_file, db_file, search_method='repository_search')
        print('Database populated done')
        
        print('Finding organizations')
        search_organizations(config_file, HEADERS)
        populate_organizations (org_file, db_file)
        print('Organizations collected')

        print('Finding repositories from organizations')
        get_repositories_from_organizations(acronym, org_file, HEADERS)
        print('Repositories from organizations collected')
        
        create_and_populate_database(repo_from_orgs_file, db_file, search_method='organization_search')
        print('Repositories from organizations populated in database')
        
        print('Finding users')
        search_users(config_file, HEADERS)
        populate_users(user_file, db_file)
        print('Users collected')
        
        print('Finding repositories from users')
        get_repositories_from_users(acronym, user_file, HEADERS)
        print('Repositories from users collected')
        
        create_and_populate_database(repo_from_users_file, db_file, search_method='user_search')
        print('Repositories from users populated in database')

        get_features_data(repo_file, db_file, HEADERS, FEATURES)
        print('Extra features done')

        get_organization_data(repo_file, db_file, HEADERS)
        print('Organizations done')
        
        get_contributor_data(repo_file, db_file, HEADERS)
        print('Contributors done')

if __name__ == "__main__":
    scrape()









