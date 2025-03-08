#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:03:22 2024

@author: juanitagomez
"""

import pandas as pd
import sqlite3
import base64
import time
import os
from dotenv import load_dotenv
from scraping.repo_scraping_utils import github_api_request


def get_readme_content(full_name, headers):
    """
    Fetches the README content for a given repository.

    Parameters
    ----------
    full_name : str
        The full name of the repository (e.g., "owner/repo").
    headers : dict
        HTTP headers for the request.

    Returns
    -------
    str or None
        The decoded README content if available, otherwise None.
    """
    url = f"https://api.github.com/repos/{full_name}/readme"
    try:
        readme, _ = github_api_request(url, headers)
        return base64.b64decode(readme['content']).decode('utf-8')

    except Exception as e:
            print(f"Failed to decode README for {full_name}: {e}")
            return None
    
    
    
# def get_readme_content(full_name, headers) -> str:
#     """
#     Fetch the README content for a specific repository.

#     :param owner: Repository owner.
#     :param repo: Repository name.
#     :return: Decoded README content.
#     """
#     url = f"{GITHUB_API_URL}/repos/{full_name}/readme"
#     try:      
#         response = requests.get(url, headers=headers)
#         content = response.json().get("content", "")
#         return base64.b64decode(content).decode('utf-8')
#     except Exception as e:
#         print(f"Error fetching README for {full_name}: {response.status_code} {response.text}")
#         return None    
    

def get_readme_data(repo_file, db_file, headers):
    """
    Fetches and stores the README content for repositories listed in a JSON file.
    
    Parameters
    ----------
    repo_file : str
        Path to the JSON file containing repository information, including the "full_name" field.
    db_file : str
        Path to the SQLite database file where the README content will be stored.
    headers : dict
        HTTP headers for authentication and API requests.
    
    Returns
    -------
    None
        The function updates the database in place and prints progress for each repository.
    
    Notes
    -----
    - Assumes the database has a `repositories` table with a `full_name` column.
    - Calls `get_readme_content(full_name, headers)` to fetch the README from GitHub.
    - Commits updates to the database after each repository is processed.
    - Catches and ignores errors during database updates.
    """
    repo_df = pd.read_json(repo_file)
    repo_df = repo_df.drop_duplicates(subset=['full_name'])
    repo_df = repo_df.reset_index(drop=True)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
    # Ensure the repositories table has the organization column
        cursor.execute("ALTER TABLE repositories ADD COLUMN readme TEXT;")  # Adjust the column type as needed
    except:
        pass
    
    for i in range(len(repo_df)):
        full_name = repo_df["full_name"][i]
        readme = get_readme_content(full_name, headers)
        try:
            conn.execute(
                "UPDATE repositories SET readme = ? WHERE full_name = ?;",
                (readme, full_name)
            )  # Adjust the column type as needed
        except:
            pass
    
        conn.commit()
        print(str(i+1)+"/"+str(len(repo_df)))

#------------------------------------------------------------------------------

# Uncomment for running this file only

# #university_acronyms = ['UCSB', 'UCD', 'UCLA', 'UCSD', 'UCB', 'UCSC']


# for acronym in university_acronyms:
    
#     start_time = time.time()
#     repo_file = f'../Data/json/repository_data_{acronym}.json'
#     db_file = f'../Data/db/repository_data_{acronym}_database.db'

#     get_readme(repo_file, db_file, headers)
    
#     end_time = time.time()
#     print("--- %s seconds ---" % (end_time - start_time))    




