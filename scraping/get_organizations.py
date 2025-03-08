#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:03:22 2024

@author: juanitagomez
"""

import pandas as pd
import sqlite3
import time
import os
from dotenv import load_dotenv
from scraping.repo_scraping_utils import github_api_request


def get_organization_details(org_login, headers):
    """
    Retrieves detailed information about an organization.

    Args:
        org_login (str): The GitHub login of the organization.
        headers (dict): HTTP headers for the request.

    Returns:
        dict: A dictionary containing organization details.
    """
    url = f"https://api.github.com/orgs/{org_login}"
    try:
        org_data, _ = github_api_request(url, headers)
        return {
            "login": org_data.get("login"),
            "name": org_data.get("name"),
            "description": org_data.get("description"),
            "company": org_data.get("company"),
            "created_at": org_data.get("created_at"),
            "updated_at": org_data.get("updated_at"),
            "location": org_data.get("location"),
            "email": org_data.get("email"),
            "url": org_data.get("blog"),
        }
    except Exception as e:
        print(f"Error fetching details for organization {org_login}: {e}")
        return None



def get_organization_data(repo_file, db_file, headers):
    
    # TODO: Should probably read the db instead
    repo_df = pd.read_json(repo_file)
    repo_df = repo_df.drop_duplicates(subset=['full_name'])
    repo_df = repo_df.reset_index(drop=True)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
    # Ensure the repositories table has the organization column
        cursor.execute("ALTER TABLE repositories ADD COLUMN organization TEXT;")  # Adjust the column type as needed
    except:
        pass
    # Create or ensure the organizations table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS organizations (
            login TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            location TEXT,
            company TEXT,
            email TEXT,
            url TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    
    for i in range(len(repo_df)):
        full_name = repo_df["full_name"][i]
        owner = repo_df['owner'][i]['login']
        owner_url = f"https://api.github.com/users/{owner}"

        try:
            # Fetch owner data to check its type
            owner_data, _ = github_api_request(owner_url, headers)
            owner_type = owner_data.get("type")
            
            if owner_type == "Organization":
                # Mark the repository as owned by an organization
                repo_df.at[i, "organization"] = True
                conn.execute(
                "UPDATE repositories SET organization = ? WHERE full_name = ?;",
                (True, full_name)
)

                # Fetch organization details
                details = get_organization_details(owner, headers)
                if details:
                    # Insert organization details into the database
                    conn.execute("""
                        INSERT OR REPLACE INTO organizations 
                        (login, name, description, location, company, email, url, created_at, updated_at)
                        VALUES 
                        (:login, :name, :description, :location, :company, :email, :url, :created_at, :updated_at)
                    """, details)
                    
        except Exception as e:
            print(f"Error processing owner {owner}: {e}")

        
        # Commit changes to the database
        conn.commit()
        print(f"Processed {i + 1}/{len(repo_df)} repositories.")

    conn.close()
# TODO: Should I try to build a JSON object with this too?
    return repo_df


#------------------------------------------------------------------------------

# Uncomment for running this file only

#university_acronyms = ['UCSB', 'UCD', 'UCLA', 'UCSD', 'UCB', 'UCSC']


# for acronym in university_acronyms:
    
#     start_time = time.time()
#     repo_file = f'../Data/json/repository_data_{acronym}.json'
#     db_file = f'../Data/db/repository_data_{acronym}_database.db'

#     get_organization_data(repo_file, db_file)
    
#     end_time = time.time()
#     print("--- %s seconds ---" % (end_time - start_time))    


