#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:03:22 2024

@author: juanitagomez
"""

import pandas as pd
import sqlite3
from scraping.repo_scraping_utils import github_api_request, get_next_link

#TODO: Figure out how to get duplicates

def get_contributors(owner, repo_name, headers):
    """
    Retrieves the list of contributors for a given repository.

    Args:
        owner (str): Owner of the repository.
        repo_name (str): Name of the repository.
        headers (dict): HTTP headers for the request.

    Returns:
        list: A list of contributors.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contributors?q=contributions&order=desc"
    params = {'per_page': 100}
    contributors = []
    while url:
        try:
            contributors_data, headers_response = github_api_request(url, headers, params)
        except:
            break
        if contributors_data:
            contributors.extend(contributors_data)
            next_url = get_next_link(headers_response)
            url = next_url
            params = None
        else:
            break
    return contributors if contributors else []

def get_contributor_details(username, headers):
    """
    Retrieves detailed information about a contributor.

    Args:
        username (str): The GitHub username of the contributor.
        headers (dict): HTTP headers for the request.

    Returns:
        dict: A dictionary containing contributor details.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        contributor_data, _ = github_api_request(url, headers)
        return {
            "login": contributor_data.get("login"),
            "name": contributor_data.get("name"),
            "bio": contributor_data.get("bio"),
            "location": contributor_data.get("location"),
            "company": contributor_data.get("company"),
            "email": contributor_data.get("email"),
            "twitter": contributor_data.get("twitter_username"),
            "organizations": contributor_data.get("organizations_url"),  # This is a URL, requires additional fetch
        }
    except Exception as e:
        print(f"Error fetching details for user {username}: {e}")
        return None


def get_contributor_data(repo_file, db_file, headers):

    # TODO: This should probably read the database instead 
    repo_df = pd.read_json(repo_file)
    repo_df = repo_df.drop_duplicates(subset=['full_name'])
    repo_df = repo_df.reset_index(drop=True)
    repo_df["contributors"] = None
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE repositories ADD COLUMN contributors TEXT;")  # Adjust the column type as needed
    except:
        pass
   
    # Create the contributors table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contributors (
        login TEXT PRIMARY KEY,
        name TEXT,
        bio TEXT,
        location TEXT,
        company TEXT,
        email TEXT,
        twitter TEXT,
        organizations TEXT
    )
    """)
    
    # Create the contributions table (join table)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS contributions (
        repository_name TEXT NOT NULL,
        contributor_login TEXT NOT NULL,
        PRIMARY KEY (repository_name, contributor_login)
    )
    """)
    
    for i in range(len(repo_df)):  #TODO: Fix API rate limits
        full_name = repo_df["full_name"][i]
        owner, repo_name = full_name.split("/")
        contributors = get_contributors(owner, repo_name, headers)
        contributors_login = []
        
        for contributor in contributors:
            contributor_login = contributor['login']
            details = get_contributor_details(contributor_login, headers)
            
            if details:
            #     # Fetch organizations data if needed
            #     organizations_url = details.pop("organizations", None)
            #     if organizations_url:
            #         try:
            #             org_data, _ = github_api_request(organizations_url, headers)
            #             organizations = ", ".join(org.get("login", "") for org in org_data)
            #             details["organizations"] = organizations
            #         except Exception as e:
            #             print(f"Error fetching organizations for user {contributor_login}: {e}")
            #             details["organizations"] = None
            #             print(i + '/' + len(repo_df))
            #             return i

                
                conn.execute("""
                    INSERT OR REPLACE INTO contributors (login, name, bio, location, company, email, twitter)
                    VALUES (:login, :name, :bio, :location, :company, :email, :twitter)
                """, details)
                

            conn.execute("INSERT OR IGNORE INTO contributions (repository_name, contributor_login) VALUES (?, ?)", (full_name, contributor_login))
            contributors_login.append(contributor_login)

                           
        # Update database with this new column
        repo_df["contributors"][i] = contributors_login
        contributors_login_string = str(contributors_login)
        # This is to add the contributors as a list in the repositories table 
        conn.execute(
            "UPDATE repositories SET contributors = ? WHERE full_name = ?;",
            (contributors_login_string, full_name)
        )
        conn.commit()
        print(str(i)+"/"+str(len(repo_df)))

    conn.close()
# TODO: Should I try to build a JSON object with this too?
    return repo_df

