#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
from repofinder.scraping.repo_scraping_utils import github_api_request


def get_organization_details(org_login, headers, rate_limiter=None):
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
        org_data, _ = github_api_request(url, headers, rate_limiter=rate_limiter)
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
    """
    Processes a list of repositories to identify those owned by organizations
    and stores organization metadata in a SQLite database.
    Only processes repositories that are not archived, have size > 0, are not forks, and are not templates.
    
    This function:
    - Reads repository metadata from the database.
    - Identifies which repositories are owned by GitHub organizations.
    - Updates the 'repositories' table to mark organizational ownership.
    - Creates or updates an 'organizations' table with detailed organization info.
    
    Args:
        repo_file (str): Path to the JSON file (unused, reads from DB instead).
        db_file (str): Path to the SQLite database file.
        headers (dict): HTTP headers for authenticated GitHub API requests.
    
    Returns:
        pd.DataFrame: A DataFrame of the repositories with an added 'organization' column.
    """
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Read repositories from database, filtering for non-archived, size > 0, not a fork, and not a template
    query = """
        SELECT full_name, owner
        FROM repositories 
        WHERE (archived = 0 OR archived = FALSE OR archived IS NULL)
          AND (size > 0 OR size IS NULL)
          AND (fork = 0 OR fork = FALSE OR fork IS NULL)
          AND (is_template = 0 OR is_template = FALSE OR is_template IS NULL)
    """
    repo_df = pd.read_sql_query(query, conn)
    repo_df = repo_df.reset_index(drop=True)
    
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

    # Process sequentially (no multithreading)
    total_repos = len(repo_df)
    print(f"Processing {total_repos} repositories for organization data...")
    
    processed_count = 0
    for idx, row in repo_df.iterrows():
        full_name = row["full_name"]
        owner = row['owner']  # owner is already a string from the database
        owner_url = f"https://api.github.com/users/{owner}"
        
        try:
            owner_data, _ = github_api_request(owner_url, headers)
            if not owner_data:
                processed_count += 1
                continue
            
            owner_type = owner_data.get("type")
            
            if owner_type == "Organization":
                details = get_organization_details(owner, headers)
                repo_df.at[idx, "organization"] = True
                conn.execute("UPDATE repositories SET organization = ? WHERE full_name = ?;", (True, full_name))
                if details:
                    conn.execute("""
                        INSERT OR REPLACE INTO organizations 
                        (login, name, description, location, company, email, url, created_at, updated_at)
                        VALUES 
                        (:login, :name, :description, :location, :company, :email, :url, :created_at, :updated_at)
                    """, details)
        except Exception as e:
            print(f"Error processing owner {owner}: {e}")
        
        processed_count += 1
        if processed_count % 50 == 0 or processed_count == total_repos:
            conn.commit()
            print(f"Processed {processed_count}/{total_repos} repositories.")
    
    conn.commit()  # Final commit
    print(f"Completed: {processed_count}/{total_repos} repositories processed.")

    conn.close()
# TODO: Should I try to build a JSON object with this too?
    return repo_df

