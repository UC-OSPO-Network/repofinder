#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:03:22 2024

@author: juanitagomez
"""

import pandas as pd
import sqlite3
import base64
from repofinder.scraping.repo_scraping_utils import github_api_request


def get_feature_content(full_name, headers, feature):
    """
    Tries to fetch the content for a given feature (like README, license, etc.)
    from either the GitHub API or from fallback paths in the repo contents.

    Parameters
    ----------
    full_name : str
        The repository full name, e.g., "owner/repo".
    feature : str
        The feature to retrieve (e.g., "readme", "contributing").
    headers : dict
        Headers with auth and API options.

    Returns
    -------
    str or None
        Content string if available, otherwise None.
    """

    base_url = f"https://api.github.com/repos/{full_name}"

    # Direct API endpoints (with special treatment)
    if feature == "license":
        try:
            repo, _ = github_api_request(base_url, headers)
            return repo.get("license", {}).get("key")
        except Exception as e:
            print(f"Failed to fetch license for {full_name}: {e}")
            return None
 
    if feature == "subscribers_count": # These are the watchers
        try:
            repo_data, _ = github_api_request(f"{base_url}", headers)
            return repo_data.get("subscribers_count", 0)
        except Exception as e:
            print(f"Failed to fetch subscribers count for {full_name}: {e}")
            return None

    if feature == "readme":
        try:
            res, _ = github_api_request(f"{base_url}/readme", headers)
            return base64.b64decode(res['content']).decode('utf-8')
        except Exception as e:
            print(f"Failed to decode README for {full_name}: {e}")
            return None

    if feature == "release_downloads":
        try:
            releases, _ = github_api_request(f"{base_url}/releases", headers)
            total_downloads = 0
            for release in releases:
                for asset in release.get("assets", []):
                    total_downloads += asset.get("download_count", 0)
            return total_downloads
        except Exception as e:
            print(f"Failed to fetch release downloads for {full_name}: {e}")
            return None
    

    # Optional API endpoint for structured data (e.g., code of conduct)
    feature_api_endpoints = {
        "code_of_conduct": f"{base_url}/community/code_of_conduct",
        "security_policy": f"{base_url}/security/policy"
    }

    if feature in feature_api_endpoints:
        try:
            res, _ = github_api_request(feature_api_endpoints[feature], headers)
            return res.get("key") or res.get("content")  # fallback to raw content if no key
        except:
            pass  # Try fallback paths next

    # Fallback file paths to try for each feature
    fallback_paths = {
        "code_of_conduct": [
            ".github/CODE_OF_CONDUCT.md",
            "CODE_OF_CONDUCT.md"
        ],
        "contributing": [
            ".github/CONTRIBUTING.md",
            "CONTRIBUTING.md"
        ],
        "security_policy": [
            ".github/SECURITY.md",
            "SECURITY.md"
        ],
        "issue_templates": [
            ".github/ISSUE_TEMPLATE/config.yml",  # GitHub issue template config
            ".github/ISSUE_TEMPLATE"
        ],
        "pull_request_template": [
            ".github/PULL_REQUEST_TEMPLATE.md",
            "PULL_REQUEST_TEMPLATE.md"
        ]
    }

    for path in fallback_paths.get(feature, []):
        try:
            res, _ = github_api_request(f"{base_url}/contents/{path}", headers)
            if isinstance(res, list):
                # If it's a directory listing, just return True (exists)
                return "Directory exists"
            return base64.b64decode(res['content']).decode('utf-8')
        except:
            continue

    print(f"{feature} not found for {full_name}.")
    return None

def get_features_data(repo_file, db_file, headers, features_list):
    """
    Iterates over repositories and stores specified features in the database.

    Parameters
    ----------
    repo_file : str
        Path to the JSON file containing the list of repositories.
    db_file : str
        Path to the SQLite database.
    headers : dict
        HTTP headers for GitHub API.
    features_list : list
        List of features to retrieve and store (e.g., ['readme', 'license']).
    """
    repo_df = pd.read_json(repo_file).drop_duplicates(subset=["full_name"]).reset_index(drop=True)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Add columns if they don't exist
    for feature in features_list:
        try:
            if feature == "subscribers_count" or feature== "release_downloads":
                cursor.execute(f"ALTER TABLE repositories ADD COLUMN {feature} INTEGER;")
            else:
                cursor.execute(f"ALTER TABLE repositories ADD COLUMN {feature} TEXT;")
        except sqlite3.OperationalError:
            pass

    for i, row in repo_df.iterrows():
        full_name = row["full_name"]
        values = []
        for feature in features_list:
            result = get_feature_content(full_name, headers, feature)
            values.append(str(result) if result is not None else None)
        set_clause = ", ".join([f"{feature} = ?" for feature in features_list])
        sql = f"UPDATE repositories SET {set_clause} WHERE full_name = ?"
        cursor.execute(sql, (*values, full_name))
        conn.commit()
        print(f"{i+1}/{len(repo_df)}: Processed {full_name}")

    conn.close()

