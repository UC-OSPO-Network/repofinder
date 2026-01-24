#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    
    if feature == "readme":
        try:
            res, _ = github_api_request(f"{base_url}/readme", headers)
            return base64.b64decode(res['content']).decode('utf-8')
        except Exception as e:
            print(f"Failed to decode README for {full_name}: {e}")
            return None

    
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
    
    
    feature_api_endpoints = {
        "code_of_conduct": f"{base_url}/community/code_of_conduct",
        "security_policy": f"{base_url}/security/policy"
    }

    # Community files endpoints
    
    community_url = f"{base_url}/community/profile"
    community = ["code_of_conduct_file", "contributing", "issue_template", "pull_request_template"]


    if feature in community:
        try:
            res, _ = github_api_request(community_url, headers)

            file_info = res['files'].get(feature)
            if not file_info:
                return None

            res, _ = github_api_request(file_info["url"], headers)
            return base64.b64decode(res["content"]).decode("utf-8")
            
        except Exception as e:
            print(e)
            
    if feature == "security_policy":

        fallback_paths = [
            ".github/SECURITY.md",
            "SECURITY.md",
            "docs/SECURITY.md",
        ]

        for path in fallback_paths:
            try:
                res, status = github_api_request(f"{base_url}/contents/{path}", headers)
                if res is None or status == 404:
                    continue
                return base64.b64decode(res["content"]).decode("utf-8")

            except Exception as e:
                print(e)
                return None

def get_features_data(repo_file, db_file, headers, features_list):
    """
    Iterates over repositories and stores specified features in the database.
    Only processes repositories that are not archived, have size > 0, are not forks, and are not templates.

    Parameters
    ----------
    repo_file : str
        Path to the JSON file containing the list of repositories (unused, reads from DB instead).
    db_file : str
        Path to the SQLite database.
    headers : dict
        HTTP headers for GitHub API.
    features_list : list
        List of features to retrieve and store (e.g., ['readme', 'license']).
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Check if features_list is empty
    if not features_list:
        print("No features to process. Exiting.")
        conn.close()
        return
    
    # Add columns if they don't exist (must be done before querying)
    for feature in features_list:
        try:
            if feature == "subscribers_count" or feature== "release_downloads":
                cursor.execute(f"ALTER TABLE repositories ADD COLUMN {feature} INTEGER;")
            else:
                cursor.execute(f"ALTER TABLE repositories ADD COLUMN {feature} TEXT;")
        except sqlite3.OperationalError:
            pass
    
    # Build query to check which features are missing for each repository
    # We'll filter for repos where at least one feature is missing
    feature_conditions = " OR ".join([f"({feature} IS NULL OR {feature} = '')" for feature in features_list])
    
    # Read repositories from database, filtering for non-archived, size > 0, not a fork, not a template
    # and where at least one feature is missing
    query = f"""
        SELECT full_name, {', '.join(features_list)}
        FROM repositories 
        WHERE (archived = 0 OR archived = FALSE OR archived IS NULL)
          AND (size > 0 OR size IS NULL)
          AND (fork = 0 OR fork = FALSE OR fork IS NULL)
          AND (is_template = 0 OR is_template = FALSE OR is_template IS NULL)
          AND ({feature_conditions})
    """
    repo_df = pd.read_sql_query(query, conn)

    total = len(repo_df)
    print(f"Processing {total} repositories for features: {', '.join(features_list)}")
    
    if total == 0:
        print("No repositories found matching the criteria (non-archived and size > 0).")
        conn.close()
        return
    
    for i, row in repo_df.iterrows():
        full_name = row["full_name"]
        progress_pct = ((i + 1) / total) * 100
        print(f"[{i+1}/{total} ({progress_pct:.1f}%)] Processing {full_name}...", end=' ', flush=True)

        # If release_downloads is already present (non-null/non-empty), skip
        # gathering any other attributes for this repository to save requests.
        if "release_downloads" in features_list:
            existing_dl = row.get("release_downloads")
            if pd.notna(existing_dl) and str(existing_dl).strip() != "":
                print(" skipped (release_downloads already present)")
                continue
        
        values = []
        features_to_update = []
        for feature in features_list:
            # Check if feature is already in database (not NULL and not empty)
            existing_value = row[feature]
            if pd.notna(existing_value) and str(existing_value).strip() != '':
                # Feature already exists, skip API request and keep existing value
                values.append(existing_value)
                features_to_update.append(feature)
            else:
                # Feature is missing, fetch from API
                result = get_feature_content(full_name, headers, feature)
                values.append(str(result) if result is not None else None)
                features_to_update.append(feature)
        # Only update if we have features to update
        if features_to_update:
            set_clause = ", ".join([f"{feature} = ?" for feature in features_to_update])
            sql = f"UPDATE repositories SET {set_clause} WHERE full_name = ?"
            cursor.execute(sql, (*values, full_name))
            conn.commit()
        print("Done")
    
    print(f"\nCompleted: Processed {total}/{total} repositories")

    conn.close()

