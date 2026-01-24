#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import json
import pandas as pd
import os


def compute_predictions_sbc(acronym, config_file, db_file, subset=None):
    """
    Computes repository affiliation scores based on organization, contributor, and repository metadata.

    The function connects to a SQLite database containing repository, organization, and contributor
    information. It uses university-specific keywords and identifiers (loaded from a JSON config file)
    to compute heuristic scores for each repository indicating its likelihood of being affiliated
    with a given university. Scores are normalized between 0 and 1.

    Args:
        acronym (str): University acronym used in the output file path (e.g., 'UCSC').
        config_file (str): Path to a JSON config file with university-specific metadata, including:
            - UNIVERSITY_NAME
            - UNIVERSITY_ACRONYM
            - UNIVERSITY_EMAIL_DOMAIN
            - UNIVERSITY_WEBSITE_URL
            - ADDITIONAL_QUERIES (list of extra keywords)
        db_file (str): Path to the SQLite database containing the repository metadata.
        subset (str, optional): Path to CSV file with html_url column to filter repositories.
            If provided, only computes predictions for repositories in the subset.
            If None, processes all repositories (default: None).
    Returns:
        str: Path to the generated CSV file containing the repository-level scores.
    """
    
    # Load the configuration file
    with open(config_file) as config_file:
        config = json.loads(config_file.read())

    university_name = config["UNIVERSITY_NAME"].lower()
    university_acronym = config["UNIVERSITY_ACRONYM"].lower()
    university_email_domain = config["UNIVERSITY_EMAIL_DOMAIN"].lower()
    university_website_url = config["UNIVERSITY_WEBSITE_URL"].lower()
    additional_queries = [q.lower() for q in config["ADDITIONAL_QUERIES"]]
    keywords = [university_name, university_acronym] + additional_queries

    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    def calculate_keyword_score(text, keywords, base_score):
        score = 0
        for keyword in keywords:
            score += text.lower().count(keyword) * base_score
        return score

    # Load subset if provided
    subset_urls = None
    if subset:
        subset_df = pd.read_csv(subset)
        if 'html_url' not in subset_df.columns:
            raise ValueError(f"Subset CSV must contain 'html_url' column. Found columns: {list(subset_df.columns)}")
        subset_urls = set(subset_df['html_url'].dropna().unique())
        print(f"Filtering to {len(subset_urls)} repositories from subset")

    # Retrieve all repositories (or filtered by subset)
    if subset_urls:
        # Filter repositories to only those in subset
        # Convert set to tuple for SQL IN clause
        subset_urls_list = list(subset_urls)
        placeholders = ','.join(['?' for _ in subset_urls_list])
        query = f"SELECT full_name, html_url FROM repositories WHERE html_url IN ({placeholders})"
        cursor.execute(query, subset_urls_list)
    else:
        cursor.execute("SELECT full_name, html_url FROM repositories")
    repositories = cursor.fetchall()

    results = []

    for full_name, html_url in repositories:
        organization_score = 0
        contributor_score = 0
        repository_score = 0

        cursor.execute("""
            SELECT organization, owner, contributors, readme, description, homepage
            FROM repositories
            WHERE full_name = ?
        """, (full_name,))
        repo_data = cursor.fetchone()

        if repo_data:
            is_organization, owner, contributors, readme, description, homepage = repo_data
            readme = readme.lower() if readme else ""
            description = description.lower() if description else ""
            homepage = homepage.lower() if homepage else ""
            full_name_lower = full_name.lower()

            # ---- Compute repository_score ----
            fields = [readme, description, full_name_lower, homepage]
            if university_email_domain in homepage:
                repository_score = 100
            for field in fields:
                repository_score += calculate_keyword_score(field, keywords, 20)

            # ---- Compute organization_score ----
            if is_organization:
                cursor.execute("""
                    SELECT name, description, company, url, email
                    FROM organizations
                    WHERE login = ?
                """, (owner,))
                organization = cursor.fetchone()

                if organization:
                    name, org_description, company, url, email = organization
                    url = url.lower() if url else ""
                    email = email.lower() if email else ""
                    name = name.lower() if name else ""
                    org_description = org_description.lower() if org_description else ""

                    if university_website_url in url or university_email_domain in email:
                        organization_score = 100
                    else:
                        organization_score += calculate_keyword_score(name, keywords, 30)
                        organization_score += calculate_keyword_score(org_description, keywords, 30)

            # ---- Compute contributor_score ----
            contributors_list = contributors.split(",") if contributors else []
            for contributor in contributors_list:
                cursor.execute("""
                    SELECT email, name, bio, location, company
                    FROM contributors
                    WHERE login = ?
                """, (contributor,))
                contributor_data = cursor.fetchone()

                if contributor_data:
                    email, name, bio, location, company = contributor_data
                    fields = [email, name, bio, location, company]

                    for field in fields:
                        if field:
                            field = field.lower()
                            if university_email_domain in field:
                                contributor_score += 50
                            else:
                                contributor_score += calculate_keyword_score(field, keywords, 20)

        # ---- Combine and normalize scores ----
        total_score = min(organization_score + contributor_score + repository_score, 100)
        result = {
            "html_url": html_url,
            "organization_score": round(organization_score / 100, 2),
            "contributor_score": round(contributor_score / 100, 2),
            "repository_score": round(repository_score / 100, 2),
            "total_score": round(total_score / 100, 2),
        }
        results.append(result) 
    
    # Add "_subset" suffix to filename if subset is provided
    subset_suffix = "_subset" if subset else ""
    output_path = f"results/{acronym}/predictions_sbc_{acronym}{subset_suffix}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return output_path
    
    
    