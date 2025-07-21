#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sqlite3
import csv
import numpy as np
import re
import os

# Function to check if a field contains university-specific information
def contains_university_info(field, attribute):
    """
    Checks if a specific attribute (e.g., university acronym, website URL) is in the given field.
    
    Parameters:
        field (str): The field to check (e.g., organization URL, repository description).
        attribute (str): The specific university attribute to look for.
    
    Returns:
        int: 1 if the attribute is found, 0 otherwise.
    """
    if field is None:
        return 0
    field = field.lower()
    return int(attribute.lower() in field)


def contains_university_name(field, university_name):
    """
    Checks if a field contains a variation of the university name.
    
    Parameters:
        field (str): The text field to search.
        university_name (str): The full university name.
    
    Returns:
        int: 1 if a variation of the university name is found, 0 otherwise.
    """
    if field is None:
        return 0
    
    field = field.lower()

    # Convert the university name into a regex-friendly format
    cleaned_name = re.sub(r"[^a-zA-Z\s]", "", university_name)
    name_parts = re.split(r'\s+', cleaned_name.strip().lower())  # Split by spaces
    regex_parts = [fr"{part}\s*[^a-zA-Z0-9]*\s*" for part in name_parts]
    dynamic_regex = ".*" + "".join(regex_parts) + ".*"
    # Match against the field
    return int(bool(re.search(dynamic_regex, field, re.IGNORECASE)))

def contributor_match_percentage(contributors, university_website_url, university_acronym, university_name, university_query):
    """
    Calculates the percentage of contributors matching university-related keywords.

    Parameters:
        contributors (list of tuples): Each tuple contains (bio, company, email).
        university_website_url (str): The university's website URL.
        university_acronym (str): The university's acronym.
        university_name (str): The full university name.
        university_query (str): A general university-related keyword.

    Returns:
        list: Percentages (out of total contributors) for each matching criterion.
    """
    total_contributors = len([c for c in contributors if any(c)])  # Count only non-empty contributors

    if total_contributors == 0:
        return [0] * 7  # Return zeroes if no valid contributors

    # Define matching functions
    def match_percentage(field_idx, check_func, attribute):
        matches = sum(check_func(contributor[field_idx], attribute) for contributor in contributors)
        return matches / total_contributors  # Compute percentage

    # Compute percentages for each criterion
    results = [
        match_percentage(2, contains_university_info, university_website_url),  # University URL in email
        match_percentage(0, contains_university_info, university_acronym),  # University acronym in bio
        match_percentage(1, contains_university_info, university_acronym),  # University acronym in company
        match_percentage(0, contains_university_name, university_name),  # University name in bio
        match_percentage(1, contains_university_name, university_name),  # University name in company
        match_percentage(0, contains_university_info, university_query),  # University query in bio
        match_percentage(1, contains_university_info, university_query),  # University query in company
    ]

    return results
    
def build_matrix_with_attributes(env, db_file, acronym):
    """
    Builds a labeled binary matrix for university-affiliated open-source repositories 
    based on attributes from organizations, repositories, and top contributors.
    
    Args:
        env (str): Path to the JSON file with university environment/config metadata.
        db_file (str): Path to the SQLite database file.
        acronym (str): University acronym (used for output file naming).
        path (str): Output path for saving the matrix CSV file.
    
    Returns:
        str: The path to the generated CSV file containing the labeled matrix.
    """

    with open(env) as envfile:
        config = json.loads(envfile.read())
    
    # Assign values to variables using keys from the config
    university_name = config["UNIVERSITY_NAME"]
    university_acronym = config["UNIVERSITY_ACRONYM"]
    university_location = config["UNIVERSITY_LOCATION"]
    university_email_domain = config["UNIVERSITY_EMAIL_DOMAIN"]
    university_website_url = config["UNIVERSITY_WEBSITE_URL"]
    university_query = config["ADDITIONAL_QUERIES"][0]    
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Fetch all repositories and their organization details
    cursor.execute('''
        WITH top_contributors AS (
            SELECT r.id, 
                   TRIM(value) AS contributor,
                   ROW_NUMBER() OVER (PARTITION BY r.id ORDER BY (SELECT NULL)) AS rn
            FROM repositories r,
                 json_each(r.contributors)
            WHERE r.contributors IS NOT NULL
        )
        SELECT r.manual_label, r.html_url, r.id, r.homepage, r.readme, r.description, r.full_name, r.organization, r.owner, 
               o.url, o.email, o.location, o.name, o.description, o.company,
               MAX(CASE WHEN c.rn = 1 THEN c.contributor END) AS c1,
               MAX(CASE WHEN c.rn = 1 THEN ct.bio END) AS c1_bio,
               MAX(CASE WHEN c.rn = 1 THEN ct.company END) AS c1_company,
               MAX(CASE WHEN c.rn = 1 THEN ct.email END) AS c1_email,
               MAX(CASE WHEN c.rn = 2 THEN c.contributor END) AS c2,
               MAX(CASE WHEN c.rn = 2 THEN ct.bio END) AS c2_bio,
               MAX(CASE WHEN c.rn = 2 THEN ct.company END) AS c2_company,
               MAX(CASE WHEN c.rn = 2 THEN ct.email END) AS c2_email,
               MAX(CASE WHEN c.rn = 3 THEN c.contributor END) AS c3,
               MAX(CASE WHEN c.rn = 3 THEN ct.bio END) AS c3_bio,
               MAX(CASE WHEN c.rn = 3 THEN ct.company END) AS c3_company,
               MAX(CASE WHEN c.rn = 3 THEN ct.email END) AS c3_email,
               MAX(CASE WHEN c.rn = 4 THEN c.contributor END) AS c4,
               MAX(CASE WHEN c.rn = 4 THEN ct.bio END) AS c4_bio,
               MAX(CASE WHEN c.rn = 4 THEN ct.company END) AS c4_company,
               MAX(CASE WHEN c.rn = 4 THEN ct.email END) AS c4_email,
               MAX(CASE WHEN c.rn = 5 THEN c.contributor END) AS c5,
               MAX(CASE WHEN c.rn = 5 THEN ct.bio END) AS c5_bio,
               MAX(CASE WHEN c.rn = 5 THEN ct.company END) AS c5_company,
               MAX(CASE WHEN c.rn = 5 THEN ct.email END) AS c5_email
        FROM repositories r
        LEFT JOIN organizations o ON r.owner = o.login
        LEFT JOIN top_contributors c ON r.id = c.id
        LEFT JOIN contributors ct ON c.contributor = ct.login
        GROUP BY r.html_url, r.id, r.homepage, r.readme, r.description, r.full_name, r.organization, r.owner, 
                 o.url, o.email, o.location, o.name, o.description
    ''')
    
    repositories = cursor.fetchall()
    
    # Initialize the list to hold the rows of the matrix
    matrix = []
    
    # Iterate through each repository
    for repo in repositories:
        (
            manual_label, repo_html, repo_id, homepage, readme, description, full_name, is_organization, owner,
            org_url, org_email, org_location, org_name, org_description, org_company,
            c1, c1_bio, c1_company, c1_email,
            c2, c2_bio, c2_company, c2_email,
            c3, c3_bio, c3_company, c3_email,
            c4, c4_bio, c4_company, c4_email,
            c5, c5_bio, c5_company, c5_email,
        ) = repo
    

        # Initialize the row for this repository
        row = [repo_html]
    
        # Organization-related checks (only if the repository owner is an organization)
        if is_organization:
            row.append(contains_university_info(org_url, university_website_url))  # org: university url in organization url
            row.append(contains_university_info(org_email, university_email_domain))  # org: university url in organization email
            row.append(contains_university_info(org_description, university_website_url))  # org: university url in organization description
            row.append(contains_university_info(org_location, university_location))  # org: university location in organization location
            row.append(contains_university_info(org_name, university_acronym))  # org: university acronym in organization name
            row.append(contains_university_info(org_description, university_acronym))  # org: university acronym in organization description
            row.append(contains_university_info(org_company, university_acronym))  # org: university acronym in organization company
            row.append(contains_university_name(org_name, university_name))  # org: university name in organization name
            row.append(contains_university_name(org_description, university_name))  # org: university name in organization description
            row.append(contains_university_name(org_company, university_name))  # org: university name in organization company
            row.append(contains_university_info(org_name, university_query))  # org: university query in organization name
            row.append(contains_university_info(org_description, university_query))  # org: university query in organization description
            row.append(contains_university_info(org_company, university_query))  # org: university query in organization company
        else:
            row.extend([0] * 13)  # Fill with 0s if not an organization
    
        # Repository-related checks
        row.append(contains_university_info(homepage, university_website_url))  # repo: university url in repository homepage
        row.append(contains_university_info(readme, university_website_url))  # repo: university url in repository readme
        row.append(contains_university_info(description, university_website_url))  # repo: university url in repository description
        row.append(contains_university_name(full_name, university_name))  # repo: university name in repository full name
        row.append(contains_university_name(description, university_name))  # repo: university name in repository description
        row.append(contains_university_name(readme, university_name))  # repo: university name in repository readme
        row.append(contains_university_info(full_name, university_acronym))  # repo: university acronym in repository full name
        row.append(contains_university_info(description, university_acronym))  # repo: university acronym in repository description
        row.append(contains_university_info(readme, university_acronym))  # repo: university acronym in repository readme
        row.append(contains_university_info(full_name, university_query))  # repo: university query in repository full name
        row.append(contains_university_info(description, university_query))  # repo: university query in repository description
        row.append(contains_university_info(readme, university_query))  # repo: university query in repository readme
    
        # Contributor-related checks (percentage-based)
        contributors = [
            (c1_bio, c1_company, c1_email),
            (c2_bio, c2_company, c2_email),
            (c3_bio, c3_company, c3_email),
            (c4_bio, c4_company, c4_email),
            (c5_bio, c5_company, c5_email),
        ]
        
        contributor_percentages = contributor_match_percentage(
            contributors, university_website_url, university_acronym, university_name, university_query
        )
    
        # Append contributor match percentages to the row
        row.extend(contributor_percentages)
        row.append(manual_label)
    
        # Add the row to the matrix
        matrix.append(row)
    
    # Convert the matrix to a NumPy array
    matrix_np = np.array(matrix)

    
    # Save the matrix as a CSV file
    directory = f'results/{acronym}'

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    csv_file = f'results/{acronym}/repository_university_matrix_{acronym}.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    
        # Define the header row
        header = [
            'html_url',
            # Organization-related checks
            'org: university url in organization url', 'org: university url in organization email', 'org: university url in organization description',
            'org: university location in organization location', 'org: university acronym in organization name',
            'org: university acronym in organization description', 'org: university acronym in organization company',
            'org: university name in organization name', 'org: university name in organization description', 'org: university name in organization company',
            'org: university query in organization name', 'org: university query in organization description', 'org: university query in organization company',
    
            # Repository-related checks
            'repo: university url in repository homepage', 'repo: university url in repository readme', 'repo: university url in repository description',
            'repo: university name in repository full name', 'repo: university name in repository description', 'repo: university name in repository readme',
            'repo: university acronym in repository full name', 'repo: university acronym in repository description', 'repo: university acronym in repository readme',
            'repo: university query in repository full name', 'repo: university query in repository description', 'repo: university query in repository readme',
    
            # Contributor-related percentage matches
            'cont: university url in contributor email %',
            'cont: university acronym in contributor bio %', 'cont: university acronym in contributor company %',
            'cont: university name in contributor bio %', 'cont: university name in contributor company %',
            'cont: university query in contributor bio %', 'cont: university query in contributor company %',
            # Manual Label
            "manual_label"
        ]
    
        # Write the header and matrix rows
        writer.writerow(header)
        writer.writerows(matrix)
    
    # Print the shape of the matrix for verification
    print("Matrix shape:", matrix_np.shape)
    
    # Close the connection to the database
    conn.close()
    return csv_file

