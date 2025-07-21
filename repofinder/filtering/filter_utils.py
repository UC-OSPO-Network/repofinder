#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd

def get_combined_data(env, db_file, acronym, truncate=10000, subset=None):
    """
    Extracts and preprocesses repository data from a SQLite database to create
    a combined textual representation for each repository.

    
    Args:
        env (str): Path to the JSON config file containing university metadata (currently unused).
        db_file (str): Path to the SQLite database file.
        acronym (str): University acronym (currently unused in this function).
        truncate (int, optional): Maximum number of characters per text field (default is 15,000).
        subset (str, optional): Path to a CSV file with a subset of repositories to include.
    
    Returns:
        pd.DataFrame: A DataFrame with repository metadata and a `combined` text column.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
    # Ensure the repositories table has the manual label column
        cursor.execute("ALTER TABLE repositories ADD COLUMN manual_label TEXT;")  # Adjust the column type as needed
    except:
        pass

    query = '''
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
                MAX(CASE WHEN c.rn = 2 THEN ct.email END) AS c2_email
        FROM repositories r
        LEFT JOIN organizations o ON r.owner = o.login
        LEFT JOIN top_contributors c ON r.id = c.id
        LEFT JOIN contributors ct ON c.contributor = ct.login
        GROUP BY r.manual_label, r.html_url, r.id, r.homepage, r.readme, r.description, r.full_name, r.organization, r.owner, 
                  o.url, o.email, o.location, o.name, o.description, o.company
    '''

    # Execute query
    cursor.execute(query)
    repositories = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Close connection
    conn.close()

    # Create DataFrame
    df = pd.DataFrame(repositories, columns=columns)
    #TODO uncomment
    for col in df.columns.tolist():
        df[col] = df[col].astype(str).apply(lambda x: truncate_text(x, truncate))

    # Combine text columns into a single string per repository
    
    df['combined'] = df.drop(columns=['html_url', 'manual_label']).astype(str).agg(' '.join, axis=1)

    # Compute the average length
    average_length = df["combined"].apply(len).mean()

    print(f"Average length of 'combined' column: {average_length:.2f} characters")
    
    
    #df['combined'] = df.astype(str).apply(lambda row: ' '.join(map(truncate_text, row)), axis=1)
    
    if subset: 
        subset_labels_df = pd.read_csv(subset)

        # Ensure 'html_url' column exists in both DataFrames
        if 'html_url' not in df.columns or 'html_url' not in subset_labels_df.columns:
            raise ValueError("Both DataFrames must contain a 'html_url' column.")
        
        # Filter df to only include html_urls present in ucsc_new_labels
        filtered_df = df[df['html_url'].isin(subset_labels_df['html_url'])]
        
        df=filtered_df
    
    return df

def truncate_text(text, max_length):
    return text[:max_length] if isinstance(text, str) else ""
    

def get_type_combined_data(env, db_file, acronym, truncate=20000, subset=None):

    def truncate_text(text, max_length):
        return text[:max_length] if isinstance(text, str) else ""

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE repositories ADD COLUMN manual_label TEXT;")
    except:
        pass

    query = '''
        WITH contributor_counts AS (
            SELECT repository_name, COUNT(DISTINCT contributor_login) AS number_of_contributors
            FROM contributions
            GROUP BY repository_name
        )
        SELECT
            r.html_url,
            r.full_name,
            r.description,
            r.readme,
            COALESCE(cc.number_of_contributors, 0) AS number_of_contributors,
            COALESCE(r.forks, 0) AS number_of_forks,
            COALESCE(r.stargazers_count, 0) AS number_of_stars,
            COALESCE(r.ai_prediction, 0) AS ai_prediction
        FROM repositories r
        LEFT JOIN contributor_counts cc ON r.full_name = cc.repository_name
    '''

    cursor.execute(query)
    repos = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    df = pd.DataFrame(repos, columns=columns)

    for col in ['full_name', 'description', 'readme']:
        df[col] = df[col].astype(str).apply(lambda x: truncate_text(x, truncate))

    df['combined'] = (
        df['full_name'] + " " +
        df['description'] + " " +
        df['readme'] + " " +
        "Contributors: " + df['number_of_contributors'].astype(str) + " " +
        "Forks: " + df['number_of_forks'].astype(str) + " " +
        "Stars: " + df['number_of_stars'].astype(str)
    )

    if subset:
        subset_df = pd.read_csv(subset)
        if 'html_url' not in df.columns or 'html_url' not in subset_df.columns:
            raise ValueError("Both DataFrames must contain a 'html_url' column.")
        df = df[df['html_url'].isin(subset_df['html_url'])]

    return df



def update_predictions_in_db(csv_file, db_file, column):
    """
    Updates the predictions column in the 'repositories' table of the database 
    using values from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing 'Manual Label' and 'Repository Link' columns.
    db_file : str
        Path to the SQLite database file.

    Returns
    -------
    None
        Updates the database and prints progress.
    """
    
    df = pd.read_csv(csv_file)
    if 'gpt_belonging' in df.columns:
        df = df.rename(columns={'gpt_belonging': 'ai_prediction'})
        column = 'ai_prediction'
    
    
    df = df[['html_url', column]]
    
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        # Determine the appropriate column type
        if column == 'ai_prediction':
            col_type = 'REAL'
        elif column == 'gpt_category':
            col_type = 'TEXT'
        else:
            col_type = 'TEXT'  # default

        # Ensure the repositories table has the target column
        cursor.execute(f"ALTER TABLE repositories ADD COLUMN {column} {col_type};")
    except sqlite3.OperationalError:
        pass  # Column likely already exists
        
    for index, row in df.iterrows():
        repo_link = row['html_url']
        prediction = row[column]
    
        try:
            cursor.execute(
                f"UPDATE repositories SET {column} = ? WHERE html_url = ?;",
                (prediction, repo_link)
            )
        except Exception as e:
            print(f"Error updating row {index}: {e}")
        
        if index % 10 == 0 or index == len(df) - 1:  # Print progress every 10 updates
            print(f"Updated {index + 1}/{len(df)} repositories")

    conn.commit()
    conn.close()
