#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan 27 13:31:21 2025

@author: juanitagomez
"""
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
    

def get_embedding(text, client,  model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def build_matrix_with_embeddings(env, db_file, acronym, client):
    """
    Builds an embedding matrix from combined repository metadata and saves it as a CSV file.

    Args:
        env (str): Path to the JSON config file with university metadata.
        db_file (str): Path to the SQLite database file.
        acronym (str): University acronym used for naming the output file.
        client (object): An OpenAI client instance used to generate embeddings.
    
    Returns:
        str: Path to the generated CSV file containing the embedding matrix.
    """
    
    df = get_combined_data(env, db_file, acronym)

 
    # Generate embeddings
    df['embedding'] = df['combined'].apply(lambda x: get_embedding(x, client))

    # Create embedding matrix
    embedding_matrix = pd.DataFrame(df['embedding'].to_list(), index=df['html_url'])
    embedding_matrix.reset_index(inplace=True)
    embedding_matrix['manual_label'] = df['manual_label']


    embedding_matrix.to_csv(f'results/{acronym}/repository_embeddings_{acronym}.csv', index=False)
    return f'results/{acronym}/repository_embeddings_{acronym}.csv'


