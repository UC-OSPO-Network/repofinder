#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3

def create_and_populate_database(repo_file, db_file):
    """
    Creates a relational SQLite database to store repository and contributor information.

    Parameters:
        repo_file (str): Path to the JSON file containing repository data.
        db_file (str): Path to save the SQLite database. Default is 'repositories_and_contributors.db'.

    Returns:
        None
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    # file_name = repo_file.split('.')[2].split('/')[3]
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the repositories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repositories (
            full_name TEXT PRIMARY KEY
        )
    """)

    # Load the repository data and insert it into the repositories table
    repo_df = pd.read_json(repo_file) 
    
    # Make some modifications to make it work as a Database
    repo_df['owner'] = repo_df['owner'].apply(lambda x: x['login'])
    repo_df['license'] = repo_df['license'].apply(lambda x: x['key'] if x else None) 
    repo_df = repo_df.drop(columns = ['permissions', 'topics'])
    repo_df = repo_df.drop_duplicates()
    
    repo_df.to_sql('repositories', conn, if_exists='replace', index=False)









