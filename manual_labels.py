#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:11:25 2025

@author: juanitagomez
"""

import pandas as pd
import sqlite3

def update_manual_labels(csv_file, db_file):
    """
    Updates the 'manual_label' column in the 'repositories' table of the database 
    using values from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing 'manual_label' and 'html_url' columns.
    db_file : str
        Path to the SQLite database file.

    Returns
    -------
    None
        Updates the database and prints progress.
    """
    # Load CSV
    df = pd.read_csv(csv_file, usecols=['html_url', 'manual_label'])
    
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
    # Ensure the repositories table has the manual label column
        cursor.execute("ALTER TABLE repositories ADD COLUMN manual_label TEXT;")  # Adjust the column type as needed
    except:
        pass
    for index, row in df.iterrows():
        repo_link = row['html_url']
        manual_label = row['manual_label']
        
        try:
            cursor.execute(
                "UPDATE repositories SET manual_label = ? WHERE html_url = ?;",
                (manual_label, repo_link)
            )
        except Exception as e:
            print(f"Error updating {repo_link}: {e}")
        
        if index % 10 == 0 or index == len(df) - 1:  # Print progress every 10 updates
            print(f"Updated {index + 1}/{len(df)} repositories")

    conn.commit()
    conn.close()


# Update manual labels when labeling happens
acronym = 'UCSC'
csv_file_labels = f'{acronym}_new_labels.csv'
db_file = f'Data/db/repository_data_{acronym}_database.db'

update_manual_labels(csv_file_labels, db_file)
