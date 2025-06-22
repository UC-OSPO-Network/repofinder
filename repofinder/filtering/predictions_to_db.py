#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:11:25 2025

@author: juanitagomez
"""

import pandas as pd
import sqlite3

def update_predictions(csv_file, db_file):
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
    # Load CSV
    all_columns = pd.read_csv(csv_file, nrows=0).columns
    
    # Select columns that start with 'Predictions'
    prediction_cols = [col for col in all_columns if col.startswith('Predictions')]
    
    # Always include 'html_url'
    usecols = ['html_url'] + prediction_cols
    df = pd.read_csv(csv_file, usecols=usecols)
    
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
    # Ensure the repositories table has the organization column
        cursor.execute("ALTER TABLE repositories ADD COLUMN prediction REAL;")  # Adjust the column type as needed
    except:
        pass
    for index, row in df.iterrows():
        repo_link = row['html_url']
        prediction = row[prediction_cols[0]]
        
        try:
            cursor.execute(
                "UPDATE repositories SET prediction = ? WHERE html_url = ?;",
                (prediction, repo_link)
            )
        except Exception as e:
            print(f"Error updating {repo_link}: {e}")
        
        if index % 10 == 0 or index == len(df) - 1:  # Print progress every 10 updates
            print(f"Updated {index + 1}/{len(df)} repositories")

    conn.commit()
    conn.close()


