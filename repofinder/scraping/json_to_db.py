#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
import json

def _convert_to_sqlite_type(value):
    """
    Convert a Python value to a SQLite-compatible type.
    
    Parameters:
    -----------
    value : Any
        The value to convert.
    
    Returns:
    --------
    Any
        A SQLite-compatible value (str, int, float, None, or JSON string).
    """
    if pd.isna(value) or value is None:
        return None
    
    # Handle pandas numeric types
    if pd.api.types.is_integer(value):
        return int(value)
    if pd.api.types.is_float(value):
        return float(value)
    if pd.api.types.is_bool(value):
        return int(bool(value))
    
    # Handle Python native types
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return value
    
    # Convert complex types (dict, list, etc.) to JSON strings
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return None
    
    # For any other type, try to convert to string
    try:
        return str(value)
    except:
        return None

def create_and_populate_database(repo_file, db_file, search_method='repository_search'):
    """
    Creates or populates a relational SQLite database to store repository information.
    Handles duplicates by updating search_method to 'both' if repository already exists.

    Parameters:
        repo_file (str): Path to the JSON file containing repository data.
        db_file (str): Path to save the SQLite database.
        search_method (str): The search method used to find these repositories.
                            Default is 'repository_search'. Other common values:
                            'organization_search', or custom values.

    Returns:
        None
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Load the repository data first to see what columns we have
    repo_df = pd.read_json(repo_file) 
    
    # Make some modifications to make it work as a Database
    # Handle nested objects: extract owner.login
    if 'owner' in repo_df.columns and repo_df['owner'].notna().any():
        repo_df['owner'] = repo_df['owner'].apply(lambda x: x['login'] if isinstance(x, dict) and 'login' in x else None)
    
    # Handle nested objects: extract license.key
    if 'license' in repo_df.columns and repo_df['license'].notna().any():
        repo_df['license'] = repo_df['license'].apply(lambda x: x['key'] if isinstance(x, dict) and x and 'key' in x else None)
    
    # IMPORTANT: Convert ALL nested structures (dicts, lists) to JSON strings early
    # Organization repos endpoint returns more nested data than search endpoint
    # This ensures both data sources are handled consistently
    for col in repo_df.columns:
        if repo_df[col].dtype == 'object':
            # Sample values to check if column contains nested structures
            sample = repo_df[col].dropna().head(100)
            if not sample.empty:
                has_nested = any(isinstance(v, (dict, list)) for v in sample if v is not None)
                if has_nested:
                    # Convert all nested structures to JSON strings
                    repo_df[col] = repo_df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
    
    # Drop columns that are lists/dicts that we don't want to store
    # Note: We could keep them as JSON strings if needed, but dropping for now
    columns_to_drop = ['permissions', 'topics']
    for col in columns_to_drop:
        if col in repo_df.columns:
            repo_df = repo_df.drop(columns=[col])
    
    repo_df = repo_df.drop_duplicates()
    
    # Add search_method column to dataframe
    repo_df['search_method'] = search_method
    
    # Remove duplicates within the input dataframe itself (keep first occurrence)
    repo_df = repo_df.drop_duplicates(subset=['full_name'], keep='first')
    
    # Create the repositories table with all columns from the dataframe
    # First, ensure the table exists with at least full_name as PRIMARY KEY
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repositories (
            full_name TEXT PRIMARY KEY
        )
    """)

    # Get existing table columns
    cursor.execute("PRAGMA table_info(repositories)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    # Add all columns from the dataframe that don't exist in the table
    for col in repo_df.columns:
        if col not in existing_columns:
            # Determine SQLite type based on pandas dtype
            if repo_df[col].dtype in ['int64', 'int32', 'int']:
                sql_type = 'INTEGER'
            elif repo_df[col].dtype in ['float64', 'float32', 'float']:
                sql_type = 'REAL'
            elif repo_df[col].dtype == 'bool':
                sql_type = 'INTEGER'  # SQLite doesn't have boolean, use INTEGER
            else:
                sql_type = 'TEXT'
            
            try:
                cursor.execute(f"ALTER TABLE repositories ADD COLUMN {col} {sql_type};")
                existing_columns.add(col)
            except sqlite3.OperationalError:
                pass  # Column might have been added by another process
    
    # Check if table has any existing data
    cursor.execute("SELECT COUNT(*) FROM repositories")
    table_is_empty = cursor.fetchone()[0] == 0
    
    # Get existing repositories to check for duplicates (only if table is not empty)
    existing_repos = pd.DataFrame()  # Initialize empty dataframe
    if not table_is_empty:
        try:
            existing_repos = pd.read_sql_query("SELECT full_name, search_method FROM repositories", conn)
        except:
            # If search_method doesn't exist yet, just get full_name
            try:
                existing_repos = pd.read_sql_query("SELECT full_name FROM repositories", conn)
                existing_repos['search_method'] = None
            except:
                existing_repos = pd.DataFrame()
    
    if not table_is_empty and not existing_repos.empty:
        # Find duplicates and update their search_method
        existing_full_names = set(existing_repos['full_name'].tolist())
        duplicates = repo_df[repo_df['full_name'].isin(existing_full_names)]
        
        for _, row in duplicates.iterrows():
            full_name = row['full_name']
            if 'search_method' in existing_repos.columns:
                existing_method = existing_repos[existing_repos['full_name'] == full_name]['search_method'].iloc[0]
                
                # If the existing method is different from the new method, update to 'both'
                if existing_method and existing_method != search_method and existing_method != 'both':
                    # Update to 'both' since found via multiple methods
                    cursor.execute(
                        "UPDATE repositories SET search_method = ? WHERE full_name = ?",
                        ('both', full_name)
                    )
            # If already 'both' or same method, no need to update
        
        # Get new repositories (not in existing)
        new_repos = repo_df[~repo_df['full_name'].isin(existing_full_names)]
    else:
        # No existing repos, all are new
        new_repos = repo_df
    
    # Insert new repositories
    if not new_repos.empty:
        # Get the actual table schema to ensure column alignment
        cursor.execute("PRAGMA table_info(repositories)")
        table_info = cursor.fetchall()
        table_columns = {row[1] for row in table_info}  # Use set for faster lookup
        
        # Only keep columns that exist in both the dataframe and the table
        columns_to_insert = [col for col in new_repos.columns if col in table_columns]
        
        if columns_to_insert:
            # Select only the columns that exist in the table
            new_repos_aligned = new_repos[columns_to_insert].copy()
            
            # Convert boolean columns to integers for SQLite
            # Also convert any complex types to JSON strings
            # Do this for ALL columns to ensure SQLite compatibility
            for col in new_repos_aligned.columns:
                if new_repos_aligned[col].dtype == 'bool':
                    new_repos_aligned[col] = new_repos_aligned[col].astype(int)
                # Check for object columns that might contain lists/dicts
                # Apply conversion to ensure all values are SQLite-compatible
                elif new_repos_aligned[col].dtype == 'object':
                    # Convert the entire column, handling any complex types
                    new_repos_aligned[col] = new_repos_aligned[col].apply(_convert_to_sqlite_type)
            
            # If table is empty, use to_sql for faster bulk insert
            if table_is_empty:
                try:
                    # Double-check: convert any remaining problematic values
                    # This is a safety net in case some values slipped through
                    for col in new_repos_aligned.columns:
                        if new_repos_aligned[col].dtype == 'object':
                            # Sample a few values to check for complex types
                            sample = new_repos_aligned[col].dropna().head(10)
                            if not sample.empty:
                                has_complex = any(isinstance(v, (list, dict)) for v in sample if v is not None)
                                if has_complex:
                                    new_repos_aligned[col] = new_repos_aligned[col].apply(_convert_to_sqlite_type)
                    
                    new_repos_aligned.to_sql('repositories', conn, if_exists='append', index=False)
                    print(f"Inserted {len(new_repos_aligned)} repositories into empty table.")
                except Exception as e:
                    print(f"Error using to_sql: {e}. Falling back to row-by-row insertion.")
                    # Fall through to row-by-row insertion
                    table_is_empty = False
            
            # Use row-by-row insertion if table has data (to handle duplicates) or if to_sql failed
            if not table_is_empty and not new_repos_aligned.empty:
                inserted_count = 0
                for idx, row in new_repos_aligned.iterrows():
                    try:
                        # Build INSERT OR IGNORE statement
                        placeholders = ', '.join(['?' for _ in columns_to_insert])
                        columns_str = ', '.join(columns_to_insert)
                        # Convert each value to SQLite-compatible type
                        values = []
                        for col in columns_to_insert:
                            try:
                                converted_val = _convert_to_sqlite_type(row[col])
                                values.append(converted_val)
                            except Exception as col_err:
                                print(f"Warning: Error converting column '{col}' for {row.get('full_name', 'unknown')}: {col_err}. Using None.")
                                values.append(None)
                        
                        cursor.execute(
                            f"INSERT OR IGNORE INTO repositories ({columns_str}) VALUES ({placeholders})",
                            values
                        )
                        if cursor.rowcount > 0:
                            inserted_count += 1
                    except sqlite3.IntegrityError as e:
                        # If still fails, just skip this row (it's a duplicate)
                        continue
                    except Exception as e:
                        print(f"Error inserting row for {row.get('full_name', 'unknown')}: {e}")
                        # Try to identify which column caused the issue
                        try:
                            for i, col in enumerate(columns_to_insert):
                                val = row[col]
                                if isinstance(val, (list, dict)):
                                    print(f"  Column '{col}' (index {i}) contains {type(val).__name__}: {str(val)[:100]}")
                        except:
                            pass
                        continue
                
                if inserted_count > 0:
                    print(f"Inserted {inserted_count} new repositories (skipped duplicates).")
        else:
            # If no matching columns, something is wrong - log a warning
            print(f"Warning: No matching columns found between dataframe and table. "
                  f"Dataframe columns: {list(new_repos.columns)}, Table columns: {list(table_columns)}")
    
    conn.commit()
    conn.close()





def populate_organizations(org_file, db_file):
    """
    Populates a SQLite database table with organization information scraped from GitHub.

    Parameters:
        org_file (str): Path to the JSON file containing organization data.
        db_file (str): Path to save the SQLite database.

    Returns:
        None
        Creates or replaces the 'orgs_scraped' table with organization data.
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the organizations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orgs_scraped (
            login TEXT PRIMARY KEY
        )
    """)

    # Load the organization data
    org_df = pd.read_json(org_file) 
    
    # Handle nested structures (dicts, lists) by converting to JSON strings
    for col in org_df.columns:
        if org_df[col].dtype == 'object':
            # Sample values to check if column contains nested structures
            sample = org_df[col].dropna().head(100)
            if not sample.empty:
                has_nested = any(isinstance(v, (dict, list)) for v in sample if v is not None)
                if has_nested:
                    # Convert all nested structures to JSON strings
                    org_df[col] = org_df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
    
    # Make some modifications to make it work as a Database
    org_df = org_df.drop_duplicates()
    
    # Get existing table columns
    cursor.execute("PRAGMA table_info(orgs_scraped)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    # Add all columns from the dataframe that don't exist in the table
    for col in org_df.columns:
        if col not in existing_columns:
            # Determine SQLite type based on pandas dtype
            if org_df[col].dtype in ['int64', 'int32', 'int']:
                sql_type = 'INTEGER'
            elif org_df[col].dtype in ['float64', 'float32', 'float']:
                sql_type = 'REAL'
            elif org_df[col].dtype == 'bool':
                sql_type = 'INTEGER'  # SQLite doesn't have boolean, use INTEGER
            else:
                sql_type = 'TEXT'
            
            try:
                cursor.execute(f"ALTER TABLE orgs_scraped ADD COLUMN {col} {sql_type};")
                existing_columns.add(col)
            except sqlite3.OperationalError:
                pass  # Column might have been added by another process
    
    # Convert values to SQLite-compatible types
    for col in org_df.columns:
        if org_df[col].dtype == 'bool':
            org_df[col] = org_df[col].astype(int)
        elif org_df[col].dtype == 'object':
            org_df[col] = org_df[col].apply(_convert_to_sqlite_type)
    
    try:
        org_df.to_sql('orgs_scraped', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Error inserting organizations into database: {e}")
        raise
    
    conn.commit()
    conn.close()


def populate_users(user_file, db_file):
    """
    Populates a SQLite database table with user information scraped from GitHub.

    Parameters
    ----------
    user_file : str
        Path to the JSON file containing user data.
    db_file : str
        Path to save the SQLite database.

    Returns
    -------
    None
        Creates or replaces the 'users_scraped' table with user data.
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users_scraped (
            login TEXT PRIMARY KEY
        )
    """)

    # Load the user data
    user_df = pd.read_json(user_file) 
    
    # Handle nested structures (dicts, lists) by converting to JSON strings
    for col in user_df.columns:
        if user_df[col].dtype == 'object':
            # Sample values to check if column contains nested structures
            sample = user_df[col].dropna().head(100)
            if not sample.empty:
                has_nested = any(isinstance(v, (dict, list)) for v in sample if v is not None)
                if has_nested:
                    # Convert all nested structures to JSON strings
                    user_df[col] = user_df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
    
    # Make some modifications to make it work as a Database
    user_df = user_df.drop_duplicates()
    
    # Get existing table columns
    cursor.execute("PRAGMA table_info(users_scraped)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    # Add all columns from the dataframe that don't exist in the table
    for col in user_df.columns:
        if col not in existing_columns:
            # Determine SQLite type based on pandas dtype
            if user_df[col].dtype in ['int64', 'int32', 'int']:
                sql_type = 'INTEGER'
            elif user_df[col].dtype in ['float64', 'float32', 'float']:
                sql_type = 'REAL'
            elif user_df[col].dtype == 'bool':
                sql_type = 'INTEGER'  # SQLite doesn't have boolean, use INTEGER
            else:
                sql_type = 'TEXT'
            
            try:
                cursor.execute(f"ALTER TABLE users_scraped ADD COLUMN {col} {sql_type};")
                existing_columns.add(col)
            except sqlite3.OperationalError:
                pass  # Column might have been added by another process
    
    # Convert values to SQLite-compatible types
    for col in user_df.columns:
        if user_df[col].dtype == 'bool':
            user_df[col] = user_df[col].astype(int)
        elif user_df[col].dtype == 'object':
            user_df[col] = user_df[col].apply(_convert_to_sqlite_type)
    
    try:
        user_df.to_sql('users_scraped', conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Error inserting users into database: {e}")
        raise
    
    conn.commit()
    conn.close()






