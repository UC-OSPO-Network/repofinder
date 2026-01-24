#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd

def get_combined_data(env, db_file, acronym, truncate=20000, subset=None, truncation_type="start", start_length=15000, end_length=5000):
    """
    Extract and preprocess repository data from a SQLite database.

    This function queries repository data from a SQLite database, joins with
    organization and contributor data, applies text truncation to all columns,
    and creates a combined textual representation for each repository.

    Parameters
    ----------
    env : str
        Path to the JSON config file containing university metadata
        (currently unused, kept for API compatibility).
    db_file : str
        Path to the SQLite database file containing repository data.
    acronym : str
        University acronym (currently unused, kept for API compatibility).
    truncate : int, optional
        Maximum number of characters per text field. Truncation only occurs
        if text length exceeds this value (default: 20000).
    subset : str, optional
        Path to a CSV file with html_url column to filter repositories
        (e.g., test_set file). If None, processes all repositories
        (default: None).
    truncation_type : str, optional
        Type of truncation: "start" (keep first N characters) or
        "start_end" (keep first N and last M characters) (default: "start").
    start_length : int, optional
        Number of characters from start for "start_end" truncation type
        (default: 15000).
    end_length : int, optional
        Number of characters from end for "start_end" truncation type
        (default: 5000).

    Returns
    -------
    pd.DataFrame
        DataFrame containing repository metadata with columns:
        - html_url: Repository URL
        - manual_label: Manual classification label
        - readme, description, full_name: Repository metadata
        - organization, owner: Repository ownership info
        - size: Repository size
        - org_description, url, email, location: Organization metadata
        - c1, c1_bio, c1_company, c1_email: Top contributor 1 info
        - c2, c2_bio, c2_company, c2_email: Top contributor 2 info
        - combined: Concatenated text representation of all fields

    Notes
    -----
    - Truncation is applied to all text columns before creating the combined field.
    - The `combined` field excludes 'html_url' and 'manual_label' columns.
    - If a subset file is provided, only repositories matching html_urls in the
      subset are returned.
    - Automatically adds 'manual_label' column to the repositories table if
      it doesn't exist.
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
        SELECT r.manual_label, r.html_url, r.id, r.homepage, r.readme, r.description, r.full_name, r.organization, r.owner, r.size,
                r.archived, r.fork, r.is_template,
                o.url, o.email, o.location, o.name, o.description AS org_description, o.company,
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
        WHERE (r.fork = 0 OR r.fork = FALSE OR r.fork IS NULL)
          AND (r.size > 0 OR r.size IS NULL)
          AND (r.archived = 0 OR r.archived = FALSE OR r.archived IS NULL)
          AND (r.is_template = 0 OR r.is_template = FALSE OR r.is_template IS NULL)
        GROUP BY r.manual_label, r.html_url, r.id, r.homepage, r.readme, r.description, r.full_name, r.organization, r.owner, r.size,
                  r.archived, r.fork, r.is_template,
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
        df[col] = df[col].astype(str).apply(
            lambda x: truncate_text(x, truncate, truncation_type=truncation_type, 
                                   start_length=start_length, end_length=end_length)
        )

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

def truncate_text(text, max_length, truncation_type="start", start_length=15000, end_length=5000):
    """
    Truncate text based on the specified truncation type.

    This function truncates text using different strategies: "start" keeps
    the first N characters, while "start_end" keeps the first N and last M
    characters separated by "...". For "start_end", truncation only occurs
    if the text length exceeds max_length.

    Parameters
    ----------
    text : str
        The text to truncate. Non-string inputs return empty string.
    max_length : int
        Maximum length threshold for truncation. For "start" type, this is
        the number of characters to keep. For "start_end" type, truncation
        only occurs if text length exceeds this value.
    truncation_type : str, optional
        Type of truncation strategy (default: "start"):
        - "start": Keep first max_length characters
        - "start_end": Keep first start_length and last end_length characters,
          separated by "..." (only if text length > max_length)
    start_length : int, optional
        Number of characters to keep from the start for "start_end" type
        (default: 15000).
    end_length : int, optional
        Number of characters to keep from the end for "start_end" type
        (default: 5000).

    Returns
    -------
    str
        Truncated text. For "start_end" type with truncation, returns
        "{first_part}...{last_part}". Returns empty string for non-string input.
    """
    if not isinstance(text, str):
        return ""
    
    if truncation_type == "start":
        return text[:max_length]
    elif truncation_type == "start_end":
        # Check if text is larger than max_length (20000) before truncating
        # For start_end truncation, max_length should equal start_length + end_length
        if len(text) <= max_length:
            # Text is 20000 characters or less, return as-is without truncation
            return text
        
        # Text is larger than 20000, proceed with truncation
        # Take start_length from beginning and end_length from end
        start_part = text[:start_length]
        end_part = text[-end_length:] if end_length > 0 else ""
        
        # Combine with separator
        return f"{start_part}...{end_part}"
    else:
        # Default to start truncation for unknown types
        return text[:max_length]
    

def get_type_combined_data(env, db_file, acronym, truncate=20000, subset=None, truncation_type="start", start_length=15000, end_length=5000):
    """
    Extract and preprocess repository data for type classification.

    This function queries repository data from a SQLite database for the
    purpose of project type classification (e.g., DEV, EDU, DOCS, WEB, DATA, OTHER).
    It includes repository metadata, contributor counts, and applies text
    truncation to specific fields.

    Parameters
    ----------
    env : str
        Path to the JSON config file (currently unused, kept for API compatibility).
    db_file : str
        Path to the SQLite database file containing repository data.
    acronym : str
        University acronym (currently unused, kept for API compatibility).
    truncate : int, optional
        Maximum number of characters per text field. Truncation only occurs
        if text length exceeds this value (default: 20000).
    subset : str, optional
        Path to a CSV file with html_url column to filter repositories
        (e.g., test_set file). If None, processes all repositories
        (default: None).
    truncation_type : str, optional
        Type of truncation: "start" (keep first N characters) or
        "start_end" (keep first N and last M characters) (default: "start").
    start_length : int, optional
        Number of characters from start for "start_end" truncation type
        (default: 15000).
    end_length : int, optional
        Number of characters from end for "start_end" truncation type
        (default: 5000).

    Returns
    -------
    pd.DataFrame
        DataFrame containing repository data with columns:
        - html_url: Repository URL
        - full_name, description, readme: Repository metadata (truncated)
        - number_of_contributors: Count of contributors
        - number_of_forks: Fork count
        - number_of_stars: Star count
        - ai_prediction: AI prediction value
        - combined: Concatenated text representation

    Notes
    -----
    - Truncation is applied to 'full_name', 'description', and 'readme' columns.
    - The `combined` field includes full_name, description, readme, and metadata.
    - If a subset file is provided, only repositories matching html_urls in the
      subset are returned.
    - Automatically adds 'manual_label' column to the repositories table if
      it doesn't exist.
    """

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
            COALESCE(r.affiliation_prediction_gpt_5_mini, 0) AS ai_prediction
        FROM repositories r
        LEFT JOIN contributor_counts cc ON r.full_name = cc.repository_name
    '''

    cursor.execute(query)
    repos = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    df = pd.DataFrame(repos, columns=columns)

    for col in ['full_name', 'description', 'readme']:
        df[col] = df[col].astype(str).apply(
            lambda x: truncate_text(x, truncate, truncation_type=truncation_type,
                                   start_length=start_length, end_length=end_length)
        )

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
    Update predictions column in the repositories table from a CSV file.

    This function reads prediction values from a CSV file and updates the
    corresponding column in the SQLite database. It maps CSV columns to
    database columns and handles appropriate data types.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing prediction data. Must contain
        'html_url' column and either 'gpt_belonging' (affiliation) or
        'gpt_category' (type) column.
    db_file : str
        Path to the SQLite database file containing the repositories table.
    column : str
        Name of the column to update in the repositories table. Valid options:
        - 'affiliation_prediction_gpt-4o': For affiliation predictions (REAL type)
        - 'affiliation_prediction_gpt-5-mini': For affiliation predictions (REAL type)
        - 'type_prediction_gpt-4o': For type predictions (TEXT type)
        - 'type_prediction_gpt-5-mini': For type predictions (TEXT type)

    Returns
    -------
    None
        Updates the database in-place and prints progress every 10 updates.

    Raises
    ------
    ValueError
        If CSV file doesn't contain the expected prediction column, or if
        column parameter doesn't match the CSV data type.

    Notes
    -----
    - CSV files contain 'gpt_belonging' for affiliation predictions and
      'gpt_category' for type predictions.
    - Affiliation predictions are stored as REAL (float) type in the database.
    - Type predictions are stored as TEXT type in the database.
    - The target column is automatically created if it doesn't exist.
    - Progress is printed every 10 updates and on the final update.
    - Updates are committed to the database after all rows are processed.
    """
    
    df = pd.read_csv(csv_file)
    
    # Determine prediction type from column parameter
    if column.startswith('affiliation_prediction_'):
        # Expect gpt_belonging in CSV
        if 'gpt_belonging' not in df.columns:
            raise ValueError(f"CSV file must contain 'gpt_belonging' column for affiliation predictions. Found columns: {list(df.columns)}")
        csv_column = 'gpt_belonging'
        col_type = 'REAL'
    elif column.startswith('type_prediction_'):
        # Expect gpt_category in CSV
        if 'gpt_category' not in df.columns:
            raise ValueError(f"CSV file must contain 'gpt_category' column for type predictions. Found columns: {list(df.columns)}")
        csv_column = 'gpt_category'
        col_type = 'TEXT'
    else:
        raise ValueError(f"Invalid column name '{column}'. Must start with 'affiliation_prediction_' or 'type_prediction_'")
    
    # Select required columns
    if 'html_url' not in df.columns:
        raise ValueError(f"CSV file must contain 'html_url' column. Found columns: {list(df.columns)}")
    
    df = df[['html_url', csv_column]]
    
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        # Ensure the repositories table has the target column
        cursor.execute(f"ALTER TABLE repositories ADD COLUMN {column} {col_type};")
    except sqlite3.OperationalError:
        pass  # Column likely already exists
        
    for index, row in df.iterrows():
        repo_link = row['html_url']
        prediction = row[csv_column]
    
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
