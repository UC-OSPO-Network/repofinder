#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import time
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from repofinder.filtering.filter_utils import get_type_combined_data


def _process_type_row(row_data):
    """
    Process a single repository row for type classification with retry logic.

    This function handles API calls to OpenAI GPT for type classification of
    a single repository, including rate limiting, retry logic with exponential
    backoff, and response parsing.

    Parameters
    ----------
    row_data : tuple
        Tuple containing:
            - row (dict): Repository data dictionary
            - categories_definition (str): Definition of repository categories
            - client: OpenAI client instance
            - model (str): GPT model name (e.g., "gpt-4o", "gpt-5-mini")
            - rate_limiter (Semaphore): Thread-safe rate limiter
            - system_message (str): System message for GPT
            - row_idx (int): Original DataFrame index

    Returns
    -------
    tuple
        Three-element tuple containing:
            - row_idx (int): Original DataFrame index
            - category (str): Category prediction or "error" string
            - explanation (str): Explanation text or error message

    Notes
    -----
    - Implements exponential backoff retry for transient errors (rate limits,
      timeouts, server errors)
    - Non-retryable errors return immediately with "error" as category
    - Maximum 3 retry attempts with delays: 1s, 2s, 4s
    """
    row, categories_definition, client, model, rate_limiter, system_message, row_idx = row_data
        
    repo_info = f"""
    Repository Information:
    HTML URL: {row.get('html_url', '')}
    Full Name: {row.get('full_name', '')}
    Description: {row.get('description', '')}
    Readme: {row.get('readme', '')}
    Stars: {row.get('number_of_stars', '')}
    Forks: {row.get('number_of_forks', '')}
    Contributors: {row.get('number_of_contributors', '')}
    AI Prediction: {row.get('ai_prediction', '')}
    """.strip()

    prompt = f"""
    {categories_definition}
    
    Here is a repository:
    
    {repo_info}
    
    Your task:
    Return only the predicted category (one of: DEV, EDU, DOCS, WEB, DATA, OTHER), and a short explanation.
    
    Format your response exactly as:
    
    Category: <one of the 7 categories>  
    Explanation: <brief explanation>
    """.strip()

    # Rate limiting
    with rate_limiter:
        # Retry logic for transient errors
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                if model == "gpt-5" or model == "gpt-5-mini":
                    kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        "timeout": 60.0  # 60 second timeout
                    }
                else:
                    kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,  # deterministic
                        "timeout": 60.0  # 60 second timeout
                    }
                
                # Set seed only if using gpt-4o or gpt-4-turbo
                if model.startswith("gpt-4o"):
                    kwargs["seed"] = 42

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content.strip()

                # Parse response
                category = "error"
                explanation = ""
                for line in content.splitlines():
                    if line.lower().startswith("category:"):
                        category = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("explanation:"):
                        explanation = line.split(":", 1)[1].strip()

                # Success - break out of retry loop
                break
                
            except Exception as e:
                html_url = row.get('html_url', 'unknown')
                error_msg = str(e).lower()
                
                # Check if error is retryable (rate limits, timeouts, server errors)
                is_retryable = any(keyword in error_msg for keyword in [
                    'rate limit', 'timeout', '503', '502', '500', '429', 'server error'
                ])
                
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retryable error for repo {html_url} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    if attempt == max_retries - 1:
                        print(f"Error querying OpenAI for repo {html_url} after {max_retries} attempts: {e}")
                    else:
                        print(f"Non-retryable error for repo {html_url}: {e}")
                    category = "error"
                    explanation = str(e)
                    break
    
    return row_idx, category, explanation


def compute_ai_type_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False,
                                max_workers=30, rate_limit=10, checkpoint_interval=100, resume=True,
                                truncate=20000, truncation_type="start", start_length=15000, end_length=5000,
                                affiliated_only=False, affiliation_threshold=None):
    """
    Classify repositories into categories using OpenAI GPT with concurrent processing.

    This function processes repositories from a database and uses OpenAI GPT
    models to predict their type category (DEV, EDU, DOCS, WEB, DATA, OTHER).
    It supports concurrent processing, checkpointing, and resume capability.

    Parameters
    ----------
    acronym : str
        University or project acronym (used for output path).
    config_file : str
        JSON config path with university/project info (currently unused,
        kept for API compatibility).
    db_file : str
        Path to SQLite database file.
    client : OpenAI
        OpenAI client instance for API calls.
    model : str, optional
        OpenAI model name to use (default: "gpt-4o").
        Examples: "gpt-4o", "gpt-5-mini", "gpt-3.5-turbo".
    subset : str or bool, optional
        Path to CSV file with html_url column to filter repositories
        (e.g., test_set file). If False or None, processes all repositories
        (default: False).
    max_workers : int, optional
        Maximum number of concurrent worker threads for API calls
        (default: 10).
    rate_limit : int, optional
        Maximum number of concurrent API calls (rate limiter)
        (default: 10).
    checkpoint_interval : int, optional
        Save progress every N completed repositories (default: 100).
    resume : bool, optional
        If True, resume from existing output file if it exists
        (default: True).
    truncate : int, optional
        Maximum number of characters per text field. Truncation only
        occurs if text length exceeds this value (default: 20000).
    truncation_type : str, optional
        Type of truncation: "start" (keep first N characters) or
        "start_end" (keep first N and last M characters) (default: "start").
    start_length : int, optional
        Number of characters from start for "start_end" truncation type
        (default: 15000).
    end_length : int, optional
        Number of characters from end for "start_end" truncation type
        (default: 5000).
    affiliated_only : bool, optional
        If True, only compute type predictions for affiliated repositories
        (default: False).
    affiliation_threshold : float, optional
        Threshold for affiliation filtering. If None, uses default thresholds:
        UCSB=0.7, UCSC=0.65, UCSD=0.4, others=0.5 (default: None).

    Returns
    -------
    str
        Path to the saved predictions CSV file. File contains columns:
        - html_url: Repository URL
        - gpt_category: Category prediction or "error" string
        - gpt_explanation: Explanation text or error message
        - Other columns from original data

    Raises
    ------
    KeyboardInterrupt
        If interrupted by user, progress is saved before raising.
    Exception
        On unexpected errors, progress is saved before raising.

    Notes
    -----
    - Progress is checkpointed periodically and on interruption/errors.
    - Resume functionality skips already processed repositories.
    - Implements concurrent processing with ThreadPoolExecutor for efficiency.
    - Uses exponential backoff retry logic for transient API errors.
    """
    
    # Determine if subset is provided and build predictions path accordingly
    subset_path = None if (subset is False or subset is None or not subset) else subset
    
    # Load subset file if provided to extract project_type and check for missing repos
    subset_df = None
    project_type_map = {}
    if subset_path:
        print(f"Filtering repositories using subset file: {subset_path}")
        try:
            subset_df = pd.read_csv(subset_path)
            if 'html_url' not in subset_df.columns:
                print(f"Warning: 'html_url' column not found in subset file. Cannot merge project_type or check for missing repos.")
            else:
                # Extract project_type column if it exists
                if 'project_type' in subset_df.columns:
                    project_type_map = dict(zip(subset_df['html_url'], subset_df['project_type']))
                    print(f"Found 'project_type' column in subset file with {len(project_type_map)} entries.")
                
                # Check which repositories in subset are not in database
                db_conn = sqlite3.connect(db_file)
                try:
                    db_urls_df = pd.read_sql_query("SELECT DISTINCT html_url FROM repositories WHERE html_url IS NOT NULL", db_conn)
                    db_urls_set = set(db_urls_df['html_url'].dropna().unique())
                    subset_urls_set = set(subset_df['html_url'].dropna().unique())
                    missing_urls = subset_urls_set - db_urls_set
                    
                    if missing_urls:
                        print(f"\n⚠️  Found {len(missing_urls)} repositories in type_test_set that are NOT in the database:")
                        for url in sorted(missing_urls):
                            print(f"   - {url}")
                        print()
                    else:
                        print(f"✓ All {len(subset_urls_set)} repositories from type_test_set are present in the database.\n")
                except Exception as e:
                    print(f"Warning: Could not check for missing repositories in database: {e}")
                finally:
                    db_conn.close()
        except Exception as e:
            print(f"Warning: Could not read subset file {subset_path}: {e}")
            subset_df = None
    
    # Add suffixes to filename based on parameters
    subset_suffix = "_subset" if subset_path else ""
    affiliated_suffix = "_affiliated" if affiliated_only else ""
    predictions_path = f'results/{acronym}/repo_type_{model}_{acronym}{subset_suffix}{affiliated_suffix}.csv'
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    
    # STEP 1: Load repositories from database first
    df = get_type_combined_data(config_file, db_file, acronym, truncate=truncate, subset=subset_path,
                               truncation_type=truncation_type, start_length=start_length, end_length=end_length)
    
    initial_count = len(df)
    print(f"\nTotal repositories loaded: {initial_count}")
    
    # STEP 1.5: Filter by affiliation if requested
    if affiliated_only:
        # Determine threshold
        if affiliation_threshold is None:
            default_thresholds = {
                "UCSB": 0.7,
                "UCSC": 0.65,
                "UCSD": 0.4,
            }
            threshold = default_thresholds.get(acronym, 0.5)
        else:
            threshold = affiliation_threshold
        
        print(f"Filtering for affiliated repositories only (threshold: {threshold})")
        
        # Query database directly for affiliation predictions
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        try:
            # Check which affiliation prediction column exists
            cursor.execute("PRAGMA table_info(repositories)")
            columns = [row[1] for row in cursor.fetchall()]
            
            prediction_col = None
            for col in columns:
                if 'affiliation_prediction_gpt' in col.lower() and '5' in col.lower() and 'mini' in col.lower():
                    prediction_col = col
                    break
            
            if not prediction_col:
                # Try alternative naming
                if "affiliation_prediction_gpt-5-mini" in columns:
                    prediction_col = "affiliation_prediction_gpt-5-mini"
                elif "affiliation_prediction_gpt_5_mini" in columns:
                    prediction_col = "affiliation_prediction_gpt_5_mini"
            
            if prediction_col:
                # Get affiliated repository URLs from database
                query = f"""
                    SELECT html_url FROM repositories 
                    WHERE CAST({prediction_col} AS REAL) > ?
                """
                cursor.execute(query, (threshold,))
                affiliated_urls = set(row[0] for row in cursor.fetchall() if row[0])
                
                # Filter DataFrame to only include affiliated repositories
                df = df[df['html_url'].isin(affiliated_urls)]
                affiliated_count = len(df)
                print(f"  - Repositories after affiliation filter: {affiliated_count}")
                print(f"  - Excluded (not affiliated): {initial_count - affiliated_count}")
            else:
                print(f"  - Warning: No affiliation prediction column found in database, skipping affiliation filter")
                
        except Exception as e:
            print(f"  - Warning: Could not query affiliation predictions from database: {e}")
        finally:
            conn.close()
    
    total_repositories = len(df)
    print(f"\nTotal repositories to check: {total_repositories}")
    
    # STEP 2: Load existing predictions from file and compare
    existing_results = {}
    existing_urls_set = set()
    existing_df = None
    if resume and os.path.exists(predictions_path):
        try:
            existing_df = pd.read_csv(predictions_path)
            if 'html_url' in existing_df.columns and 'gpt_category' in existing_df.columns:
                # Create a dict mapping html_url to (gpt_category, gpt_explanation)
                # Also create a set for fast lookup
                for _, row in existing_df.iterrows():
                    html_url = row.get('html_url')
                    if pd.notna(html_url):
                        existing_urls_set.add(html_url)
                        # Only include if it has a valid prediction (not error, not empty)
                        if pd.notna(row.get('gpt_category')) and row.get('gpt_category') != 'error':
                            existing_results[html_url] = (
                                row.get('gpt_category'),
                                row.get('gpt_explanation', '')
                            )
                print(f"\nFound {len(existing_urls_set)} repositories in predictions file")
                print(f"  - {len(existing_results)} with valid predictions")
                print(f"  - {len(existing_urls_set) - len(existing_results)} with errors or empty")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting fresh.")
    else:
        print(f"\nNo existing predictions file found (or resume=False). Starting fresh.")
    
    # STEP 3: Compare repositories with predictions file
    df_urls_set = set(df['html_url'].dropna().unique())
    already_in_predictions = df_urls_set.intersection(existing_urls_set)
    new_repositories = df_urls_set - already_in_predictions
    
    print(f"\nRepository comparison:")
    print(f"  - Total repositories from database: {total_repositories}")
    print(f"  - Already in predictions file: {len(already_in_predictions)}")
    print(f"  - New repositories to process: {len(new_repositories)}")
    
    # Store full DataFrame for merging at the end
    full_df = df.copy()
    
    # Filter DataFrame to only include repositories that need processing
    if len(new_repositories) > 0:
        df_to_process = df[df['html_url'].isin(new_repositories)].copy()
        print(f"\nComputing predictions for {len(new_repositories)} new repositories...")
    else:
        print(f"\nAll repositories already have predictions. Loading existing results...")
        # If all repositories are already in predictions file, merge with existing_df
        if existing_df is not None:
            # Merge existing predictions with current DataFrame
            df = full_df.merge(existing_df[['html_url', 'gpt_category', 'gpt_explanation']], 
                         on='html_url', how='left')
            # Add project_type column from subset file if available
            if project_type_map:
                df['project_type'] = df['html_url'].map(project_type_map)
            df.to_csv(predictions_path, index=False, escapechar='\\')
            print(f"✓ All predictions loaded from existing file: {predictions_path}")
            return predictions_path
        else:
            print("Warning: No existing predictions file found but all repositories are marked as processed.")
            return predictions_path
    
    categories_definition = """
    You must classify each GitHub repository into exactly one of the following categories based on its primary purpose.
    When multiple purposes are present, classification follows the precedence rules described below.
    
    DEV: A repository primarily used for the development and maintenance of a software artifact, including tools,
    libraries, components, applications, services, or APIs. The presence of documentation, a website, or example
    code does not override this classification if active software development is the main function.
    
    EDU: A repository primarily used for educational purposes, including course-related materials (e.g.,
    assignments, labs, class projects, websites), teaching infrastructure, or instructional content explicitly tied to a
    course, workshop, or training program. This category includes course websites, teaching demos, and repositories
    created to support internships, tutorials, or learning exercises, provided their main goal is instruction
    rather than production use. **If a repository contains software or an application that was developed as part
    of completing a course or academic requirement, it must be classified as EDU, even if it resembles a
    standalone or production-style software project.**
    
    DOCS: A repository primarily used to store or track non-educational documents, such as reports, policies,
    white papers, specifications, or meeting notes. Repositories whose content is documentation for a software
    project should be classified as DEV or WEB instead, depending on their primary role.
    
    WEB: A repository primarily used to host a public-facing website or informational page, such as a project homepage, 
    documentation website, or static or CMS-based informational site. This includes repositories built with static site generators 
    (e.g., Jekyll, Hugo), as well as site-specific styles, layouts, or templates, when the main purpose is presentation 
    rather than software development. Reusable themes, design systems, or templates intended for general adoption are 
    excluded and classified as DEV. Personal websites, portfolios, or “about me” pages are excluded and classified as OTHER.
    
    DATA: A repository primarily used to store, curate, or distribute datasets, including research datasets,
    benchmarks, or data collections (including images). Lightweight scripts for data loading or inspection do not change this
    classification.
    
    OTHER: A repository that does not clearly align with any of the above categories, such as empty repositories,
    personal experiments, configuration-only repositories, or miscellaneous content without a clear primary
    purpose.
    """.strip()

    
    # Cache system message (same for all rows)
    system_message = "You are an expert GitHub repository classifier. Always respond in the requested format."
    
    # Rate limiting semaphore
    rate_limiter = Semaphore(rate_limit)
    
    # Prepare row data for processing - convert DataFrame to list of dicts for easier handling
    df_dict = df_to_process.to_dict('records')
    
    # Initialize results arrays (maintain original DataFrame order for df_to_process)
    results = [None] * len(df_to_process)
    explanations = [None] * len(df_to_process)
    rows_to_process = []
    
    # All rows in df_to_process need processing (we already filtered out existing ones)
    for i, row in enumerate(df_dict):
        rows_to_process.append((i, row))  # Store (original_index, row_data)
    
    num_to_process = len(rows_to_process)
    total_rows = len(df_to_process)
    
    # Print summary
    print(f"Starting processing of {num_to_process} repositories with {max_workers} workers...")
    
    if num_to_process == 0:
        print("All repositories already processed!")
        # Merge existing predictions with full DataFrame
        if existing_df is not None:
            df = full_df.merge(existing_df[['html_url', 'gpt_category', 'gpt_explanation']], 
                             on='html_url', how='left')
        else:
            df = full_df.copy()
            df['gpt_category'] = None
            df['gpt_explanation'] = None
        # Add project_type column from subset file if available
        if project_type_map:
            df['project_type'] = df['html_url'].map(project_type_map)
        df.to_csv(predictions_path, index=False, escapechar='\\')
        return predictions_path
    
    # Prepare row data for worker threads
    rows_data = [
        (row, categories_definition, client, model, rate_limiter, system_message, original_idx)
        for original_idx, row in rows_to_process
    ]
    
    # Helper function to save checkpoint (defined outside thread pool for reuse)
    def save_checkpoint(current_count):
        """Save current progress to disk."""
        try:
            # Add predictions to df_to_process
            df_to_process['gpt_category'] = results
            df_to_process['gpt_explanation'] = explanations
            
            # Merge new predictions with existing ones
            if existing_df is not None:
                # Combine existing and new predictions
                combined_df = pd.concat([existing_df, df_to_process], ignore_index=True)
                # Remove duplicates, keeping the most recent (new predictions)
                combined_df = combined_df.drop_duplicates(subset=['html_url'], keep='last')
            else:
                combined_df = df_to_process.copy()
            
            # Merge with full DataFrame to include all repositories
            df_merged = full_df.merge(combined_df[['html_url', 'gpt_category', 'gpt_explanation']], 
                                    on='html_url', how='left')
            
            # Add project_type column from subset file if available
            if project_type_map:
                df_merged['project_type'] = df_merged['html_url'].map(project_type_map)
            
            df_merged.to_csv(predictions_path, index=False, escapechar='\\')
            print(f"✓ Checkpoint saved: {current_count}/{total_rows} new repositories processed")
            return True
        except Exception as save_error:
            print(f"⚠ Warning: Failed to save checkpoint: {save_error}")
            return False
    
    # Initialize completed_count
    completed_count = 0
    last_checkpoint = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks: map future -> original_df_index
            future_to_original_idx = {}
            for row_data in rows_data:
                original_idx = row_data[-1]  # Last element is the original index
                future = executor.submit(_process_type_row, row_data)
                future_to_original_idx[future] = original_idx
            
            # Collect results as they complete
            for future in as_completed(future_to_original_idx):
                try:
                    original_idx, category, explanation = future.result()
                    results[original_idx] = category
                    explanations[original_idx] = explanation
                    completed_count += 1
                    percentage = (completed_count / total_rows) * 100
                    print(f"Progress: {completed_count}/{total_rows} repositories processed ({percentage:.1f}%)")
                    
                except Exception as e:
                    original_idx = future_to_original_idx[future]
                    completed_count += 1
                    percentage = (completed_count / total_rows) * 100
                    print(f"Unexpected error processing row {original_idx + 1}: {e}")
                    print(f"Progress: {completed_count}/{total_rows} repositories processed ({percentage:.1f}%)")
                    results[original_idx] = "error"
                    explanations[original_idx] = str(e)
                
                # Periodic checkpointing (save every N repositories, regardless of success/failure)
                if completed_count - last_checkpoint >= checkpoint_interval:
                    save_checkpoint(completed_count)
                    last_checkpoint = completed_count
        
        print(f"Completed processing all {total_rows} new repositories!")
        
        # Final checkpoint save (save remaining progress even if we haven't reached the interval)
        if completed_count > last_checkpoint:
            print(f"Saving final checkpoint ({completed_count}/{total_rows} new repositories)...")
            save_checkpoint(completed_count)
            print(f"✓ Final results saved to {predictions_path}")
        
        # Final merge: combine new predictions with existing ones
        df_to_process['gpt_category'] = results
        df_to_process['gpt_explanation'] = explanations
        
        if existing_df is not None:
            # Combine existing and new predictions
            combined_df = pd.concat([existing_df, df_to_process], ignore_index=True)
            # Remove duplicates, keeping the most recent (new predictions)
            combined_df = combined_df.drop_duplicates(subset=['html_url'], keep='last')
        else:
            combined_df = df_to_process.copy()
        
        # Merge with full DataFrame to include all repositories
        df_final = full_df.merge(combined_df[['html_url', 'gpt_category', 'gpt_explanation']], 
                                on='html_url', how='left')
        
        # Add project_type column from subset file if available
        if project_type_map:
            df_final['project_type'] = df_final['html_url'].map(project_type_map)
        
        df_final.to_csv(predictions_path, index=False, escapechar='\\')
        print(f"✓ Final predictions saved: {len(df_final)} total repositories ({len(new_repositories)} new + {len(already_in_predictions)} existing)")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user! Saving progress before exit...")
        save_checkpoint(completed_count)
        print(f"✓ Progress saved: {completed_count}/{total_rows} new repositories processed")
        print(f"  You can resume by running the same command again (it will skip already processed repositories)")
        raise  # Re-raise to allow caller to handle interruption
    
    except Exception as e:
        print(f"\n⚠ Unexpected error occurred: {e}")
        print(f"Saving progress before exit...")
        save_checkpoint(completed_count)
        print(f"✓ Progress saved: {completed_count}/{total_rows} new repositories processed")
        print(f"  You can resume by running the same command again (it will skip already processed repositories)")
        raise  # Re-raise to allow caller to handle error

    return predictions_path

