#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import time
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
                                max_workers=10, rate_limit=10, checkpoint_interval=100, resume=True,
                                truncate=20000, truncation_type="start", start_length=15000, end_length=5000):
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
    
    predictions_path = f'results/{acronym}/repo_type_{model}_{acronym}.csv'
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    
    # Load existing results if resuming
    existing_results = {}
    if resume and os.path.exists(predictions_path):
        try:
            existing_df = pd.read_csv(predictions_path)
            if 'html_url' in existing_df.columns and 'gpt_category' in existing_df.columns:
                # Create a dict mapping html_url to (gpt_category, gpt_explanation)
                for _, row in existing_df.iterrows():
                    if pd.notna(row.get('gpt_category')) and row.get('gpt_category') != 'error':
                        existing_results[row['html_url']] = (
                            row.get('gpt_category'),
                            row.get('gpt_explanation', '')
                        )
                print(f"Found {len(existing_results)} already processed repositories. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting fresh.")
    
    # Pass subset to get_type_combined_data (if subset is False or None, pass None; otherwise pass the file path)
    subset_path = None if (subset is False or subset is None or not subset) else subset
    if subset_path:
        print(f"Filtering repositories using subset file: {subset_path}")
    df = get_type_combined_data(config_file, db_file, acronym, truncate=truncate, subset=subset_path,
                               truncation_type=truncation_type, start_length=start_length, end_length=end_length)
    
    categories_definition = """
    You must classify each GitHub repository into exactly one of the following categories:
    
    
    DEV: a repository primarily used for development of a tool, component, application, app, or API  
    EDU: a repository primarily used for educational purposes, including course-related student work (e.g., assignments or class projects) or instructional materials (e.g., tutorials or lectures). 
    DOCS: a repository primarily used for tracking and storage of non-educational documents  
    WEB: a repository primarily used to host a public-facing website, documentation site, or informational page. Indicators include terms like site, webpage, homepage, docs site, or project page. It should not be a personal portfolio or "about me" page — those go in OTHER.
    DATA: a repository primarily used to store data sets  
    OTHER: use this category only if there is no strong correlation to any other repository category, for example, empty repositories
    
    
    Choose the most appropriate category based on the repository information.
    """.strip()
    
    # Cache system message (same for all rows)
    system_message = "You are an expert GitHub repository classifier. Always respond in the requested format."
    
    # Rate limiting semaphore
    rate_limiter = Semaphore(rate_limit)
    
    # Prepare row data for processing - convert DataFrame to list of dicts for easier handling
    df_dict = df.to_dict('records')
    
    # Initialize results arrays (maintain original DataFrame order)
    results = [None] * len(df)
    explanations = [None] * len(df)
    rows_to_process = []
    
    # Fill in existing results and identify rows to process
    num_skipped = 0
    
    for i, row in enumerate(df_dict):
        html_url = row.get('html_url', '')
        
        # Check if already processed
        if html_url in existing_results:
            results[i] = existing_results[html_url][0]
            explanations[i] = existing_results[html_url][1]
            num_skipped += 1
        else:
            rows_to_process.append((i, row))  # Store (original_index, row_data)
    
    num_to_process = len(rows_to_process)
    total_rows = len(df)
    
    # Print summary
    if num_skipped > 0:
        print(f"Skipping {num_skipped} already processed repositories. Processing {num_to_process} remaining...")
    else:
        print(f"Starting processing of {num_to_process} repositories with {max_workers} workers...")
    
    if num_to_process == 0:
        print("All repositories already processed!")
        df['gpt_category'] = results
        df['gpt_explanation'] = explanations
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
            df['gpt_category'] = results
            df['gpt_explanation'] = explanations
            df.to_csv(predictions_path, index=False, escapechar='\\')
            print(f"✓ Checkpoint saved: {current_count}/{total_rows} repositories processed")
            return True
        except Exception as save_error:
            print(f"⚠ Warning: Failed to save checkpoint: {save_error}")
            return False
    
    # Initialize completed_count with already processed repositories
    completed_count = num_skipped
    last_checkpoint = num_skipped
    
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
        
        print(f"Completed processing all {total_rows} repositories!")
        
        # Final checkpoint save (save remaining progress even if we haven't reached the interval)
        if completed_count > last_checkpoint:
            print(f"Saving final checkpoint ({completed_count}/{total_rows} repositories)...")
            save_checkpoint(completed_count)
            print(f"✓ Final results saved to {predictions_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user! Saving progress before exit...")
        save_checkpoint(completed_count)
        print(f"✓ Progress saved: {completed_count}/{total_rows} repositories processed")
        print(f"  You can resume by running the same command again (it will skip already processed repositories)")
        raise  # Re-raise to allow caller to handle interruption
    
    except Exception as e:
        print(f"\n⚠ Unexpected error occurred: {e}")
        print(f"Saving progress before exit...")
        save_checkpoint(completed_count)
        print(f"✓ Progress saved: {completed_count}/{total_rows} repositories processed")
        print(f"  You can resume by running the same command again (it will skip already processed repositories)")
        raise  # Re-raise to allow caller to handle error
    
    return predictions_path

