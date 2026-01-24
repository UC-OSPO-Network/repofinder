#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
from openai import OpenAI
from repofinder.filtering.filter_utils import get_combined_data
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

def _format_field_with_truncation_info(field_name, field_value, truncation_type, truncate, start_length, end_length):
    """
    Format a field value with explicit truncation information for LLM prompts.

    This function formats field values with truncation indicators, explicitly
    informing the LLM when text has been truncated and which portions are shown.
    Supports two truncation types: "start" (keep first N characters) and
    "start_end" (keep first N and last M characters, separated by "...").

    Parameters
    ----------
    field_name : str
        The name of the field being formatted (e.g., "Readme", "Description").
    field_value : str or None
        The actual field value. Can be None, empty, or a string.
    truncation_type : str
        Type of truncation applied: "start" or "start_end".
    truncate : int
        Maximum length threshold for truncation (e.g., 20000). Truncation
        only occurs if text length exceeds this value.
    start_length : int
        Number of characters to keep from the start for "start_end" truncation.
    end_length : int
        Number of characters to keep from the end for "start_end" truncation.

    Returns
    -------
    str
        Formatted string with field name and value, including truncation
        indicators when applicable. Returns "{field_name}: (empty)" for
        empty or None values.
    """
    # Handle pandas NaN and None values
    if pd.isna(field_value) or field_value is None:
        return f"{field_name}: (empty)"
    
    field_str = str(field_value)
    
    # Skip empty or placeholder values
    field_str_lower = field_str.lower().strip()
    if not field_str_lower or field_str_lower in ['nan', 'none', '']:
        return f"{field_name}: (empty)"
    
    # Check if field was truncated based on truncation type
    if truncation_type == "start_end":
        # For start_end: check if "..." separator exists (indicating truncation happened)
        # The truncate_text function adds "..." when text exceeds max_length (truncate parameter, e.g., 20000)
        # Only truncates if text length > truncate (20000 characters)
        if "..." in field_str:
            # Split the text at "..." to separate first and last parts
            # Use maxsplit=1 to only split on the first occurrence
            parts = field_str.split("...", 1)
            first_part = parts[0] if len(parts) > 0 else ""
            last_part = parts[1] if len(parts) > 1 else ""
            
            # Format with explicit labels for first and last parts
            return f"""{field_name} (NOTE: This text was truncated to show only the first {start_length:,} characters and last {end_length:,} characters):
    - First part of {field_name.lower()} (first {start_length:,} characters): {first_part}
    - Last part of {field_name.lower()} (last {end_length:,} characters): {last_part}"""
        else:
            # No truncation separator found - text was short enough to include fully
            # Check length to confirm it wasn't truncated (text must be <= 20000 for no truncation)
            if len(field_str) <= truncate:
                return f"{field_name}: {field_str}"
            else:
                # This shouldn't happen with our truncate_text function, but handle it anyway
                return f"{field_name}: {field_str}"
    elif truncation_type == "start":
        # For start truncation: if length equals or exceeds truncate limit, it was likely truncated
        if len(field_str) >= truncate:
            return f"{field_name} (NOTE: This text was truncated. Showing only the first {truncate:,} characters):\n{field_str}"
        else:
            return f"{field_name}: {field_str}"
    else:
        # Unknown truncation type - just return as-is
        return f"{field_name}: {field_str}"

def _process_row(row_data):
    """
    Process a single repository row and return GPT prediction with retry logic.

    This function handles API calls to OpenAI GPT for a single repository,
    including rate limiting, retry logic with exponential backoff, and response
    parsing. It formats repository information with truncation indicators and
    constructs the prompt for the LLM.

    Parameters
    ----------
    row_data : tuple
        Tuple containing:
            - row (dict): Repository data dictionary
            - config (dict): University configuration dictionary
            - belonging_definition (str): Definition of repository belonging
            - client: OpenAI client instance
            - model (str): GPT model name (e.g., "gpt-4o", "gpt-5-mini")
            - rate_limiter (Semaphore): Thread-safe rate limiter
            - system_message (str): System message for GPT
            - row_idx (int): Original DataFrame index
            - truncation_type (str): Type of truncation ("start" or "start_end")
            - truncate (int): Truncation threshold
            - start_length (int): Start length for "start_end" truncation
            - end_length (int): End length for "start_end" truncation

    Returns
    -------
    tuple
        Three-element tuple containing:
            - row_idx (int): Original DataFrame index
            - answer (float or str): Probability value (0-1) or "error" string
            - explanation (str): Explanation text or error message

    Notes
    -----
    - Implements exponential backoff retry for transient errors (rate limits,
      timeouts, server errors)
    - Non-retryable errors return immediately with "error" as answer
    - Maximum 3 retry attempts with delays: 1s, 2s, 4s
    """
    row, config, belonging_definition, client, model, rate_limiter, system_message, row_idx, truncation_type, truncate, start_length, end_length = row_data
    
    # Build the information to send to GPT with explicit truncation info
    readme_value = row.get('readme', '')
    description_value = row.get('description', '')  # Repository description
    org_description_value = row.get('org_description', '')  # Organization description
    c1_bio_value = row.get('c1_bio', '')
    c2_bio_value = row.get('c2_bio', '')
    
    repo_info = f"""
    Repository Information:
    - HTML URL: {row.get('html_url', '')}
    - Homepage: {row.get('homepage', '')}
    - {_format_field_with_truncation_info('Readme', readme_value, truncation_type, truncate, start_length, end_length)}
    - {_format_field_with_truncation_info('Description', description_value, truncation_type, truncate, start_length, end_length)}
    - Full Name: {row.get('full_name', '')}
    - Organization: {row.get('organization', '')}
    - Owner: {row.get('owner', '')}
    - Organization URL: {row.get('url', '')}
    - Organization Email: {row.get('email', '')}
    - Organization Location: {row.get('location', '')}
    - Organization Name: {row.get('name', '')}
    - {_format_field_with_truncation_info('Organization Description', org_description_value, truncation_type, truncate, start_length, end_length)}
    - Organization Company: {row.get('company', '')}
    
    Top Contributor 1:
    - Login: {row.get('c1', '')}
    - {_format_field_with_truncation_info('Bio', c1_bio_value, truncation_type, truncate, start_length, end_length)}
    - Company: {row.get('c1_company', '')}
    - Email: {row.get('c1_email', '')}
    
    Top Contributor 2:
    - Login: {row.get('c2', '')}
    - {_format_field_with_truncation_info('Bio', c2_bio_value, truncation_type, truncate, start_length, end_length)}
    - Company: {row.get('c2_company', '')}
    - Email: {row.get('c2_email', '')}
    """.strip()

    # Add the university information
    university_info = f"""
    University Context:
    - University Name: {config['UNIVERSITY_NAME']}
    - University Acronym: {config['UNIVERSITY_ACRONYM']}
    - University Email Domain: {config['UNIVERSITY_EMAIL_DOMAIN']}
    - Additional Queries: {', '.join(config['ADDITIONAL_QUERIES'])}
    """.strip()

    # Final prompt
    prompt = f"""
    You are tasked with determining the likelihood that a GitHub repository belongs to a university based on the following definition:
    
    {belonging_definition}
    
    Here is the information about the repository:
    
    {repo_info}
    
    And here is the university context:
    
    {university_info}
    
    Based on this information and the definition:
    
    Answer with a single value: a probability between 0 and 1 (e.g., 0.87) representing how likely it is that the repository belongs to the university.
    
    Provide a brief explanation (1-2 sentences) justifying your answer.
    Your response must be formatted exactly like this:
    
    Probability: <value between 0 and 1>  
    Explanation: <your explanation here>
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
                answer = "error"
                explanation = ""
                for line in content.splitlines():
                    if line.lower().startswith("probability:"):
                        answer = float(line.split(":", 1)[1])
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
                    answer = "error"
                    explanation = str(e)
                    break
    
    return row_idx, answer, explanation

def compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-5-mini", subset=False, 
                           max_workers=30, rate_limit=10, checkpoint_interval=100, resume=True,
                           truncate=20000, truncation_type="start", start_length=15000, end_length=5000):
    """
    Classify whether repositories belong to a university using OpenAI's GPT API.

    This function processes repositories from a database and uses OpenAI GPT
    models to predict their affiliation with a university. It supports concurrent
    processing, checkpointing, resume capability, and handles empty repositories
    by assigning them a prediction of "-1".

    Parameters
    ----------
    acronym : str
        University acronym (e.g., "UCSD", "UCSC").
    config_file : str
        Path to JSON config file containing university information.
    db_file : str
        Path to SQLite database file containing repository data.
    client : OpenAI
        OpenAI client instance for API calls.
    model : str, optional
        OpenAI model name to use (default: "gpt-5-mini").
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
        - gpt_belonging: Probability prediction (0-1) or "-1" for empty repos
        - gpt_explanation: Explanation text or error message
        - Other columns from original data (e.g. readme, description, etc.)

    Raises
    ------
    KeyboardInterrupt
        If interrupted by user, progress is saved before raising.
    Exception
        On unexpected errors, progress is saved before raising.

    Notes
    -----
    - Empty repositories (size <= 0) are assigned prediction "-1" without
      API calls to save costs.
    - Progress is checkpointed periodically and on interruption/errors.
    - Resume functionality skips already processed repositories.
    - Implements concurrent processing with ThreadPoolExecutor for efficiency.
    - Uses exponential backoff retry logic for transient API errors.
    """

    def clean_model_name(model):
        """
        Clean the model name by replacing "." with "-"
        """
        model = model.replace(".", "-")
        return model
    
    # STEP 1: Determine subset path and build predictions path
    # Pass subset to get_combined_data (if subset is False or None, pass None; otherwise pass the file path)
    subset_path = None if (subset is False or subset is None or not subset) else subset
    
    # Load subset file if provided to check for missing repos
    subset_df = None
    if subset_path:
        print(f"Filtering repositories using subset file: {subset_path}")
        try:
            subset_df = pd.read_csv(subset_path)
            if 'html_url' not in subset_df.columns:
                print(f"Warning: 'html_url' column not found in subset file. Cannot check for missing repos.")
            else:
                # Check which repositories in subset are not in database
                db_conn = sqlite3.connect(db_file)
                try:
                    db_urls_df = pd.read_sql_query("SELECT DISTINCT html_url FROM repositories WHERE html_url IS NOT NULL", db_conn)
                    db_urls_set = set(db_urls_df['html_url'].dropna().unique())
                    subset_urls_set = set(subset_df['html_url'].dropna().unique())
                    missing_urls = subset_urls_set - db_urls_set
                    
                    if missing_urls:
                        print(f"\n⚠️  Found {len(missing_urls)} repositories in test set that are NOT in the database:")
                        for url in sorted(missing_urls):
                            print(f"   - {url}")
                        print()
                    else:
                        print(f"✓ All {len(subset_urls_set)} repositories from test set are present in the database.\n")
                except Exception as e:
                    print(f"Warning: Could not check for missing repositories in database: {e}")
                finally:
                    db_conn.close()
        except Exception as e:
            print(f"Warning: Could not read subset file {subset_path}: {e}")
            subset_df = None
    
    # Add "_subset" suffix to filename if subset is provided
    subset_suffix = "_subset" if subset_path else ""
    predictions_path = f'results/{acronym}/predictions_ai_{clean_model_name(model)}_{acronym}{subset_suffix}.csv'
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    
    df = get_combined_data(config_file, db_file, acronym, truncate=truncate, subset=subset_path, 
                           truncation_type=truncation_type, start_length=start_length, end_length=end_length)
    
    # Filter repositories: not archived, size > 0, not a fork, not a template
    initial_count = len(df)
    print(f"\nTotal repositories loaded: {initial_count}")
    
    # Apply filters
    # Keep repositories that are: not archived, size > 0, not fork, not template
    # Handle various data types (boolean, int, string) and NULL/NaN values
    
    if 'archived' in df.columns:
        # Convert to numeric for comparison, keeping NaN
        archived_numeric = pd.to_numeric(df['archived'], errors='coerce')
        archived_bool = df['archived'] == False
        # Keep: archived is 0, False, or NaN
        df = df[(archived_numeric == 0) | (archived_bool) | (archived_numeric.isna())]
    
    if 'size' in df.columns:
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        # Keep: size > 0 OR size is NaN (matching SQL query: size > 0 OR size IS NULL)
        df = df[(df['size'] > 0) | (df['size'].isna())]
    
    if 'fork' in df.columns:
        # Convert to numeric for comparison, keeping NaN
        fork_numeric = pd.to_numeric(df['fork'], errors='coerce')
        fork_bool = df['fork'] == False
        # Keep: fork is 0, False, or NaN
        df = df[(fork_numeric == 0) | (fork_bool) | (fork_numeric.isna())]
    
    if 'is_template' in df.columns:
        # Convert to numeric for comparison, keeping NaN
        template_numeric = pd.to_numeric(df['is_template'], errors='coerce')
        template_bool = df['is_template'] == False
        # Keep: is_template is 0, False, or NaN
        df = df[(template_numeric == 0) | (template_bool) | (template_numeric.isna())]
    
    filtered_count = len(df)
    print(f"After filtering (not archived, size > 0, not fork, not template): {filtered_count}")
    print(f"  - Filtered out: {initial_count - filtered_count} repositories")
    
    # Print breakdown of what was filtered
    if initial_count > 0:
        print(f"  - Retention rate: {(filtered_count / initial_count * 100):.1f}%")
    
    total_repositories = len(df)
    print(f"\nTotal repositories to check: {total_repositories}")
    print(f"Note: This is the number of repositories that will be candidates for prediction.")
    print(f"      The number of 'affiliated' repositories will be determined AFTER predictions are computed and threshold is applied.")
    
    # STEP 2: Load existing predictions from file and compare
    existing_results = {}
    existing_urls_set = set()
    if resume and os.path.exists(predictions_path):
        try:
            existing_df = pd.read_csv(predictions_path)
            if 'html_url' in existing_df.columns and 'gpt_belonging' in existing_df.columns:
                # Create a dict mapping html_url to (gpt_belonging, gpt_explanation)
                # Also create a set for fast lookup
                for _, row in existing_df.iterrows():
                    html_url = row.get('html_url')
                    if pd.notna(html_url):
                        # Only add to existing_urls_set if it has a VALID prediction
                        # This way, repositories with errors will be reprocessed
                        if pd.notna(row.get('gpt_belonging')) and row.get('gpt_belonging') != 'error':
                            existing_urls_set.add(html_url)
                            existing_results[html_url] = (
                                row.get('gpt_belonging'),
                                row.get('gpt_explanation', '')
                            )
                print(f"Found {len(existing_urls_set)} repositories in predictions file")
                print(f"  - {len(existing_results)} with valid predictions")
                print(f"  - {len(existing_urls_set) - len(existing_results)} with errors or empty")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting fresh.")
    else:
        print(f"No existing predictions file found (or resume=False). Starting fresh.")
    
    # STEP 3: Compare repositories with predictions file
    df_urls_set = set(df['html_url'].dropna().unique())
    already_in_predictions = df_urls_set.intersection(existing_urls_set)
    new_repositories = df_urls_set - existing_urls_set
    
    print(f"\nRepository comparison:")
    print(f"  - Total repositories from database: {total_repositories}")
    print(f"  - Already in predictions file: {len(already_in_predictions)}")
    print(f"  - New repositories to process: {len(new_repositories)}")
    print(f"\nComputing predictions for {len(new_repositories)} new repositories...")
    
    with open(config_file, encoding="utf-8") as envfile:
        config = json.load(envfile)

    # Prepare the definition (cached, same for all rows)
    belonging_definition = """A repository is considered affiliated with a university if there is clear, public evidence that its development, maintenance, or governance is connected to the university. This includes any of the following criteria:

    1. Research or Academic Unit Development:Developed or maintained by a research group, lab, center, academic department, or institute that is formally part of the university.
    Evidence: README statements such as “developed at …”, linked lab websites, or institutional pages under the university domain.

    2. Contributor Affiliation (Explicit Evidence):
    One or more key contributors (maintainers, lead developers, primary committers) are affiliated with the university, with explicit evidence such as:

    - University email addresses listed for contact with the university domain.
    - Public profiles or bios naming the university,
    - README sections listing team members and their university affiliation.

    3. Institutional or Administrative Development:
    Developed or maintained by a non-academic but official university unit (e.g., libraries, Open Source Program Offices, IT departments, research computing groups).

    4. Official Ownership, Sponsorship, or Endorsement: The repository is owned, sponsored, or explicitly endorsed by the university.
    Evidence includes:
    - Hosting under an official university GitHub organization,
    - README or website statements indicating university sponsorship or ownership,
    - Copyright notices naming the university.

    5. Documented Collaboration or Partnership
    - The README or project website explicitly states a collaboration, partnership, or joint development effort with the university or one of its units.

    6. University-Linked Project Infrastructure:
    The project’s official homepage, documentation site, or primary web presence is hosted under the university’s domain or clearly associated subdomain.

    7. Educational Outreach and Online Courses}:
        Online learning initiatives affiliated with the university, including: (1) Repositories linked to online specializations, or courses offered on platforms like Coursera, edX, or similar (2) Course materials, code examples, or tools developed specifically for such offerings.
    """.strip()

    # Cache system message (same for all rows)
    system_message = "You are a careful repository classifier that follows the exact response format."
    
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
    num_empty = 0
    
    for i, row in enumerate(df_dict):
        html_url = row.get('html_url', '')
        size = row.get('size', 0)
        
        # Check if already in predictions file (skip these)
        if html_url in existing_urls_set:
            # Use existing prediction if available, otherwise mark as skipped
            if html_url in existing_results:
                results[i] = existing_results[html_url][0]
                explanations[i] = existing_results[html_url][1]
            else:
                # In predictions file but no valid prediction - skip it
                results[i] = "error"
                explanations[i] = "Already in predictions file but marked as error"
            num_skipped += 1
        # Check if repository is empty (size <= 0) - only for NEW repositories
        else:
            try:
                size_val = int(float(size)) if size is not None and not pd.isna(size) else 0
                if size_val <= 0:
                    results[i] = "-1"
                    explanations[i] = "Empty repository (size <= 0)"
                    num_empty += 1
                else:
                    rows_to_process.append((i, row))  # Store (original_index, row_data)
            except (ValueError, TypeError):
                # If size cannot be converted, treat as empty
                results[i] = "-1"
                explanations[i] = "Empty repository (size <= 0)"
                num_empty += 1
    
    num_to_process = len(rows_to_process)
    total_rows = len(df)
    
    # Print summary
    if num_empty > 0:
        print(f"Marking {num_empty} empty repositories (size <= 0) with prediction '-1'")
    if num_skipped > 0:
        print(f"Skipping {num_skipped} already processed repositories. Processing {num_to_process} remaining...")
    else:
        print(f"Starting processing of {num_to_process} non-empty repositories with {max_workers} workers...")
    
    if num_to_process == 0:
        print("All repositories already processed!")
        df['gpt_belonging'] = results
        df['gpt_explanation'] = explanations
        df.to_csv(predictions_path, index=False, escapechar='\\')
        return predictions_path
    
    # Prepare row data for worker threads: (row, config, ..., original_df_index, truncation_info)
    rows_data = [
        (row, config, belonging_definition, client, model, rate_limiter, system_message, original_idx,
         truncation_type, truncate, start_length, end_length)
        for original_idx, row in rows_to_process
    ]
    
    # Helper function to save checkpoint (defined outside thread pool for reuse)
    def save_checkpoint(current_count):
        """Save current progress to disk."""
        try:
            df['gpt_belonging'] = results
            df['gpt_explanation'] = explanations
            df.to_csv(predictions_path, index=False, escapechar='\\')
            print(f"✓ Checkpoint saved: {current_count}/{total_rows} repositories processed")
            return True
        except Exception as save_error:
            print(f"⚠ Warning: Failed to save checkpoint: {save_error}")
            return False
    
    # Initialize completed_count with already processed and empty repositories
    completed_count = num_skipped + num_empty
    last_checkpoint = num_skipped + num_empty
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks: map future -> original_df_index
            future_to_original_idx = {}
            for row_data in rows_data:
                original_idx = row_data[-1]  # Last element is the original index
                future = executor.submit(_process_row, row_data)
                future_to_original_idx[future] = original_idx
            
            # Collect results as they complete
            for future in as_completed(future_to_original_idx):
                try:
                    original_idx, answer, explanation = future.result()
                    results[original_idx] = answer
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
