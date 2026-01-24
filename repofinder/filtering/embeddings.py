#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from repofinder.filtering.filter_utils import get_combined_data, truncate_text

def get_embedding(text, client, model="text-embedding-3-small", max_chars=10000, start_length=7000, end_length=3000):
    """
    Get embedding for text, truncating if necessary to fit model context limits.
    
    Args:
        text (str): Text to embed.
        client: OpenAI client instance.
        model (str): Embedding model name (default: "text-embedding-3-small").
        max_chars (int): Maximum characters threshold for truncation (default: 8000).
            If text exceeds this, uses start_end truncation.
        start_length (int): Characters to keep from start for start_end truncation (default: 5000).
        end_length (int): Characters to keep from end for start_end truncation (default: 3000).
            text-embedding-3-small has 8192 token limit. Using conservative 8k chars (~6k tokens).
    
    Returns:
        list: Embedding vector.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Truncate text if it's too long (embedding models have token limits)
    # text-embedding-3-small: 8192 tokens
    # Tokenization varies, so use conservative limit: 8000 chars ≈ ~6000 tokens (safe margin)
    # Using start_end truncation: keep first 5000 and last 3000 chars (total ~8000)
    if len(text) > max_chars:
        text = truncate_text(text, max_chars, truncation_type="start_end", 
                            start_length=start_length, end_length=end_length)
    
    # Replace newlines before sending (this doesn't change token count significantly)
    text = text.replace("\n", " ")
    
    # Final safety check: if still too long after truncation, use simple start truncation
    # This handles edge cases where truncate_text might not work as expected
    if len(text) > 8000:
        text = text[:8000]
    
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def build_matrix_with_embeddings(env, db_file, acronym, client, subset=None):
    """
    Builds an embedding matrix from combined repository metadata and saves it as a CSV file.

    Args:
        env (str): Path to the JSON config file with university metadata.
        db_file (str): Path to the SQLite database file.
        acronym (str): University acronym used for naming the output file.
        client (object): An OpenAI client instance used to generate embeddings.
        subset (str, optional): Path to CSV file with html_url column. If provided,
            includes repositories from subset AND any repositories with manual_label.
    
    Returns:
        str: Path to the generated CSV file containing the embedding matrix.
    """
    
    # Get all data first (without subset filtering in get_combined_data)
    df = get_combined_data(env, db_file, acronym, subset=None)

    # Filter out repositories without a readme
    initial_count = len(df)
    df = df[df['readme'].notna() & (df['readme'].str.strip() != '') & (df['readme'].str.lower() != 'nan')]
    readme_filtered_count = len(df)
    
    print(f"Filtering repositories by readme availability:")
    print(f"  - Initial repositories: {initial_count}")
    print(f"  - After readme filter: {readme_filtered_count}")
    print(f"  - Excluded (no readme): {initial_count - readme_filtered_count}")
    
    # Apply subset filtering if provided
    if subset:
        subset_df = pd.read_csv(subset)
        if 'html_url' not in subset_df.columns:
            raise ValueError("Subset CSV must contain 'html_url' column.")
        
        subset_urls = set(subset_df['html_url'].dropna().unique())
        
        # Keep rows that are either:
        # 1. In the subset (by html_url), OR 
        # 2. Have a manual_label (not null/empty/-1/None)
        has_manual_label = (
            df['manual_label'].notna() & 
            (df['manual_label'].astype(str).str.strip() != '') & 
            (df['manual_label'].astype(str).str.lower() != 'nan') &
            (df['manual_label'].astype(str).str.lower() != 'none') &
            (df['manual_label'].astype(str) != '-1') &
            (df['manual_label'] != -1)  # Also check numeric -1
        )
        in_subset = df['html_url'].isin(subset_urls)
        
        df = df[in_subset | has_manual_label]
        
        subset_filtered_count = len(df)
        manual_count = has_manual_label.sum()
        subset_count = in_subset.sum()
        overlap_count = (in_subset & has_manual_label).sum()
        
        print(f"Applying subset filtering:")
        print(f"  - Repositories in subset: {subset_count}")
        print(f"  - Repositories with manual labels: {manual_count}")
        print(f"  - Overlap (in subset AND manual label): {overlap_count}")
        print(f"  - Final matrix size: {subset_filtered_count}")
        print(f"  - Excluded by subset filter: {readme_filtered_count - subset_filtered_count}")
    
    final_count = len(df)
    print(f"\nFinal embedding matrix will contain {final_count} repositories")
 
    # Generate embeddings
    df['embedding'] = df['combined'].apply(lambda x: get_embedding(x, client))

    # Create embedding matrix
    embedding_matrix = pd.DataFrame(df['embedding'].to_list(), index=df['html_url'])
    embedding_matrix.reset_index(inplace=True)
    embedding_matrix['manual_label'] = df['manual_label']

    # Add subset suffix to filename if subset was used
    if subset:
        output_path = f'results/{acronym}/repository_embeddings_{acronym}_subset.csv'
    else:
        output_path = f'results/{acronym}/repository_embeddings_{acronym}.csv'

    embedding_matrix.to_csv(output_path, index=False)
    return output_path


