#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from repofinder.filtering.filter_utils import get_combined_data

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


