#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:10:35 2025

@author: juanitagomez
"""

import pandas as pd
import sqlite3
from openai import OpenAI
from repofinder.filtering.embeddings import get_combined_data
import json
import time

def compute_ai_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False):
    """
    Classify whether repositories belong to a university using OpenAI's GPT API.

    Args:
        df (pd.DataFrame): DataFrame with repository metadata.
        config (dict): Dictionary containing university information:
            - UNIVERSITY_NAME
            - UNIVERSITY_ACRONYM
            - UNIVERSITY_EMAIL_DOMAIN
            - ADDITIONAL_QUERIES (list of extra hints)
        openai_api_key (str): Your OpenAI API key.
        model (str): OpenAI model to use. Default is "gpt-4".
    Returns:
        pd.DataFrame: Original DataFrame with an added 'gpt_belonging' column ("1" or "0").
    """

    # if not subset:
    #     # Filter out rows where manual_label is missing or empty
    #     df['manual_label'] = pd.to_numeric(df['manual_label'], errors='coerce')
    #     df = df[df['manual_label'].isin([0, 1])].copy()
    
    df = get_combined_data(config_file, db_file, acronym, truncate=20000, subset=None)
    
    results = []
    explanations = []
    
    with open(config_file, encoding="utf-8") as envfile:
        config = json.load(envfile)

    # Prepare the definition
    belonging_definition = """
    A repository is considered to belong to a university if it satisfies any of the following criteria:
    
    1. Research Group Affiliation:
       Developed or maintained by a research group, academic department, research center, or lab that is officially part of the university.
    
    2. Contributor Affiliation:
       One or more key contributors (maintainers, primary committers) are students, faculty, researchers, or staff affiliated with the university. Evidence includes:
       - Public profiles listing the university,
       - University email addresses,
       - Project documentation or repository metadata.
    
    3. Institutional Development:
       Developed by an institutional unit of the university (e.g., libraries, Open Source Program Offices, IT departments, administrative offices).
    
    4. Official Sponsorship or Ownership:
       Sponsored, endorsed, or owned by the university, indicated by GitHub organization ownership, README mentions, or associated websites.
    
    5. Educational Outreach and Online Courses :
        Online learning initiatives affiliated with the university, including:
       - Repositories linked to online specializations, MOOCs, or courses offered on platforms like Coursera, edX, or similar,
       - Course materials, code examples, or tools developed specifically for such offerings
    """.strip()

    for _, row in df.iterrows():
        # Build the information to send to GPT
        repo_info = f"""
Repository Information:
- HTML URL: {row.get('html_url', '')}
- Homepage: {row.get('homepage', '')}
- Readme: {row.get('readme', '')}
- Description: {row.get('description', '')}
- Full Name: {row.get('full_name', '')}
- Organization: {row.get('organization', '')}
- Owner: {row.get('owner', '')}
- Organization URL: {row.get('url', '')}
- Organization Email: {row.get('email', '')}
- Organization Location: {row.get('location', '')}
- Organization Name: {row.get('name', '')}
- Organization Description: {row.get('description', '')}
- Organization Company: {row.get('company', '')}

Top Contributor 1:
- Login: {row.get('c1', '')}
- Bio: {row.get('c1_bio', '')}
- Company: {row.get('c1_company', '')}
- Email: {row.get('c1_email', '')}

Top Contributor 2:
- Login: {row.get('c2', '')}
- Bio: {row.get('c2_bio', '')}
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

        # Call the OpenAI API
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a careful repository classifier that follows the exact response format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0  # deterministic
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
            
            # if answer not in [1, 0]:
            #     answer = "0"  # fallback if something weird
            #     explanation = explanation or "Invalid response format, defaulted to '0'."
    
        except Exception as e:
            print(f"Error querying OpenAI for repo {row.get('html_url', '')}: {e}")
            answer = "error"
            explanation = str(e)
    
        results.append(answer)
        explanations.append(explanation)

    # Add results to the DataFrame
    df['gpt_belonging'] = results
    df['gpt_explanation'] = explanations
    predictions_path = f'results/{acronym}/ai_{model}_{acronym}_predictions.csv'
    df.to_csv(predictions_path, index=False, escapechar='\\')
    return predictions_path


def update_ai_predictions_in_db(db_path, csv_predictions, column_name):
    """
    Read a CSV with 'html_url' and 'gpt_belonging' columns,
    and update the specified column in the SQLite database
    using 'html_url' as the key.

    Args:
        db_path (str): Path to the SQLite database.
        csv_path (str): Path to the CSV file.
        column_name (str): Name of the prediction column to add/update in the database.
    """
    df = pd.read_csv(csv_predictions)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Try to add the new column to the table
    try:
        cursor.execute(f"ALTER TABLE repositories ADD COLUMN {column_name} TEXT")
        print(f"Added '{column_name}' column to the repositories table.")
    except sqlite3.OperationalError as e:
        if f"duplicate column name: {column_name}" in str(e):
            print(f"'{column_name}' column already exists. Skipping ALTER TABLE.")
        else:
            raise

    # Check and prepare data
    if 'html_url' not in df.columns or 'gpt_belonging' not in df.columns:
        raise ValueError("Expected columns 'html_url' and 'gpt_belonging' not found in the CSV.")

    update_data = df[['html_url', 'gpt_belonging']].dropna()
    update_data = update_data.rename(columns={'gpt_belonging': column_name})

    # Update each row using html_url as the key
    for _, row in update_data.iterrows():
        html_url = row['html_url']
        prediction = row[column_name]
        cursor.execute(
            f"UPDATE repositories SET {column_name} = ? WHERE html_url = ?",
            (prediction, html_url)
        )

    conn.commit()
    conn.close()
    print(f"Updated {len(update_data)} rows in the '{column_name}' column based on html_url.")

    

