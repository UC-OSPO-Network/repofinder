
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:10:35 2025

@author: juanitagomez
"""

from repofinder.filtering.filter_utils import get_type_combined_data


def compute_ai_type_predictions(acronym, config_file, db_file, client, model="gpt-4o", subset=False):
    """
    Classify repositories into categories (DEV, EDU, DOCS, WEB, DATA, OTHER)
    using OpenAI GPT based on core repo metadata.

    Args:
        acronym (str): University or project acronym (used for output path).
        config_file (str): JSON config path with university/project info.
        db_file (str): Path to SQLite database file.
        client: OpenAI API client instance.
        model (str): OpenAI model to use (default "gpt-4o").
        subset (bool or str): Optional path to CSV subset to filter repos.

    Returns:
        str: Path to CSV file with predictions.
    """

    df = get_type_combined_data(config_file, db_file, acronym, truncate=20000)

    results = []
    explanations = []

    categories_definition = """
    You must classify each GitHub repository into exactly one of the following categories:
    
    
    DEV: a repository primarily used for development of a tool, component, application, app, or API  
    EDU: a repository primarily used for educational purposes, including course-related student work (e.g., assignments or class projects) or instructional materials (e.g., tutorials or lectures). 
    DOCS: a repository primarily used for tracking and storage of non-educational documents  
    WEB: a repository primarily used to host a public-facing website, documentation site, or informational page. Indicators include terms like site, webpage, homepage, docs site, or project page. It should not be a personal portfolio or “about me” page — those go in OTHER.
    DATA: a repository primarily used to store data sets  
    OTHER: use this category only if there is no strong correlation to any other repository category, for example, empty repositories
    
    
    Choose the most appropriate category based on the repository information.
    """.strip()

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):  # start index at 1
        print(f"Processing {acronym} row {i} of {total}")
        
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
    Return only the predicted category (one of: DEV, HW repo, EDU, DOCS, WEB, DATA, OTHER), and a short explanation.
    
    Format your response exactly as:
    
    Category: <one of the 7 categories>  
    Explanation: <brief explanation>
    """.strip()

        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert GitHub repository classifier. Always respond in the requested format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0
            }
            if model.startswith("gpt-4o"):
                kwargs["seed"] = 42

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()

            category = "error"
            explanation = ""

            for line in content.splitlines():
                if line.lower().startswith("category:"):
                    category = line.split(":", 1)[1].strip()
                elif line.lower().startswith("explanation:"):
                    explanation = line.split(":", 1)[1].strip()

        except Exception as e:
            category = "error"
            explanation = str(e)

        results.append(category)
        explanations.append(explanation)

    df['gpt_category'] = results
    df['gpt_explanation'] = explanations
    
    predictions_path = f'results/{acronym}/repo_type_{model}_{acronym}.csv'
    df.to_csv(predictions_path, index=False, escapechar='\\')

    return predictions_path

