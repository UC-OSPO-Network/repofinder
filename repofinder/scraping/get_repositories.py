
import json
import logging
import os
from repofinder.scraping.repo_scraping_utils import search_repositories_with_queries, build_queries

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_repositories(config, headers):
    """
    Searches for repositories based on query terms built from an environment file 
    and saves the results to a JSON file.

    Parameters
    ----------
    env : str
        The path to the environment configuration file (e.g., a JSON file) containing
        necessary parameters for query construction.

    Returns
    -------
    None
        The function saves the retrieved repositories in a JSON file named 
        `repository_data_<university_acronym>.json` in the `Data/` directory.

    """

    query_terms, university_acronym = build_queries(config)
    repositories = search_repositories_with_queries(query_terms, headers)
    logger.info(f"Total repositories found: {len(repositories)}")
    os.makedirs('Data/json', exist_ok=True)
    os.makedirs('Data/db', exist_ok=True)
    output_filename_json = f"Data/json/repository_data_{university_acronym}.json"
    
    with open(output_filename_json, 'w', encoding='utf-8') as f:
        json.dump(repositories, f, ensure_ascii=False, indent=4)




