
import json
import logging
import os
import time
from scraping.repo_scraping_utils import search_repositories_with_queries, build_queries

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def repo_finder(env, path, headers):   
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

    Notes
    -----
    - The function reads the environment file and constructs search queries.
    - It retrieves repositories using `search_repositories_with_queries`.
    - The results are logged and stored in a JSON file.
    - The execution time is printed at the end.
    """

    query_terms, university_acronym = build_queries(env)
    repositories = search_repositories_with_queries(query_terms, headers)
    logger.info(f"Total repositories found: {len(repositories)}")
    output_filename_json = f"{path}/Data/json/repository_data_{university_acronym}.json"
    
    with open(output_filename_json, 'w', encoding='utf-8') as f:
        json.dump(repositories, f, ensure_ascii=False, indent=4)




