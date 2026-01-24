import json
import logging
import os
from repofinder.scraping.repo_scraping_utils import search_users_with_queries, build_user_queries

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def search_users(config, headers):
    """
    Searches for users based on query terms built from a configuration file 
    and saves the results to a JSON file.

    Parameters
    ----------
    config : str
        The path to the configuration file (e.g., a JSON file) containing
        necessary parameters for query construction.
    headers : dict
        HTTP headers for authenticated GitHub API requests.

    Returns
    -------
    None
        The function saves the retrieved users in a JSON file named 
        `user_data_{university_acronym}.json` in the `Data/json/` directory.
    """
    
    query_terms, university_acronym = build_user_queries(config)
    users = search_users_with_queries(query_terms, headers)
    logger.info(f"Total users found: {len(users)}")
    os.makedirs('Data/json', exist_ok=True)
    os.makedirs('Data/db', exist_ok=True)
    output_filename_json = f"Data/json/user_data_{university_acronym}.json"
    with open(output_filename_json, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

