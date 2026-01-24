import json
import logging
import os
from repofinder.scraping.repo_scraping_utils import search_organizations_with_queries, build_org_queries

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
    

def search_organizations(config, headers):
    """
    Searches for organizations based on query terms built from an environment file 
    and saves the results to a JSON file.

    Parameters
    ----------
    env : str
        The path to the environment configuration file (e.g., a JSON file) containing
        necessary parameters for query construction.

    Returns
    -------
    None
        The function saves the retrieved organizations in a JSON file named 
        `repository_data_<university_acronym>.json` in the `Data/` directory.

    """
    
    query_terms, university_acronym = build_org_queries(config)
    organizations = search_organizations_with_queries(query_terms, headers)
    output_filename_json = f"Data/json/organization_data_{university_acronym}.json"
    with open(output_filename_json, 'w', encoding='utf-8') as f:
        json.dump(organizations, f, ensure_ascii=False, indent=4)
    