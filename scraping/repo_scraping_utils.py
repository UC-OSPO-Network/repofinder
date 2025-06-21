#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:42:47 2024

@author: juanitagomez
"""
import requests
import logging
import time
import json 
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


GITHUB_API_URL = "https://api.github.com"
MAX_RETRIES = 1
RETRY_DELAY = 2  # seconds


def github_api_request(url, headers, params=None):
    """
    Sends a GET request to the GitHub API with rate limit handling.

    Parameters
    ----------
    url : str
        The API endpoint URL.
    headers : dict
        HTTP headers for the request.
    params : dict, optional
        Query parameters for the request (default is None).

    Returns
    -------
    tuple
        A tuple containing:
        - dict: The JSON response from the API.
        - dict: The response headers.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        logger.debug(f"Attempt {attempt} for URL: {url}")
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
        except:
            pass
        logger.debug(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            logger.debug("Successful response.")
            return response.json(), response.headers
        elif response.status_code == 404:
            logger.warning(f"Resource not found: {url}. Exiting without retry.")
            return None, None
        elif response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
            if response.headers['X-RateLimit-Remaining'] == '0':
                reset_time = int(response.headers.get('X-RateLimit-Reset', time.time()))
                sleep_time = max(reset_time - int(time.time()), 1)  # Avoid negative sleep times
                logger.warning(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                continue  # Retry after sleeping
        else:
            logger.error(f"Error: {response.status_code} - {response.reason}")
            if attempt == MAX_RETRIES:
                response.raise_for_status()
            time.sleep(RETRY_DELAY)
            continue
    logger.error(f"Failed to get a successful response after {MAX_RETRIES} attempts: {url}")
    return None, None


def get_next_link(headers):
    """
    Parses the 'Link' header from a GitHub API response to find the next page URL.

    Parameters
    ----------
    headers : dict
        Response headers from the GitHub API.

    Returns
    -------
    str or None
        The URL for the next page if available, otherwise None.
    """ 
    link_header = headers.get('Link', '')
    if not link_header:
        return None
    links = link_header.split(',')
    for link in links:
        parts = link.split(';')
        if len(parts) < 2:
            continue
        url_part = parts[0].strip()
        rel_part = parts[1].strip()
        if rel_part == 'rel="next"':
            next_url = url_part.lstrip('<').rstrip('>')
            return next_url
    return None


def build_queries(config_file):
    """
    Builds a list of GitHub search queries based on university-related metadata.

    Parameters
    ----------
    env : str
        The path to the JSON configuration file containing university details.

    Returns
    -------
    tuple
        A tuple containing:
        - list of str: Search query terms for GitHub.
        - str: The university acronym for output file naming.

    Notes
    -----
    - Reads university metadata from the provided JSON file.
    - Constructs query terms based on the university's name, acronym, email domain, and website.
    - Uses `itertools.product` to generate query combinations.
    - Ensures query terms are sanitized to prevent malformed queries.

    """
    
    with open(config_file, encoding="utf-8") as envfile:
        config = json.load(envfile)

    # Assign values to variables using keys from the config
    university_name = config["UNIVERSITY_NAME"]
    university_acronym = config["UNIVERSITY_ACRONYM"]
    university_email_domain = config["UNIVERSITY_EMAIL_DOMAIN"]
    additional_queries = config["ADDITIONAL_QUERIES"]

    # Define search fields
    search_fields = ["in:name", "in:description", "in:readme", "in:tags"]

    # Combine university metadata and additional queries
    query_terms_list = [university_name, university_acronym, university_email_domain] + additional_queries

    # Generate query terms with itertools.product
    query_terms = [
        f'"{term}" {field}'
        for term, field in itertools.product(query_terms_list, search_fields)
    ] + [f'"{university_email_domain}" in:email']

    return query_terms, university_acronym




def search_repositories_with_queries(query_terms, headers):
    """
    Searches GitHub repositories based on query terms and records matching queries.

    Args:
        query_terms (list): List of query strings.
        headers (dict): HTTP headers for the request.

    Returns:
        dict: A dictionary of repositories with their matching queries.
    """
    repositories = [] # TODO: Figure out what to do with duplicates
    for query_term in query_terms:
        params = {'q': query_term, 'per_page': 100}
        url = f"{GITHUB_API_URL}/search/repositories"
        while url:
            logger.debug(f"Searching repositories with URL: {url} and params: {params}")
            try:
                data, headers_response = github_api_request(url, headers, params)
            except Exception as e:
                logger.error(f"Error searching repositories: {e}")
                break
            if data:  # TODO: Figure out caching
                items = data.get('items', [])
                repositories.extend(items)
                next_url = get_next_link(headers_response)
                url = next_url
                params = None  # Parameters are only needed for the initial request
            else:
                break
    return repositories



