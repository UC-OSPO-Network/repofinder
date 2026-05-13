#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import logging
import time
import json 
import itertools
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


GITHUB_API_URL = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
GRAPHQL_REPO_BATCH_SIZE = 50
GRAPHQL_REPO_PAGE_SIZE = 100


def github_api_request(url, headers, params=None, rate_limiter=None):
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
    rate_limiter : Semaphore, optional
        Thread-safe rate limiter for concurrent requests (default is None).

    Returns
    -------
    tuple
        A tuple containing:
        - dict: The JSON response from the API.
        - dict: The response headers.
    """
    # Use rate limiter if provided
    if rate_limiter:
        rate_limiter.acquire()
    
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            logger.debug(f"Attempt {attempt} for URL: {url}")
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"Failed after {MAX_RETRIES} attempts: {url} - {e}")
                    return None, None
                # Check if retryable
                error_msg = str(e).lower()
                is_retryable = any(keyword in error_msg for keyword in [
                    'timeout', '503', '502', '500', '429', 'connection'
                ])
                if is_retryable:
                    wait_time = RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(f"Retryable error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Non-retryable error: {e}")
                    return None, None
            
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
                    sleep_time = max(reset_time - int(time.time()), 1)
                    logger.warning(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                    time.sleep(sleep_time)
                    continue  # Retry after sleeping
            else:
                logger.error(f"Error: {response.status_code} - {response.reason}")
                if attempt == MAX_RETRIES:
                    return None, None
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                time.sleep(wait_time)
                continue
        logger.error(f"Failed to get a successful response after {MAX_RETRIES} attempts: {url}")
        return None, None
    finally:
        if rate_limiter:
            rate_limiter.release()


def graphql_api_request(query, headers, rate_limiter=None):
    """
    Sends a POST request to the GitHub GraphQL API with retry and rate-limit handling.

    Parameters
    ----------
    query : str
        GraphQL query string.
    headers : dict
        HTTP headers for authenticated GitHub API requests.
    rate_limiter : Semaphore, optional
        Thread-safe rate limiter for concurrent requests (default is None).

    Returns
    -------
    dict or None
        The GraphQL response data, or None if the request fails.
    """
    if rate_limiter:
        rate_limiter.acquire()

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            logger.debug(f"GraphQL attempt {attempt}")
            try:
                response = requests.post(
                    GITHUB_GRAPHQL_URL,
                    headers=headers,
                    json={"query": query},
                    timeout=30,
                )
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"GraphQL request failed after {MAX_RETRIES} attempts: {e}")
                    return None
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"GraphQL request error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code == 200:
                payload = response.json()
                if payload.get("errors"):
                    logger.error(f"GraphQL errors: {payload['errors']}")
                    return None
                return payload.get("data")

            if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
                sleep_time = max(reset_time - int(time.time()), 1)
                logger.warning(f"GraphQL rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                continue

            logger.error(f"GraphQL error: {response.status_code} - {response.reason}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

        return None
    finally:
        if rate_limiter:
            rate_limiter.release()


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


def _is_bot_login(login):
    if not isinstance(login, str):
        return False
    bots_to_skip = ["copilot", "dependabot", "github-actions"]
    login_lower = login.lower()
    return any(bot in login_lower for bot in bots_to_skip) or login.endswith("[bot]")


def _repository_graphql_fields():
    return """
        repositories(first: %s, after: %s, ownerAffiliations: OWNER) {
          nodes {
            databaseId
            name
            nameWithOwner
            description
            homepageUrl
            url
            isPrivate
            isFork
            isArchived
            isDisabled
            stargazerCount
            forkCount
            createdAt
            updatedAt
            pushedAt
            primaryLanguage { name }
            owner {
              login
              avatarUrl
              url
              __typename
            }
            licenseInfo {
              key
              name
              spdxId
              url
            }
            repositoryTopics(first: 100) {
              nodes {
                topic { name }
              }
            }
            defaultBranchRef { name }
          }
          pageInfo {
            hasNextPage
            endCursor
          }
        }
    """


def _build_repository_batch_query(owner_kind, owners):
    fields = []
    for index, owner in enumerate(owners):
        alias = f"owner_{index}"
        after = json.dumps(owner.get("cursor")) if owner.get("cursor") else "null"
        login = json.dumps(owner["login"])
        fields.append(
            f"""
            {alias}: {owner_kind}(login: {login}) {{
              {_repository_graphql_fields() % (GRAPHQL_REPO_PAGE_SIZE, after)}
            }}
            """
        )
    return "query BatchRepositories {\n" + "\n".join(fields) + "\n}"


def _repository_node_to_rest_dict(node):
    owner = node.get("owner") or {}
    license_info = node.get("licenseInfo")
    topics = [
        topic_node.get("topic", {}).get("name")
        for topic_node in (node.get("repositoryTopics") or {}).get("nodes", [])
        if topic_node.get("topic", {}).get("name")
    ]

    return {
        "id": node.get("databaseId"),
        "name": node.get("name"),
        "full_name": node.get("nameWithOwner"),
        "html_url": node.get("url"),
        "description": node.get("description"),
        "homepage": node.get("homepageUrl"),
        "private": node.get("isPrivate"),
        "fork": node.get("isFork"),
        "archived": node.get("isArchived"),
        "disabled": node.get("isDisabled"),
        "stargazers_count": node.get("stargazerCount"),
        "forks_count": node.get("forkCount"),
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "pushed_at": node.get("pushedAt"),
        "language": (node.get("primaryLanguage") or {}).get("name"),
        "topics": topics,
        "default_branch": (node.get("defaultBranchRef") or {}).get("name"),
        "owner": {
            "login": owner.get("login"),
            "avatar_url": owner.get("avatarUrl"),
            "html_url": owner.get("url"),
            "type": "Organization" if owner.get("__typename") == "Organization" else "User",
        },
        "license": {
            "key": license_info.get("key"),
            "name": license_info.get("name"),
            "spdx_id": license_info.get("spdxId"),
            "url": license_info.get("url"),
        } if license_info else None,
    }


def _fetch_repositories_rest_paginated(owner_dict, headers):
    repos_url = owner_dict.get("repos_url")
    if not repos_url:
        return []

    repos = []
    url = repos_url
    params = {"per_page": 100}
    while url:
        repositories, response_headers = github_api_request(url, headers, params)
        if repositories:
            repos.extend(repositories)
            url = get_next_link(response_headers or {})
            params = None
        else:
            break
    return repos


def _fetch_repositories_graphql(owner_kind, owner_dicts, headers):
    owners = [{"login": owner["login"], "cursor": None, "source": owner} for owner in owner_dicts]
    repos = []
    processed = 0

    while owners:
        batch = owners[:GRAPHQL_REPO_BATCH_SIZE]
        owners = owners[GRAPHQL_REPO_BATCH_SIZE:]
        query = _build_repository_batch_query(owner_kind, batch)
        data = graphql_api_request(query, headers)

        if data is None:
            for owner in batch:
                repos.extend(_fetch_repositories_rest_paginated(owner["source"], headers))
                processed += 1
            continue

        for index, owner in enumerate(batch):
            owner_data = data.get(f"owner_{index}")
            if not owner_data:
                repos.extend(_fetch_repositories_rest_paginated(owner["source"], headers))
                processed += 1
                continue

            repositories = owner_data.get("repositories") or {}
            repos.extend(
                _repository_node_to_rest_dict(node)
                for node in repositories.get("nodes", [])
                if node
            )

            page_info = repositories.get("pageInfo") or {}
            if page_info.get("hasNextPage"):
                owners.append({
                    "login": owner["login"],
                    "cursor": page_info.get("endCursor"),
                    "source": owner["source"],
                })
            else:
                processed += 1

        if processed and processed % 10 == 0:
            print(f"Processed {processed}/{len(owner_dicts)} {owner_kind}s...")

    return repos, processed


def build_repo_queries(config_file):
    """
    Builds a list of GitHub search queries based on university-related metadata.

    Parameters
    ----------
    config_file : str
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
    # Add filters for archived:false and size:>0 to all queries
    base_filters = "archived:false size:>0"
    query_terms = [
        f'"{term}" {field} {base_filters}'
        for term, field in itertools.product(query_terms_list, search_fields)
    ] + [f'"{university_email_domain}" in:email {base_filters}']

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


def build_org_queries(config_file):
    """
    Builds a list of GitHub search queries based on university-related metadata.

    Parameters
    ----------
    config_file : str
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
    search_fields = ["in:name","in:login","in:description","in:company"]

    # Combine university metadata and additional queries
    query_terms_list = [university_acronym, university_email_domain, university_name] + additional_queries

    # Generate query terms with itertools.product
    # Add type:Organization filter to ensure we only get organizations
    query_terms = [
        f'"{term}" {field} type:Organization'
        for term, field in itertools.product(query_terms_list, search_fields)
    ] + [f'"{university_email_domain}" in:blog type:Organization'] 

    return query_terms, university_acronym


def build_user_queries(config_file):
    """
    Builds a list of GitHub user search queries based on university-related metadata.

    Parameters
    ----------
    config_file : str
        The path to the JSON configuration file containing university details.

    Returns
    -------
    tuple
        A tuple containing:
        - list of str: Search query terms for GitHub users.
        - str: The university acronym for output file naming.

    Notes
    -----
    - Reads university metadata from the provided JSON file.
    - Constructs query terms based on the university's name, acronym, email domain, and additional queries.
    - Searches in bio and company fields for all terms.
    - Searches in blog and email fields for email domain only.
    """
    
    with open(config_file, encoding="utf-8") as envfile:
        config = json.load(envfile)

    # Assign values to variables using keys from the config
    university_name = config["UNIVERSITY_NAME"]
    university_acronym = config["UNIVERSITY_ACRONYM"]
    university_email_domain = config["UNIVERSITY_EMAIL_DOMAIN"]
    additional_queries = config["ADDITIONAL_QUERIES"]

    # Define search fields for bio and company (applied to all terms)
    search_fields = ["in:name","in:login","in:bio","in:company"]

    # Combine university metadata and additional queries
    query_terms_list = [university_name, university_acronym, university_email_domain] + additional_queries

    # Generate query terms with itertools.product for bio and company
    # Add type:User filter to ensure we only get users (not organizations)
    query_terms = [
        f'"{term}" {field} type:User'
        for term, field in itertools.product(query_terms_list, search_fields)
    ]
    
    # Add email domain specific searches for blog and email
    # Include type:User filter for these as well
    query_terms.extend([
        f'"{university_email_domain}" in:blog type:User',
    ])

    return query_terms, university_acronym


def search_users_with_queries(query_terms, headers):
    """
    Searches GitHub users based on query terms.

    Args:
        query_terms (list): List of query strings.
        headers (dict): HTTP headers for the request.

    Returns:
        list: A list of user objects.
    """
    users = []  # TODO: Figure out what to do with duplicates
    for query_term in query_terms:
        params = {'q': query_term, 'per_page': 100}
        url = f"{GITHUB_API_URL}/search/users"
        while url:
            logger.debug(f"Searching users with URL: {url} and params: {params}")
            try:
                data, headers_response = github_api_request(url, headers, params)
            except Exception as e:
                logger.error(f"Error searching users: {e}")
                break
            if data:  # TODO: Figure out caching
                items = data.get('items', [])
                users.extend(items)
                next_url = get_next_link(headers_response)
                url = next_url
                params = None  # Parameters are only needed for the initial request
            else:
                break
    return users


def search_organizations_with_queries(query_terms, headers):
    """
    Searches GitHub organizations based on query terms.

    Args:
        query_terms (list): List of query strings.
        headers (dict): HTTP headers for the request.

    Returns:
        list: A list of organization objects.
    """
    organizations = [] # TODO: Figure out what to do with duplicates
    for query_term in query_terms:
        params = {'q': query_term, 'per_page': 100}
        url = f"{GITHUB_API_URL}/search/users"
        while url:
            logger.debug(f"Searching organizations with URL: {url} and params: {params}")
            try:
                data, headers_response = github_api_request(url, headers, params)
            except Exception as e:
                logger.error(f"Error searching organizations: {e}")
                break
            if data:  # TODO: Figure out caching
                items = data.get('items', [])
                organizations.extend(items)
                next_url = get_next_link(headers_response)
                url = next_url
                params = None  # Parameters are only needed for the initial request
            else:
                break
    return organizations


def get_repositories_from_organizations(university_acronym, org_json, headers):
    org_df = pd.read_json(org_json)
    org_dicts = org_df.to_dict('records')
    
    # Filter out bots before counting
    valid_orgs = [org_dict for org_dict in org_dicts 
                  if org_dict.get("type") == "Organization" 
                  and isinstance(org_dict.get("login"), str)
                  and not _is_bot_login(org_dict.get("login", ""))]
    total_orgs = len(valid_orgs)

    repos, processed = _fetch_repositories_graphql("organization", valid_orgs, headers)
    
    output_filename_json = f"Data/json/repository_data_org_scraping_{university_acronym}.json"
    os.makedirs('Data/json', exist_ok=True)
    with open(output_filename_json, 'w', encoding='utf-8') as f:
        json.dump(repos, f, ensure_ascii=False, indent=4)
    
    print(f"Completed: Collected repositories from {processed}/{total_orgs} organizations")


def get_repositories_from_users(university_acronym, user_json, headers):
    """
    Retrieves repositories owned by users (not organizations) from a JSON file of users.
    
    Parameters
    ----------
    university_acronym : str
        The university acronym for output file naming.
    user_json : str
        Path to the JSON file containing user data.
    headers : dict
        HTTP headers for authenticated GitHub API requests.
    
    Returns
    -------
    None
        Saves repositories to a JSON file named 
        `repository_data_user_scraping_{university_acronym}.json` in the `Data/json/` directory.
    """
    user_df = pd.read_json(user_json)
    user_dicts = user_df.to_dict('records')
    
    # Filter out bots before counting
    valid_users = [user_dict for user_dict in user_dicts 
                   if user_dict.get("type") != "Organization"
                   and isinstance(user_dict.get("login"), str)
                   and not _is_bot_login(user_dict.get("login", ""))]
    total_users = len(valid_users)

    repos, processed = _fetch_repositories_graphql("user", valid_users, headers)
    
    output_filename_json = f"Data/json/repository_data_user_scraping_{university_acronym}.json"
    os.makedirs('Data/json', exist_ok=True)
    with open(output_filename_json, 'w', encoding='utf-8') as f:
        json.dump(repos, f, ensure_ascii=False, indent=4)
    
    print(f"Completed: Collected repositories from {processed}/{total_users} users")
        
def process_items_concurrently(items, process_func, max_workers=10, rate_limit=10):
    """
    Process items concurrently with rate limiting.
    
    Parameters
    ----------
    items : list
        List of items to process.
    process_func : callable
        Function to process each item. Should accept (item, rate_limiter) and return result.
    max_workers : int, optional
        Maximum number of concurrent workers (default: 10).
    rate_limit : int, optional
        Maximum concurrent API calls (default: 10).
    
    Returns
    -------
    list
        List of results from process_func.
    """
    rate_limiter = Semaphore(rate_limit)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item, rate_limiter): item for item in items}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
    
    return results
