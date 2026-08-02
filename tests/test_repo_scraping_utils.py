import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.ModuleType("pandas")
    unittest.addModuleCleanup(sys.modules.pop, "pandas", None)

if "requests" not in sys.modules:
    requests = types.ModuleType("requests")
    requests.exceptions = types.SimpleNamespace(RequestException=Exception)
    requests.get = None
    requests.post = None
    sys.modules["requests"] = requests
    unittest.addModuleCleanup(sys.modules.pop, "requests", None)

from repofinder.scraping import repo_scraping_utils as utils


class GraphQLRepositoryFetchingTests(unittest.TestCase):
    def test_graphql_request_keeps_partial_data_with_errors(self):
        class Response:
            status_code = 200
            headers = {}

            def json(self):
                return {
                    "errors": [{"message": "Could not resolve to a User"}],
                    "data": {
                        "owner_0": None,
                        "owner_1": {"repositories": {"nodes": []}},
                    },
                }

        with patch.object(utils.requests, "post", return_value=Response()):
            data = utils.graphql_api_request("query { viewer { login } }", {})

        self.assertEqual(data["owner_0"], None)
        self.assertIn("owner_1", data)

    def test_repository_node_maps_rest_compatible_fields(self):
        node = {
            "databaseId": 123,
            "id": "R_123",
            "name": "repo",
            "nameWithOwner": "owner/repo",
            "description": "description",
            "diskUsage": 42,
            "homepageUrl": "https://example.com",
            "url": "https://github.com/owner/repo",
            "sshUrl": "git@github.com:owner/repo.git",
            "isPrivate": False,
            "isFork": False,
            "isArchived": False,
            "isDisabled": False,
            "isTemplate": False,
            "hasIssuesEnabled": True,
            "hasProjectsEnabled": True,
            "hasWikiEnabled": False,
            "hasDiscussionsEnabled": False,
            "visibility": "PUBLIC",
            "stargazerCount": 5,
            "forkCount": 2,
            "issues": {"totalCount": 3},
            "watchers": {"totalCount": 1},
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-02T00:00:00Z",
            "pushedAt": "2024-01-03T00:00:00Z",
            "primaryLanguage": {"name": "Python"},
            "owner": {
                "login": "owner",
                "avatarUrl": "https://example.com/avatar",
                "url": "https://github.com/owner",
                "__typename": "Organization",
            },
            "licenseInfo": {
                "key": "mit",
                "name": "MIT License",
                "spdxId": "MIT",
                "url": "https://api.github.com/licenses/mit",
            },
            "repositoryTopics": {
                "nodes": [{"topic": {"name": "science"}}],
            },
            "defaultBranchRef": {"name": "main"},
        }

        repo = utils._repository_node_to_rest_dict(node)

        self.assertEqual(repo["node_id"], "R_123")
        self.assertEqual(repo["url"], "https://api.github.com/repos/owner/repo")
        self.assertEqual(repo["clone_url"], "https://github.com/owner/repo.git")
        self.assertEqual(repo["ssh_url"], "git@github.com:owner/repo.git")
        self.assertEqual(repo["git_url"], "git://github.com/owner/repo.git")
        self.assertEqual(repo["size"], 42)
        self.assertEqual(repo["open_issues_count"], 3)
        self.assertEqual(repo["watchers_count"], 5)
        self.assertEqual(repo["subscribers_count"], 1)
        self.assertEqual(repo["has_issues"], True)
        self.assertEqual(repo["has_wiki"], False)
        self.assertEqual(repo["visibility"], "public")
        self.assertEqual(repo["topics"], ["science"])

    def test_graphql_fetch_reports_crossed_progress_thresholds(self):
        owners = [{"login": f"owner-{i}", "repos_url": f"https://example.com/{i}"} for i in range(11)]
        data = {}
        for index in range(11):
            data[f"owner_{index}"] = {
                "repositories": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }

        output = io.StringIO()
        with patch.object(utils, "graphql_api_request", return_value=data), redirect_stdout(output):
            repos, processed = utils._fetch_repositories_graphql("user", owners, {})

        self.assertEqual(repos, [])
        self.assertEqual(processed, 11)
        self.assertIn("Processed 10/11 users...", output.getvalue())
        self.assertIn("Processed 11/11 users...", output.getvalue())

    def test_graphql_fetch_falls_back_only_for_missing_alias(self):
        owners = [
            {"login": "missing", "repos_url": "https://example.com/missing"},
            {"login": "present", "repos_url": "https://example.com/present"},
        ]
        data = {
            "owner_0": None,
            "owner_1": {
                "repositories": {
                    "nodes": [
                        {
                            "databaseId": 123,
                            "name": "repo",
                            "nameWithOwner": "present/repo",
                            "url": "https://github.com/present/repo",
                            "repositoryTopics": {"nodes": []},
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            },
        }

        with (
            patch.object(utils, "graphql_api_request", return_value=data),
            patch.object(
                utils,
                "_fetch_repositories_rest_paginated",
                return_value=[{"full_name": "missing/repo"}],
            ) as rest_fallback,
            redirect_stdout(io.StringIO()),
        ):
            repos, processed = utils._fetch_repositories_graphql("user", owners, {})

        self.assertEqual(processed, 2)
        self.assertEqual(rest_fallback.call_count, 1)
        self.assertEqual(repos[0]["full_name"], "missing/repo")
        self.assertEqual(repos[1]["full_name"], "present/repo")


if __name__ == "__main__":
    unittest.main()
