#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Azure OpenAI client using the Responses API.

Uses environment variables for configuration. Does not modify the existing
OpenAI client setup in main_filtering.py.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI


# Load .env from project root (directory containing this file)
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_required_env(name: str) -> str:
    """Get a required env var; raise a clear error if missing or empty."""
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Set it in .env (e.g. {name}=your_value)."
        )
    return value


def get_azure_client() -> AzureOpenAI:
    """
    Build and return an Azure OpenAI client from environment variables.

    Required in .env:
        AZURE_OPENAI_ENDPOINT   - e.g. https://your-resource.openai.azure.com/
        AZURE_OPENAI_API_KEY    - Your Azure OpenAI API key
        AZURE_OPENAI_API_VERSION - e.g. 2024-02-15-preview
    """
    endpoint = _get_required_env("AZURE_OPENAI_ENDPOINT")
    api_key = _get_required_env("AZURE_OPENAI_API_KEY")
    api_version = _get_required_env("AZURE_OPENAI_API_VERSION")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def ask_ai(
    prompt: str,
    *,
    deployment: Optional[str] = None,
    instructions: str = "You are a helpful assistant.",
) -> str:
    """
    Send a prompt to Azure OpenAI (Responses API) and return the text output.

    Args:
        prompt: The user prompt to send.
        deployment: Deployment name (model). If None, uses AZURE_OPENAI_DEPLOYMENT.
        instructions: Optional system/instruction text for the model.

    Returns:
        The model's text response.

    Raises:
        ValueError: If required env vars or deployment is missing.
        Exception: On API or network errors.
    """
    deploy = (deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
    if not deploy:
        raise ValueError(
            "Deployment not specified. Pass deployment='...' or set "
            "AZURE_OPENAI_DEPLOYMENT in .env."
        )

    client = get_azure_client()

    # Use the Responses API (not legacy chat completions)
    response = client.responses.create(
        model=deploy,
        input=prompt,
        instructions=instructions,
    )

    # Responses API returns output_text for the main text content
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    # Fallback for different response shapes (e.g. output list of items)
    if hasattr(response, "output") and response.output:
        parts = getattr(response.output, "output", response.output)
        if isinstance(parts, list) and len(parts) > 0:
            first = parts[0]
            if hasattr(first, "content") and isinstance(first.content, list):
                for block in first.content:
                    if getattr(block, "type", None) == "output_text" and hasattr(block, "text"):
                        return block.text
            if hasattr(first, "text"):
                return first.text
    raise RuntimeError("Could not extract text from Responses API response.")


def main() -> None:
    """Example usage: one-off prompt via command line or default demo."""
    try:
        # Validate env and client before running
        client = get_azure_client()
        deployment = _get_required_env("AZURE_OPENAI_DEPLOYMENT")
        print(f"Using deployment: {deployment}")
        print("Sending example prompt...")
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Example prompt
    prompt = "Say hello in one short sentence."
    try:
        answer = ask_ai(prompt)
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
