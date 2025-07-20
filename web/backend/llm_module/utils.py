"""
File: web/backend/llm_module/utils.py

Role:
    This file serves as a collection of miscellaneous helper functions and utilities that support
    the LLM module. It is intended for general-purpose tools that can be reused across different
    parts of the LLM pipeline.

Interactions:
    - `evaluator.py`: Uses the `timing` context manager for simple performance logging of different
      pipeline stages.
    - Other modules can import functions from here as needed.
"""

import time
from contextlib import contextmanager

import requests
from django.conf import settings


@contextmanager
def timing(description: str = "Operation"):
    start = time.time()
    yield
    end = time.time()
    print(f"{description} took {end - start:.2f} seconds.")


def get_api_client():
    """
    Returns an API client (requests.Session) with authentication headers.
    """
    session = requests.Session()
    if hasattr(settings, 'API_AUTH_TOKEN'):
        session.headers.update({'Authorization': f'Token {settings.API_AUTH_TOKEN}'})
    else:
        print("Warning: API_AUTH_TOKEN not found in settings. API calls will be unauthenticated.")
    return session


def create_and_persist_evaluation_result(api_client, query_id, company_id, pdf_file_id, result_data):
    """
    Helper function to create an EvaluationResult through the API.
    """
    payload = {
        'query': query_id,
        'company': company_id,
        'pdf_file': pdf_file_id,  # Use the ID of the PDF file
        'timestamp': result_data['timestamp'],
        'per_query_data': result_data['per_query_data'],
        'references': result_data['references'],
        'model_version': result_data.get('model_version'),
        'processing_time': result_data.get('processing_time')
    }
    response = api_client.post('/api/evaluationresults/', json=payload)
    if response.status_code != 201:
        print(f"Error creating evaluation result: {response.status_code} - {response.text}")
    return response
