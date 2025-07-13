"""
Helper functions for the LLM module.
"""

import time
from contextlib import contextmanager

@contextmanager
def timing(description: str = "Operation"):
    start = time.time()
    yield
    end = time.time()
    print(f"{description} took {end - start:.2f} seconds.")

# Placeholder for utility functions related to the LLM module.

def example_util_function(param1, param2):
    """
    An example of a utility function that could be used within the LLM module.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.

    Returns:
        Description of the return value.
    """
    # ...function logic...

    return