"""
Defines result and query schemas for evaluation results.
Supports fixed outer schema and flexible per-query inner schema.
"""

from typing import Dict, Any, List
from datetime import datetime

def build_result_schema(query_id: int, company_id: int, references: List[Dict[str, Any]], per_query_data: Dict[str, Any], user: str = None, model_version: str = None, processing_time: float = None) -> Dict[str, Any]:
    """
    Builds the result schema for a query evaluation.
    """
    return {
        'query_id': query_id,
        'company_id': company_id,
        'timestamp': datetime.utcnow().isoformat(),
        'references': references,  # List of dicts: pdf, page, span, snippet
        'per_query_data': per_query_data,  # Flexible inner schema
        'user': user,
        'model_version': model_version,
        'processing_time': processing_time
    }

# Example reference dict:
# {
#   'pdf_id': 123,
#   'page_num': 5,
#   'span': [100, 200],
#   'snippet': 'CO2 emissions: 1234 t',
#   'confidence': 0.92
# }
