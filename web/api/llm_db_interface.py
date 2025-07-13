"""
Database interface for PDF vector storage and retrieval.
Abstracts away Django ORM details for use in LLM module.
"""

import numpy as np
from .models import PDFFile, PDFVector

class LLMDBInterface:
    def save_pdf_vector(self, pdf_id: int, vector: np.ndarray):
        """Save a vector for a PDF (as bytes)."""
        pdf = PDFFile.objects.get(id=pdf_id)
        vector_bytes = vector.tobytes()
        PDFVector.objects.update_or_create(
            pdf=pdf,
            defaults={'vector': vector_bytes}
        )

    def get_all_pdf_vectors(self):
        """Return a list of dicts: [{'pdf_id': int, 'vector': np.ndarray}, ...]"""
        results = []
        for pv in PDFVector.objects.select_related('pdf').all():
            vector = np.frombuffer(pv.vector, dtype=np.float32)
            results.append({'pdf_id': pv.pdf.id, 'vector': vector})
        return results
