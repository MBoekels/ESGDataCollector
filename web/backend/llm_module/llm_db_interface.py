"""
Database interface for PDF vector storage and retrieval.
Abstracts away Django ORM details for use in LLM module.
"""

import numpy as np
import hashlib
from ...api.models import PDFFile, PDFVector
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.conf import settings

class LLMDBInterface:
    def save_pdf_vector(self, pdf_id: int, vector: np.ndarray):
        """Save a vector for a PDF (as bytes), with workspace trust and hash collision handling."""
        if not settings.configured:
            raise IntegrityError("Settings must be configured to use workspace trust.")

        if not settings.configured or settings.WS_TRUST_ENABLED:
            try:
                pdf = PDFFile.objects.get(id=pdf_id)
                vector_bytes = vector.tobytes()

                # Check for hash collisions *before* saving
                existing_vector = PDFVector.objects.filter(pdf=pdf).first()
                if existing_vector:
                    current_hash = hashlib.sha256(vector_bytes).hexdigest()
                    existing_hash = hashlib.sha256(existing_vector.vector).hexdigest()

                    if current_hash != existing_hash:
                        print(f"Hash collision detected for PDF ID {pdf_id}. Verifying data...")
                        if not np.array_equal(np.frombuffer(existing_vector.vector, dtype=np.float32), vector):
                            raise IntegrityError(f"Hash collision detected, and data is different for PDF ID {pdf_id}. Aborting save.")
                        else:
                            print("Hashes are different, but data is the same. Ignoring collision.")
                            return  # Data is the same, ignore the collision

                PDFVector.objects.update_or_create(pdf=pdf, defaults={'vector': vector_bytes})

            except PDFFile.DoesNotExist:
                raise ValidationError(f"PDF with id {pdf_id} does not exist.")
            except IntegrityError as e:
                raise IntegrityError(f"Integrity error during PDF vector save: {e}")
            except Exception as e:
                raise Exception(f"Unexpected error during PDF vector save: {e}")
        else:
            raise PermissionError("Saving PDF vectors requires workspace trust.")

    def get_all_pdf_vectors(self):
        """Return a list of dicts: [{'pdf_id': int, 'vector': np.ndarray}, ...]"""
        results = []
        for pv in PDFVector.objects.select_related('pdf').all():
            vector = np.frombuffer(pv.vector, dtype=np.float32)
            results.append({'pdf_id': pv.pdf.id, 'vector': vector})
        return results
