"""
Handles vector storage for both PDF-level (Postgres) and chunk-level (FAISS) embeddings.
Should support similarity search and efficient retrieval.
"""

import os
import pickle
from typing import List, Dict, Any
import numpy as np

class PDFVectorStore:
    """
    Stores and retrieves PDF-level embeddings in a persistent store (e.g., Postgres via Django ORM).
    """
    def __init__(self, db_interface):
        self.db = db_interface  # Expects an interface to Django models

    def add_pdf_vector(self, pdf_id: int, vector: np.ndarray):
        """Store a vector for a PDF (persistently)."""
        self.db.save_pdf_vector(pdf_id, vector)

    def get_pdf_vectors(self) -> List[Dict[str, Any]]:
        """Retrieve all stored PDF vectors."""
        return self.db.get_all_pdf_vectors()

    def find_similar_pdfs(self, query_vector: np.ndarray, top_k: int = 5) -> List[int]:
        """Return IDs of top_k most similar PDFs."""
        pdf_vectors = self.get_pdf_vectors()
        sims = [(v['pdf_id'], self.cosine_similarity(query_vector, v['vector'])) for v in pdf_vectors]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [pdf_id for pdf_id, _ in sims[:top_k]]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ChunkVectorStore:
    """
    Stores and retrieves chunk-level embeddings using FAISS for fast similarity search.
    """
    def __init__(self, dim: int, index_path: str = None, meta_path: str = None):
        import faiss
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if index_path and meta_path and os.path.exists(index_path) and os.path.exists(meta_path):
            self.load_index(index_path, meta_path)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.chunk_meta = []  # List of dicts with metadata for each chunk

    def add_chunk_vectors(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        self.index.add(vectors)
        self.chunk_meta.extend(metas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        D, I = self.index.search(query_vector.reshape(1, -1), top_k)
        return [self.chunk_meta[i] for i in I[0] if i < len(self.chunk_meta)]

    def save_index(self, index_path: str, meta_path: str):
        """Saves the FAISS index and metadata to disk."""
        import faiss
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.chunk_meta, f)
        self.index_path = index_path
        self.meta_path = meta_path

    def load_index(self, index_path: str, meta_path: str):
        """Loads the FAISS index and metadata from disk."""
        import faiss
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.chunk_meta = pickle.load(f)
        self.index_path = index_path
        self.meta_path = meta_path
