"""
File: web/backend/llm_module/vector_store.py

Role:
    This file is responsible for managing the storage and retrieval of vector embeddings. It provides two
    distinct classes for this purpose:
    1.  `PDFVectorStore`: Manages PDF-level embeddings, intended for persistent storage in a database
        (e.g., PostgreSQL via Django ORM). This is used for coarse-grained retrieval.
    2.  `ChunkVectorStore`: Manages chunk-level embeddings for a single document, using an in-memory
        FAISS index for extremely fast similarity searches. This is used for the fine-grained retrieval
        step in the RAG pipeline. It can save/load its index to/from disk.

Interactions:
    - `processor.py`: The `LLMProcessor` uses an instance of `ChunkVectorStore` to find document chunks
      that are semantically similar to a user's query.
    - `evaluator.py` & `pdf_preprocessor.py`: These modules create and populate `ChunkVectorStore` instances
      with chunk embeddings. The `pdf_preprocessor` also saves the index to disk.
"""

import os
import pickle
import faiss

from typing import List, Dict, Any
import numpy as np

class ChunkVectorStore:
    """
    Stores and retrieves chunk-level embeddings using FAISS for fast similarity search.
    """
    def __init__(self, dim: int, index_path: str = None, meta_path: str = None):
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


class DocumentVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity after normalization)
        self.metas = []

    def add_document_vector(self, vector: np.ndarray, meta: dict):
        """
        Fügt einen Dokument-Vektor mit Metadaten hinzu.
        vector: numpy array shape (dim,)
        meta: dict mit Metainformationen
        """
        assert vector.shape == (self.dim,), f"Vector dimension mismatch: expected {self.dim}, got {vector.shape}"
        # Normalisierung (wichtig für inner product als cosine similarity)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self.index.add(np.expand_dims(vector.astype('float32'), axis=0))
        self.metas.append(meta)

    def save_index(self, index_filepath: str, meta_filepath: str):
        faiss.write_index(self.index, index_filepath)
        with open(meta_filepath, 'wb') as f:
            pickle.dump(self.metas, f)

    def load_index(self, index_filepath: str, meta_filepath: str):
        if os.path.exists(index_filepath) and os.path.exists(meta_filepath):
            self.index = faiss.read_index(index_filepath)
            with open(meta_filepath, 'rb') as f:
                self.metas = pickle.load(f)
        else:
            raise FileNotFoundError("Index or meta file not found.")

    def search(self, query_vector: np.ndarray, top_k=5):
        """
        Suche ähnlichste Dokument-Vektoren.
        query_vector: numpy array shape (dim,)
        return: list of (meta, score)
        """
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        D, I = self.index.search(np.expand_dims(query_vector.astype('float32'), axis=0), top_k)
        results = []
        for i, dist in zip(I[0], D[0]):
            if i < len(self.metas):
                results.append((self.metas[i], dist))
        return results
