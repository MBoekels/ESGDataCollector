"""
File: web/backend/llm_module/processor.py

Role:
    This file contains the `LLMProcessor` class, which implements the core logic for the
    Retrieval-Augmented Generation (RAG) pipeline. It is responsible for taking a user query,
    retrieving relevant context from a vector store, constructing a detailed prompt with that
    context, and calling an LLM to generate a response.

Interactions:
    - `llm_provider.py`: Is initialized with an LLM provider (e.g., `HuggingFaceLLMProvider`)
      and uses its `.generate()` method to get a response from the model. It also uses an
      embedding provider to vectorize the user query.
    - `vector_store.py`: In the `rag_analyze` method, it uses a `ChunkVectorStore` instance to
      perform a similarity search and retrieve the most relevant text chunks for the query.
    - `evaluator.py`: The `LLMEvaluator` uses this processor to perform the main analysis step.
"""

import numpy as np
from .vector_store import DocumentVectorStore, ChunkVectorStore
import re
from typing import List, Dict, Any, Optional
from .llm_provider import LLMProviderInterface, EmbeddingProviderInterface

DOCUMENT_SIMILARITY_THRESHOLD = 0.7

class LLMProcessor:
    def __init__(self, provider: LLMProviderInterface = None, embedding_provider: EmbeddingProviderInterface = None):
        self.provider = provider  # e.g. HuggingFaceLLMProvider instance
        self.embedding_provider = embedding_provider  # e.g. SentenceTransformersEmbeddingProvider

    def analyze(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Calls the LLM provider with the prompt and returns the result and confidence.
        """
        result = self.provider.generate(prompt, **kwargs)
        return {
            'summary': result['text'],
            'confidence': result.get('confidence', 0.5),  # Normalized 0-1
            'provider': self.provider.__class__.__name__
        }

    def extract_report_year_from_text_chunk(self, text_chunk: str) -> Optional[int]:
        prompt = f"""
        Extract the reporting year mentioned in the following text paragraph.
        If there is no explicit year, return 'None'.

        Text:
        \"\"\"{text_chunk}\"\"\"
        """
        result = self.analyze(prompt)
        year_str = result.get('summary', '').strip()
        match = re.search(r"\b(19|20)\d{2}\b", year_str)
        if match:
            return int(match.group(0))
        return None

    def rag_analyze(
        self,
        company_id: int,
        query_id: int,
        query_text: str,
        pdf_files: List[Any],  # List of PDFFile objects with metadata
        top_k: int = 5,
        filter_by_document_level_index: bool = False,
        extended_search: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform RAG analysis for a company and query.

        - Filter pdf_files by company_id
        - Optionally filter on document level using document vector index similarity
        - For each relevant PDF, search chunk index for top-k relevant chunks
        - Filter chunk types based on extended_search flag
        - Extract report year from text chunks with fallback to pdf.report_year
        - Return list of data points (one dict per chunk)
        """

        # Step 1: Filter pdf_files by company_id
        relevant_pdfs = [pdf for pdf in pdf_files if pdf.company_id == company_id]
        if not relevant_pdfs:
            return []

        # Step 2: If filtering by document level index, narrow down relevant_pdfs
        if filter_by_document_level_index:
            query_vector = self.embedding_provider.embed([query_text])[0]
            filtered_pdfs = []
            for pdf_file in relevant_pdfs:
                try:
                    doc_store = self._load_document_store(pdf_file.document_vector_index_path)
                except Exception as e:
                    print(f"Failed to load document vector index for PDF {pdf_file.id}: {e}")
                    continue

                doc_scores = doc_store.search(query_vector, top_k=1)
                if doc_scores:
                    top_score = doc_scores[0].get('score', 0)
                    if top_score >= DOCUMENT_SIMILARITY_THRESHOLD:
                        filtered_pdfs.append(pdf_file) # Only keep PDFs with high similarity
            relevant_pdfs = filtered_pdfs

        results = []

        # Embed the query vector once (if not done above)
        if not filter_by_document_level_index:
            query_vector = self.embedding_provider.embed([query_text])[0]

        # Step 3: For each relevant PDF, find top-k relevant chunks
        for pdf_file in relevant_pdfs:
            try:
                chunk_store = self._load_chunk_store(pdf_file.chunk_vector_index_path)
            except Exception as e:
                print(f"Failed to load chunk vector index for PDF {pdf_file.id}: {e}")
                continue

            top_chunks = chunk_store.search(query_vector, top_k=top_k)

            for chunk_meta, cosine_sim in top_chunks:
                chunk_type = chunk_meta.get("chunk_type", "unknown")

                # Step 4: Filter by chunk type if extended_search is False
                if not extended_search and chunk_type != "table_column":
                    continue

                answer = chunk_meta.get("text", "")
                confidence = cosine_sim

                # Step 5: Extract reference year
                if chunk_type == "text":
                    year = self.extract_report_year_from_text_chunk(answer)
                    if year is None:
                        year = pdf_file.report_year
                elif chunk_type == "table_column":
                    year = chunk_meta.get("report_year") or pdf_file.report_year
                else:
                    year = pdf_file.report_year

                    data_point = {
                    "company_id": company_id,
                    "pdf_id": pdf_file.id,
                    "query_id": query_id,
                    "report_year": year,
                    "source": getattr(pdf_file, 'source_url', None) or getattr(pdf_file, 'file_name', None),
                    "chunk_id": chunk_meta.get("chunk_id"),
                    "chunk_type": chunk_type,
                    "cosine_similarity": cosine_sim,
                    "answer": answer,
                    "confidence": confidence,
                    "provider": getattr(self.provider, 'name', 'unknown'),

                    # Unified reference metadata for traceability
                    "references": {
                        "chunk_id": chunk_meta.get("chunk_id"),
                        "source_pdf_id": chunk_meta.get("source_pdf_id"),
                        "chunk_type": chunk_type,
                        "page_nums": chunk_meta.get("page_nums", []),
                        "para_indices": chunk_meta.get("para_indices", []),
                        "bbox_list": chunk_meta.get("bbox_list", []),
                        "year": chunk_meta.get("year"),
                        "row_labels": chunk_meta.get("row_labels", []),
                        "values": chunk_meta.get("values", []),
                        "context_before": chunk_meta.get("context_before"),
                        "context_after": chunk_meta.get("context_after"),
                    }
                }

                results.append(data_point)

        return results

    def _load_chunk_store(self, index_path: str):
        chunk_dim = self.embedding_provider.model.get_sentence_embedding_dimension()
        chunk_store = ChunkVectorStore(dim=chunk_dim)
        chunk_store.load_index(index_path)
        return chunk_store

    def _load_document_store(self, index_path: str):
        chunk_dim = self.embedding_provider.model.get_sentence_embedding_dimension()
        document_store = DocumentVectorStore(dim=chunk_dim)
        document_store.load_index(index_path)
        return document_store
