"""
Main entry point for LLM-based evaluation logic.
Handles orchestration of PDF parsing, retrieval, LLM calls, and result formatting.
"""

import numpy as np

from .parser import PDFParser
from .vector_store import PDFVectorStore, ChunkVectorStore
from .processor import LLMProcessor
from .schemas import build_result_schema
from .llm_provider import SentenceTransformersEmbeddingProvider
from .utils import timing

class LLMEvaluator:
    """
    Orchestrates the full evaluation pipeline: parsing, retrieval, LLM, and result formatting.
    """
    def __init__(self, db_interface, llm_provider, chunk_dim: int = 384):
        # Assuming you have an embedding provider for the query
        self.embedding_provider = SentenceTransformersEmbeddingProvider()
        self.parser = PDFParser(embedding_provider=self.embedding_provider)
        self.pdf_store = PDFVectorStore(db_interface)
        self.chunk_store = ChunkVectorStore(chunk_dim)
        self.llm = LLMProcessor(llm_provider)

    def evaluate(self, pdf_path: str, query: str, query_id: int, company_id: int, user: str = None) -> dict:
        with timing("Full evaluation"):
            # 1. Parse PDF and create embedded chunks
            paragraphs = self.parser.parse_pdf(pdf_path)
            # The parser can now embed chunks directly
            chunks_with_embeddings = self.parser.chunk_paragraphs(paragraphs, embed=True)

            # 2. Add chunk vectors to the in-memory FAISS store
            vectors = np.array([c['embedding'] for c in chunks_with_embeddings])
            metas = [{'text': c['text'], 'page_nums': c['page_nums']} for c in chunks_with_embeddings]
            self.chunk_store.add_chunk_vectors(vectors, metas)

            # 3. Use the RAG pipeline in the processor
            llm_result = self.llm.rag_analyze(query, self.embedding_provider, self.chunk_store, top_k=5)

            # 4. Build references from the retrieved chunks
            references = [{
                'pdf_id': None,
                'page_num': c['page_nums'][0] if c.get('page_nums') else 'N/A',
                'span': None,
                'snippet': c['text'][:200],
                'confidence': llm_result['confidence']
            } for c in llm_result.get('references', [])]
            # 5. Build result schema
            result = build_result_schema(
                query_id=query_id,
                company_id=company_id,
                references=references,
                per_query_data={'summary': llm_result['summary']},
                user=user,
                model_version=llm_result['provider']
            )
            return result