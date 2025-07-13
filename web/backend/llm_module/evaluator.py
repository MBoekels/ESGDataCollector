"""
Main entry point for LLM-based evaluation logic.
Handles orchestration of PDF parsing, retrieval, LLM calls, and result formatting.
"""

from .parser import PDFParser
from .vector_store import PDFVectorStore, ChunkVectorStore
from .processor import LLMProcessor
from .schemas import build_result_schema
from .utils import timing

class LLMEvaluator:
    """
    Orchestrates the full evaluation pipeline: parsing, retrieval, LLM, and result formatting.
    """
    def __init__(self, db_interface, llm_provider, chunk_dim: int = 384):
        self.parser = PDFParser()
        self.pdf_store = PDFVectorStore(db_interface)
        self.chunk_store = ChunkVectorStore(chunk_dim)
        self.llm = LLMProcessor(llm_provider)

    def evaluate(self, pdf_path: str, query: str, query_id: int, company_id: int, user: str = None) -> dict:
        with timing("Full evaluation"):
            # 1. Parse PDF and chunk
            paragraphs = self.parser.parse_pdf(pdf_path)
            chunks = self.parser.chunk_paragraphs(paragraphs)
            # 2. (Assume embeddings are precomputed for chunks)
            # 3. Retrieve relevant chunks (placeholder: use all)
            relevant_chunks = chunks  # TODO: filter by similarity
            # 4. Construct prompt and call LLM
            prompt = f"Query: {query}\n\n" + "\n---\n".join([c['text'] for c in relevant_chunks])
            llm_result = self.llm.analyze(prompt)
            # 5. Build references (placeholder)
            references = [{
                'pdf_id': None,
                'page_num': c['page_nums'][0],
                'span': None,
                'snippet': c['text'][:100],
                'confidence': llm_result['confidence']
            } for c in relevant_chunks]
            # 6. Build result schema
            result = build_result_schema(
                query_id=query_id,
                company_id=company_id,
                references=references,
                per_query_data={'summary': llm_result['summary']},
                user=user,
                model_version=llm_result['provider']
            )
            return result