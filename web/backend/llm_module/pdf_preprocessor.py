import os
import pickle
import numpy as np
from typing import Dict, List, Any
from pypdf import PdfReader
from datetime import datetime

from celery import shared_task

from django.conf import settings

from .parser import PDFParser
from .vector_store import ChunkVectorStore, DocumentVectorStore
from .llm_provider import SentenceTransformersEmbeddingProvider, HuggingFaceLLMProvider, LLMProviderInterface, EmbeddingProviderInterface
from .processor import LLMProcessor
from .utils import get_api_client

class PDFPreprocessor:
    def __init__(self, embedding_provider: EmbeddingProviderInterface = None, llm_provider: LLMProviderInterface = None):
        self.embedding_provider = embedding_provider or SentenceTransformersEmbeddingProvider()
        self.llm_provider = llm_provider or HuggingFaceLLMProvider()
        self.llm_processor = LLMProcessor(provider=self.llm_provider)
        self.parser = PDFParser(embedding_provider=self.embedding_provider)
        self.index_dir = os.path.join(settings.MEDIA_ROOT, 'vector_indexes')
        os.makedirs(self.index_dir, exist_ok=True)

    def process_and_embed_pdf(self, pdf_id: int, pdf_path: str):
        print(f"Starting pre-processing for PDF: {pdf_path}")

        paragraphs = self.parser.parse_pdf(pdf_path)
        if not paragraphs:
            print(f"No text could be extracted from {pdf_path}. Skipping.")
            return

        text_chunks = self.parser.chunk_paragraphs(paragraphs, embed=True)
        table_chunks = self.parser.extract_table_chunks(pdf_path, paragraphs, pdf_id=pdf_id, embed=True)

        all_chunks = text_chunks + table_chunks
        vectors = np.array([c.pop('embedding') for c in all_chunks]).astype('float32')
        metas = all_chunks

        chunk_dim = self.embedding_provider.model.get_sentence_embedding_dimension()

        # --- Chunk-level FAISS index ---
        chunk_store = ChunkVectorStore(dim=chunk_dim)
        chunk_store.add_chunk_vectors(vectors, metas)

        api_client = get_api_client()
        hash_response = api_client.get(f'/api/pdffiles/{pdf_id}/file_hash/')
        file_hash = hash_response.json().get('file_hash') if hash_response.status_code == 200 else None

        if not file_hash:
            raise ValueError(f"Could not retrieve file hash for PDFFile {pdf_id}")

        index_filename = f"{file_hash}.faiss"
        meta_filename = f"{file_hash}.meta"
        index_filepath = os.path.join(self.index_dir, index_filename)
        meta_filepath = os.path.join(self.index_dir, meta_filename)

        chunk_store.save_index(index_filepath, meta_filepath)

        relative_chunk_index_path = os.path.join('vector_indexes', index_filename)

        # --- Document-level FAISS index ---
        document_embedding = np.mean(vectors, axis=0, keepdims=True).astype('float32')
        document_store = DocumentVectorStore(dim=chunk_dim)
        document_store.add_document_vector(document_embedding, {'pdf_id': pdf_id})

        doc_index_filename = f"{file_hash}.doc.faiss"
        doc_meta_filename = f"{file_hash}.doc.meta"
        doc_index_filepath = os.path.join(self.index_dir, doc_index_filename)
        doc_meta_filepath = os.path.join(self.index_dir, doc_meta_filename)

        document_store.save_index(doc_index_filepath, doc_meta_filepath)

        relative_doc_index_path = os.path.join('vector_indexes', doc_index_filename)

        report_year = infer_report_year(pdf_path, self.llm_processor)

        patch_response = api_client.patch(
            f'/api/pdffiles/{pdf_id}/',
            {
                'chunk_vector_index_path': relative_chunk_index_path,
                'document_vector_index_path': relative_doc_index_path,
                'report_year': report_year
            }
        )

        if patch_response.status_code == 200:
            print(f"Updated PDFFile {pdf_id} with index paths.")
        else:
            print(f"Failed to update PDFFile {pdf_id}: {patch_response.status_code} - {patch_response.text}")


@shared_task(bind=True, retry_backoff=True, retry_kwargs={'max_retries': 5})
def process_and_embed_pdf_task(self, pdf_id: int, pdf_path: str):
    try:
        preprocessor = PDFPreprocessor()
        preprocessor.process_and_embed_pdf(pdf_id, pdf_path)

        api_client = get_api_client()
        response = api_client.patch(
            f'/api/pdffiles/{pdf_id}/',
            {'processing_status': 'success'}
        )
        if response.status_code != 200:
            print(f"Warning: Failed to update processing_status for {pdf_id} to 'success'")
    except Exception as e:
        print(f"Error processing PDF {pdf_id}: {e}")
        try:
            api_client = get_api_client()
            api_client.patch(
                f'/api/pdffiles/{pdf_id}/',
                {'processing_status': 'failed'}
            )
        except Exception as api_err:
            print(f"Failed to update failure status for PDF {pdf_id}: {api_err}")
        self.retry(exc=e, countdown=60 * 5 * self.request.retries)


def get_pdf_creation_year(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        info = reader.metadata
        raw_date = info.get('/CreationDate') or info.get('/ModDate')

        if raw_date:
            year = int(raw_date[2:6])
            return year
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    return None


def infer_report_year(pdf_path, llm_processor: LLMProcessor):
    year = get_pdf_creation_year(pdf_path)
    if year:
        return year

    reader = PdfReader(pdf_path)
    first_page_text = reader.pages[0].extract_text()

    prompt = f"""The following is the beginning of a sustainability report:

{first_page_text}

Which year does this report most likely correspond to? Return only the year as a 4-digit number."""

    result = llm_processor.analyze(prompt)
    try:
        parsed = int(result['summary'][:4])
        return parsed
    except:
        return None
