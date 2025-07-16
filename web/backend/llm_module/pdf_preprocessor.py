"""
Handles the pre-processing and embedding of PDF documents.
This includes parsing, chunking, generating embeddings, and storing
the resulting vector index for later use in RAG pipelines.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Any

from celery import shared_task

from django.conf import settings

from .parser import PDFParser
from .vector_store import ChunkVectorStore
from .llm_provider import SentenceTransformersEmbeddingProvider
from api.models import PDFFile
from .pdf_preprocessor import PDFPreprocessor # Assuming you have this class

class PDFPreprocessor: #TODO: remove this class after refactoring 
    def __init__(self, embedding_provider=None):
        self.embedding_provider = embedding_provider or SentenceTransformersEmbeddingProvider()
        self.parser = PDFParser(embedding_provider=self.embedding_provider)
        # Ensure the directory for indexes exists
        self.index_dir = os.path.join(settings.MEDIA_ROOT, 'vector_indexes')
        os.makedirs(self.index_dir, exist_ok=True)

    def process_and_embed_pdf(self, pdf_file: PDFFile): #TODO: refactor to module level functions
        """
        Processes a single PDF file: parses, chunks, embeds, and saves the chunk vector index.
        """
        print(f"Starting pre-processing for PDF: {pdf_file.file.name}")
        pdf_path = pdf_file.file.path

        paragraphs = self.parser.parse_pdf(pdf_path)
        if not paragraphs:
            print(f"No text could be extracted from {pdf_path}. Skipping.")
            return

        chunks_with_embeddings = self.parser.chunk_paragraphs(paragraphs, embed=True)
        vectors = np.array([c.pop('embedding') for c in chunks_with_embeddings]).astype('float32')
        metas = chunks_with_embeddings

        chunk_dim = self.embedding_provider.model.get_sentence_embedding_dimension()
        chunk_store = ChunkVectorStore(dim=chunk_dim)
        chunk_store.add_chunk_vectors(vectors, metas)

        index_filename = f"{pdf_file.file_hash}.faiss"
        meta_filename = f"{pdf_file.file_hash}.meta"
        index_filepath = os.path.join(self.index_dir, index_filename)
        meta_filepath = os.path.join(self.index_dir, meta_filename)
        chunk_store.save_index(index_filepath, meta_filepath)

        relative_index_path = os.path.join('vector_indexes', index_filename)
        pdf_file.chunk_vector_index_path = relative_index_path
        pdf_file.save(update_fields=['chunk_vector_index_path'])
        print(f"Updated PDFFile {pdf_file.id} with index path: {relative_index_path}")

@shared_task(bind=True, retry_backoff=True, retry_kwargs={'max_retries': 5})
def process_and_embed_pdf_task(self, pdf_id: int):
    """
    Celery task to process and embed a PDF file.
    """
    try:
        pdf_file = PDFFile.objects.get(id=pdf_id)
        preprocessor = PDFPreprocessor()
        preprocessor.process_and_embed_pdf(pdf_file)
        pdf_file.processing_status = 'success'
    except PDFFile.DoesNotExist:
        print(f"PDF file with id {pdf_id} not found.")
        pdf_file.processing_status = 'failed'
    except Exception as e:
        print(f"Error processing PDF {pdf_id}: {e}")
        pdf_file.processing_status = 'failed'
        # `countdown` is in seconds.  Retry after 5 minutes, 10 minutes, etc.
        self.retry(exc=e, countdown=60 * 5 * self.request.retries)
    finally:
        pdf_file.save(update_fields=['processing_status'])

