"""
PDF parsing and chunking utilities.
Should preserve layout and support paragraph-based chunking with overlap.
"""

import pdfplumber
from typing import List, Dict, Any

from api.llm_provider import SentenceTransformersEmbeddingProvider

class PDFParser:
    """
    Parses PDF files, extracts text, and chunks by paragraph with overlap.
    Preserves layout information for source tracking.
    Optionally supports embedding chunks using a sentence-transformers provider.
    """
    def __init__(self, overlap: int = 1, embedding_model: str = None, embedding_device: str = "cpu"):
        self.overlap = overlap
        self.embedding_provider = None
        if embedding_model:
            self.embedding_provider = SentenceTransformersEmbeddingProvider(model_name=embedding_model, device=embedding_device)

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parses the PDF and returns a list of paragraphs with metadata.
        Each item contains: text, page_num, bbox (bounding box), and paragraph index.
        """
        paragraphs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if not text:
                    continue
                # Split by double newlines (paragraphs)
                raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for idx, para in enumerate(raw_paragraphs):
                    # Get bounding box for the paragraph (approximate: use page bbox)
                    bbox = page.bbox
                    paragraphs.append({
                        'text': para,
                        'page_num': page_num,
                        'bbox': bbox,
                        'para_idx': idx
                    })
        return paragraphs

    def chunk_paragraphs(self, paragraphs: List[Dict[str, Any]], chunk_size: int = 3, embed: bool = False) -> List[Dict[str, Any]]:
        """
        Chunks paragraphs with overlap for embedding/LLM input.
        If embed=True and embedding_provider is set, adds 'embedding' to each chunk dict.
        Returns a list of dicts with combined text and metadata (and optionally embedding).
        """
        chunks = []
        n = len(paragraphs)
        for i in range(0, n, chunk_size - self.overlap):
            chunk_paras = paragraphs[i:i+chunk_size]
            if not chunk_paras:
                continue
            chunk_text = '\n\n'.join([p['text'] for p in chunk_paras])
            chunk_meta = {
                'text': chunk_text,
                'page_nums': [p['page_num'] for p in chunk_paras],
                'bbox_list': [p['bbox'] for p in chunk_paras],
                'para_indices': [p['para_idx'] for p in chunk_paras]
            }
            chunks.append(chunk_meta)
        # If embedding requested and provider is set, compute embeddings for all chunks
        if embed and self.embedding_provider:
            texts = [c['text'] for c in chunks]
            embeddings = self.embedding_provider.encode(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk['embedding'] = emb
        return chunks