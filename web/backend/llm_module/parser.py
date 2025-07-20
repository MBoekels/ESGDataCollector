import hashlib
from typing import List, Dict, Any, Optional
import pdfplumber
import pandas as pd

class PDFParser:
    """
    PDFParser extracts and chunks content from PDF files, supporting two main formats:
    - Text paragraphs (with optional overlapping chunking)
    - Table columns (e.g. financial data per year)

    Optionally, chunks can be embedded using a sentence-transformers provider.

    Metadata fields in each chunk include:
        - chunk_id (str): Unique identifier for the chunk.
        - source_pdf_id (int): ID of the source PDF, if provided.
        - chunk_type (str): 'text' or 'table_column'.
        - page_nums (List[int]): Pages covered by the chunk.
        - bbox_list (List[Tuple]): Bounding boxes of content regions.
        - para_indices (List[int]): Paragraph indices (only for text chunks).
        - text (str): The main content of the chunk.
        - context_before (str): Preceding paragraph (for context).
        - context_after (str): Following paragraph (for context).
        - year (int or None): Year (for table columns only).
        - row_labels (List[str]): Row labels (for tables only).
        - values (List[str]): Corresponding row values (for tables only).
        - embedding (List[float], optional): Vector embedding if enabled.
    """

    def __init__(self, overlap: int = 1, embedding_provider=None):
        """
        Initialize the PDFParser.

        Args:
            overlap (int): Number of overlapping paragraphs between chunks.
            embedding_provider: Optional embedding provider with `.encode(List[str]) -> List[List[float]]`.
        """
        self.overlap = overlap
        self.embedding_provider = embedding_provider

    def _generate_chunk_id(self, pdf_id: int, chunk_type: str, page_num: int, text: str) -> str:
        """
        Generate a unique chunk ID using hash of the text content.

        Args:
            pdf_id (int): Source PDF ID.
            chunk_type (str): 'text' or 'table_column'.
            page_num (int): Page number.
            text (str): Content of the chunk.

        Returns:
            str: A reproducible unique chunk identifier.
        """
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return f"{pdf_id}-{chunk_type}-{page_num}-{text_hash}"

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts raw text paragraphs from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[Dict[str, Any]]: A list of paragraph dictionaries, each containing:
                - text (str)
                - page_num (int)
                - bbox (Tuple)
                - para_idx (int)
        """
        paragraphs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if not text:
                    continue
                raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for idx, para in enumerate(raw_paragraphs):
                    bbox = page.bbox
                    paragraphs.append({
                        'text': para,
                        'page_num': page_num,
                        'bbox': bbox,
                        'para_idx': idx
                    })
        return paragraphs

    def chunk_paragraphs(
        self,
        paragraphs: List[Dict[str, Any]],
        pdf_id: Optional[int] = None,
        chunk_size: int = 3,
        embed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Creates overlapping chunks from paragraphs for use in LLMs or embeddings.

        Args:
            paragraphs (List[Dict[str, Any]]): List of paragraph metadata (from `parse_pdf`).
            pdf_id (int, optional): Optional source PDF ID for traceability.
            chunk_size (int): Number of paragraphs per chunk.
            embed (bool): Whether to generate and attach embeddings.

        Returns:
            List[Dict[str, Any]]: Chunk dictionaries with the following fields:
                - chunk_id (str)
                - source_pdf_id (int or None)
                - chunk_type ('text')
                - page_nums (List[int])
                - bbox_list (List[Tuple])
                - para_indices (List[int])
                - text (str)
                - context_before (None)
                - context_after (None)
                - year (None)
                - row_labels ([])
                - values ([])
                - embedding (List[float], optional)
        """
        chunks = []
        n = len(paragraphs)
        for i in range(0, n, chunk_size - self.overlap):
            chunk_paras = paragraphs[i:i+chunk_size]
            if not chunk_paras:
                continue

            chunk_text = '\n\n'.join([p['text'] for p in chunk_paras])
            page_nums = [p['page_num'] for p in chunk_paras]
            bbox_list = [p['bbox'] for p in chunk_paras]
            para_indices = [p['para_idx'] for p in chunk_paras]

            chunk_meta = {
                'chunk_id': self._generate_chunk_id(pdf_id or 0, 'text', page_nums[0], chunk_text),
                'source_pdf_id': pdf_id,
                'chunk_type': 'text',
                'page_nums': page_nums,
                'bbox_list': bbox_list,
                'para_indices': para_indices,
                'text': chunk_text,
                'context_before': None,
                'context_after': None,
                'year': None,
                'row_labels': [],
                'values': []
            }

            chunks.append(chunk_meta)

        if embed and self.embedding_provider:
            texts = [c['text'] for c in chunks]
            embeddings = self.embedding_provider.encode(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk['embedding'] = emb

        return chunks

    def extract_table_chunks(
        self,
        pdf_path: str,
        paragraphs: List[Dict[str, Any]],
        pdf_id: Optional[int] = None,
        embed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extracts table data from PDFs and converts year-columns into independent chunks.

        Each chunk represents one year and includes context paragraphs before/after the table.

        Args:
            pdf_path (str): Path to the PDF file.
            paragraphs (List[Dict[str, Any]]): Parsed paragraphs (for context).
            pdf_id (int, optional): Optional PDF ID for traceability.
            embed (bool): Whether to attach embeddings to chunks.

        Returns:
            List[Dict[str, Any]]: Table column chunks with the following fields:
                - chunk_id (str)
                - source_pdf_id (int or None)
                - chunk_type ('table_column')
                - page_nums (List[int])
                - bbox_list (List[Tuple])
                - para_indices ([] for tables)
                - text (str): Extracted year-column and values
                - context_before (str)
                - context_after (str)
                - year (int)
                - row_labels (List[str])
                - values (List[str])
                - embedding (List[float], optional)
        """
        table_chunks = []
        paragraphs_by_page = {}
        for para in paragraphs:
            page = para['page_num']
            paragraphs_by_page.setdefault(page, []).append(para)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_tables = page.extract_tables()
                if not raw_tables:
                    continue

                all_paras = paragraphs_by_page.get(page_num, [])
                context_before = all_paras[-1]['text'] if all_paras else ""
                context_after = all_paras[0]['text'] if all_paras else ""

                for table in raw_tables:
                    if not table or len(table) < 2:
                        continue

                    df = pd.DataFrame(table[1:], columns=table[0])
                    if df.empty or df.shape[1] < 2:
                        continue

                    year_columns = [col for col in df.columns if str(col).isdigit() and 1900 < int(col) < 2100]
                    if not year_columns:
                        continue

                    for year in year_columns:
                        year_val = int(year)
                        row_labels = []
                        values = []
                        text_lines = [f"Year: {year_val}"]

                        for _, row in df.iterrows():
                            label = row[df.columns[0]]
                            value = row[year]
                            row_labels.append(label)
                            values.append(value)
                            text_lines.append(f"{label}: {value}")

                        chunk_text = '\n'.join(text_lines)
                        full_text = f"{context_before}\n\n{chunk_text}\n\n{context_after}"

                        chunk = {
                            'chunk_id': self._generate_chunk_id(pdf_id or 0, 'table_column', page_num, chunk_text),
                            'source_pdf_id': pdf_id,
                            'chunk_type': 'table_column',
                            'page_nums': [page_num],
                            'bbox_list': [page.bbox],
                            'para_indices': [],
                            'text': chunk_text.strip(),
                            'context_before': context_before,
                            'context_after': context_after,
                            'year': year_val,
                            'row_labels': row_labels,
                            'values': values
                        }

                        table_chunks.append(chunk)

        if embed and self.embedding_provider:
            texts = [c['text'] for c in table_chunks]
            embeddings = self.embedding_provider.encode(texts)
            for chunk, emb in zip(table_chunks, embeddings):
                chunk['embedding'] = emb

        return table_chunks
