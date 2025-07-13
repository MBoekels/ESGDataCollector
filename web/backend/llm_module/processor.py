"""
Handles LLM calls and prompt construction.
Should be provider-agnostic and support confidence scoring.
"""

from typing import Dict, Any



class LLMProcessor:
    """
    Handles LLM calls and prompt construction. Provider-agnostic.
    Supports RAG (Retrieval-Augmented Generation) with chunk similarity search.
    """
    def __init__(self, provider):
        self.provider = provider  # Should implement a .generate() method

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

    def rag_analyze(self, query: str, embedding_provider, chunk_store, top_k: int = 5, prompt_template: str = None, **kwargs) -> Dict[str, Any]:
        """
        RAG pipeline: retrieves top_k relevant chunks for the query, constructs a prompt, calls LLM, and returns result with references.
        - query: user question
        - embedding_provider: instance with .encode(texts) method
        - chunk_store: ChunkVectorStore instance
        - top_k: number of chunks to retrieve
        - prompt_template: optional string with {context} and {question} placeholders
        Returns: dict with summary, confidence, provider, references (chunk metadata)
        """
        # Step 1: Embed the query
        query_vec = embedding_provider.encode([query])[0]
        # Step 2: Retrieve top_k similar chunks
        top_chunks = chunk_store.search(query_vec, top_k=top_k)
        # Step 3: Build context from retrieved chunks
        context = "\n\n".join([c['text'] for c in top_chunks])
        # Step 4: Build prompt
        if prompt_template:
            prompt = prompt_template.format(context=context, question=query)
        else:
            prompt = f"Answer the following question using only the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        # Step 5: Call LLM
        result = self.provider.generate(prompt, **kwargs)
        # Step 6: Return result with references
        return {
            'summary': result['text'],
            'confidence': result.get('confidence', 0.5),
            'provider': self.provider.__class__.__name__,
            'references': top_chunks
        }
