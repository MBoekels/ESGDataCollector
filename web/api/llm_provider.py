"""
Provider-agnostic LLM provider interface for LLMProcessor.
HuggingFace implementation for local or hub models.
"""


from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np


class HuggingFaceLLMProvider:
    def __init__(self, model_name="openai-community/gpt2", task="text-generation", device=0, **kwargs):
        """
        model_name: HuggingFace model repo or local path
        task: e.g., 'text-generation', 'summarization', etc.
        device: -1 for CPU, 0 for first GPU
        kwargs: passed to pipeline
        """
        self.generator = pipeline(task, model=model_name, device=device, **kwargs)
        self.model_name = model_name

    def generate(self, prompt, **kwargs):
        result = self.generator(prompt, **kwargs)
        text = result[0].get('generated_text') or result[0].get('summary_text') or str(result[0])
        # Custom confidence: use normalized length of output as a proxy (longer = more confident)
        max_len = kwargs.get('max_new_tokens', 256)
        conf = min(len(text) / max_len, 1.0) if max_len else 1.0
        return {
            'text': text,
            'confidence': round(conf, 2),
            'provider': self.model_name
        }


class SentenceTransformersEmbeddingProvider:
    """
    Embedding provider using sentence-transformers for PDF/chunk vectorization.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu", **kwargs):
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        # Accepts a string or list of strings
        return self.model.encode(texts, convert_to_numpy=True, **kwargs)

# Dummy provider for testing remains available
class DummyLLMProvider:
    def generate(self, prompt, **kwargs):
        return {
            'text': 'This is a dummy LLM response.',
            'confidence': 0.5,
            'provider': 'DummyLLMProvider'
        }
