"""
File: web/backend/llm_module/llm_provider.py

Role:
    This file defines provider classes that act as wrappers for various Language Model (LLM) and
    embedding model libraries. It standardizes the interface for interacting with different models,
    such as those from Hugging Face (`transformers`, `sentence-transformers`). This abstraction allows
    the rest of the application to be model-agnostic.

Interactions:
    - `processor.py`: `LLMProcessor` is initialized with a provider (e.g., `HuggingFaceLLMProvider`)
      to make calls to the underlying LLM.
    - `parser.py` & `evaluator.py`: These modules use `SentenceTransformersEmbeddingProvider` to generate
      vector embeddings for text chunks and queries.

Inputs:
    - `HuggingFaceLLMProvider.generate`: Expects a `prompt` string.
    - `SentenceTransformersEmbeddingProvider.encode`: Expects a list of `texts` (strings).
"""

from abc import ABC, abstractmethod
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np


# === Interface for LLM Providers ===
class LLMProviderInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> dict:
        """
        Generate a response from the model using the given prompt.
        Should return a dictionary with at least keys: 'text', 'confidence', 'provider'.
        """
        pass


# === Interface for Embedding Providers ===
class EmbeddingProviderInterface(ABC):
    @abstractmethod
    def encode(self, texts, **kwargs) -> np.ndarray:
        """
        Encode a list of texts into vector embeddings.
        Should return a NumPy array of shape (n_samples, embedding_dim).
        """
        pass


# === Concrete LLM Provider ===
class HuggingFaceLLMProvider(LLMProviderInterface):
    def __init__(self, model_name="deepset/roberta-base-squad2", task="question-answering", device=0, **kwargs):
        self.generator = pipeline(task, model=model_name, device=device, **kwargs)
        self.model_name = model_name

    def generate(self, prompt, **kwargs):
        result = self.generator(prompt, **kwargs)
        text = result[0].get('generated_text') or result[0].get('summary_text') or str(result[0])
        max_len = kwargs.get('max_new_tokens', 256)
        conf = min(len(text) / max_len, 1.0) if max_len else 1.0
        return {
            'text': text,
            'confidence': round(conf, 2),
            'provider': self.model_name
        }


# === Concrete Embedding Provider ===
class SentenceTransformersEmbeddingProvider(EmbeddingProviderInterface):
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu", **kwargs):
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, convert_to_numpy=True, **kwargs)


# === Dummy LLM Provider for testing ===
class DummyLLMProvider(LLMProviderInterface):
    def generate(self, prompt, **kwargs):
        return {
            'text': 'This is a dummy LLM response.',
            'confidence': 0.5,
            'provider': 'DummyLLMProvider'
        }
