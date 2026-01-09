"""OpenAI embedding model wrapper."""

from collections.abc import Iterable, Sequence

from openai import OpenAI


class OpenAIEmbeddingModel:
    """Embedding model that uses OpenAI Embeddings API."""

    def __init__(self, api_base: str, api_key: str, model: str, request_timeout: float | None = None) -> None:
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=request_timeout or 600.0)
        self.model = model

    def encode(self, text: str) -> list[float]:
        """Encodes a single string into an embedding vector."""
        resp = self.client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def encode_batch(self, texts: Sequence[str] | Iterable[str]) -> list[list[float]]:
        """Encodes a batch of strings into embedding vectors."""
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [list(d.embedding) for d in resp.data]
