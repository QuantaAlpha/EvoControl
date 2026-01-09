from typing import Any

from .embeddings.openai import OpenAIEmbeddingModel
from .memory.base import MemoryBackend
from .memory.chroma import ChromaMemoryBackend
from .utils.config import GlobalMemoryConfig, load_config

EmbeddingModel = Any


class GlobalMemoryBank:
    def __init__(
        self, config_path: str | None = None, config: GlobalMemoryConfig | dict[str, Any] | None = None
    ) -> None:
        if config is None:
            if not config_path:
                raise ValueError("config_path is required when config is not provided")
            config = load_config(config_path)
        self.config: GlobalMemoryConfig = config
        self.memory_backend = self._init_memory_backend()
        self.embedding_model = self._init_embedding_model()

    def _init_memory_backend(self) -> MemoryBackend:
        backend_type = self.config.memory.backend
        if backend_type != "chroma":
            raise ValueError(f"Unsupported memory backend: {backend_type}")
        collection_name = self.config.memory.chroma.collection_name
        persist_path = self.config.memory.chroma.persist_path
        return ChromaMemoryBackend(collection_name=collection_name, persist_path=persist_path)

    def _init_embedding_model(self) -> EmbeddingModel:
        em_cfg = self.config.embedding_model
        provider = (em_cfg.provider or "openai").lower()
        if provider != "openai":
            raise ValueError(f"Unsupported embedding provider: {provider}")
        api_base = em_cfg.api_base or None
        api_key = em_cfg.api_key or None
        model = em_cfg.model or None
        if not api_base or not api_key or not model:
            raise ValueError("embedding_model requires api_base, api_key, and model")
        return OpenAIEmbeddingModel(
            api_base=api_base, api_key=api_key, model=model, request_timeout=em_cfg.request_timeout
        )

    def add_experience(self, experience: str, metadata: dict[str, Any]) -> None:
        embedding = self.embedding_model.encode(experience)
        item = {"embedding": embedding, "metadata": dict(metadata or {}), "document": experience}
        self.memory_backend.add([item])

    def retrieve_memories(self, query: str, k: int = 1) -> dict[str, list]:
        query_embedding = self.embedding_model.encode(query)
        return self.memory_backend.query(query_embedding, k)
