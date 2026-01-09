from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class OpenAIEmbeddingConfig:
    provider: str = "openai"
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    request_timeout: float | None = None


@dataclass
class ChromaBackendConfig:
    collection_name: str = "global_memory"
    persist_path: str | None = None


@dataclass
class MemoryConfig:
    backend: str = "chroma"
    chroma: ChromaBackendConfig = field(default_factory=ChromaBackendConfig)


@dataclass
class GlobalMemoryConfig:
    enabled: bool = True
    embedding_model: OpenAIEmbeddingConfig = field(default_factory=OpenAIEmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def _to_openai_embedding_config(d: dict[str, Any] | None) -> OpenAIEmbeddingConfig:
    d = d or {}
    return OpenAIEmbeddingConfig(
        provider=str(d.get("provider", "openai")),
        api_base=d.get("api_base") or d.get("base_url"),
        api_key=d.get("api_key"),
        model=d.get("model"),
        request_timeout=d.get("request_timeout"),
    )


def _to_memory_config(d: dict[str, Any] | None) -> MemoryConfig:
    d = d or {}
    backend = str(d.get("backend", "chroma"))
    chroma_dict = d.get("chroma") or {}
    chroma = ChromaBackendConfig(
        collection_name=str(chroma_dict.get("collection_name", "global_memory")),
        persist_path=chroma_dict.get("persist_path"),
    )
    return MemoryConfig(backend=backend, chroma=chroma)


def load_config(config_path: str) -> GlobalMemoryConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    enabled_val = raw.get("enabled")
    enabled = True if enabled_val is None else bool(enabled_val)
    return GlobalMemoryConfig(
        enabled=enabled,
        embedding_model=_to_openai_embedding_config(raw.get("embedding_model")),
        memory=_to_memory_config(raw.get("memory")),
    )
