"""ChromaDB-based memory backend implementation."""

import uuid

import chromadb

from .base import MemoryBackend


class ChromaMemoryBackend(MemoryBackend):
    """A memory backend using ChromaDB collections for vector storage."""

    def __init__(self, collection_name: str = "global_memory", persist_path: str | None = None) -> None:
        if isinstance(persist_path, str) and persist_path:
            try:
                self.client = chromadb.PersistentClient(path=persist_path)
            except Exception:
                self.client = chromadb.Client()
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, items: list[dict]) -> None:
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in items],
            embeddings=[item["embedding"] for item in items],
            metadatas=[item["metadata"] for item in items],
            documents=[item["document"] for item in items],
        )

    def query(self, query_embedding: list[float], k: int) -> dict[str, list]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        _ids = results["ids"][0] if results.get("ids") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        documents = results["documents"][0] if results.get("documents") else []

        return {
            "ids": _ids,
            "metadatas": metadatas,
            "documents": documents,
        }

    def reset(self) -> None:
        try:
            self.client.reset()
        except Exception:
            pass
