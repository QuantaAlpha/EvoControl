"""Abstract base classes for memory backends."""

import abc


class MemoryBackend(abc.ABC):
    """Interface for memory backends that support add and query operations."""

    @abc.abstractmethod
    def add(self, items: list[dict]) -> None:
        """Adds a batch of items to the memory backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def query(self, query_embedding: list[float], k: int) -> list[dict]:
        """Queries the backend for the top-k similar items."""
        raise NotImplementedError
