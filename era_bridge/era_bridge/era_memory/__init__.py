"""era_memory -- ERA's local, persistent memory engine."""

from .memory_config import MemoryConfig
from .memory_manager import (
    MemoryManager, MemoryNotFoundError, MemoryUnavailableError, SecretContentError, SensitiveContentError,
)
from .memory_models import Category, Confidence, ContextPackage, IngestResult, Memory, MemoryType
from .memory_retriever import CallableEmbedding, EmbeddingService, HashingEmbedding
from .memory_store import MemoryStore, MemoryStoreError

__all__ = [
    "MemoryConfig", "MemoryManager", "MemoryStore", "MemoryStoreError", "MemoryNotFoundError",
    "MemoryUnavailableError", "SecretContentError", "SensitiveContentError", "Category", "Confidence",
    "ContextPackage", "IngestResult", "Memory", "MemoryType", "EmbeddingService", "HashingEmbedding",
    "CallableEmbedding",
]
