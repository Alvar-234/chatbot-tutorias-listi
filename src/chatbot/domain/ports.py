"""Puertos: contratos que el dominio necesita. Los adaptadores los implementan.

Cambiar de proveedor, almacén vectorial o modelo de embeddings = escribir otro adaptador.
"""
from __future__ import annotations

from typing import Optional, Protocol, Sequence

from .models import Chunk, LLMResponse, ScoredChunk, SubjectFact


class ProviderError(Exception):
    """Fallo de un proveedor de inferencia (cuota, red, catálogo, etc.)."""


class LLMProvider(Protocol):
    name: str

    def generate(self, messages: list[dict], *, temperature: float = 0.0,
                 max_tokens: int = 600) -> LLMResponse:
        """Lanza ProviderError si el proveedor no puede responder."""
        ...


class Embedder(Protocol):
    model_id: str

    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        """Devuelve fragmentos vigentes con `relevance` = similitud [0-1]."""
        ...
    def reset(self) -> None: ...


class LexicalIndex(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[ScoredChunk]: ...


class StructuredRepo(Protocol):
    def find_subjects(self, text: str) -> list[str]:
        """Claves de asignaturas mencionadas (por clave o por nombre)."""
        ...
    def get_facts(self, clave: str) -> Optional[SubjectFact]: ...
