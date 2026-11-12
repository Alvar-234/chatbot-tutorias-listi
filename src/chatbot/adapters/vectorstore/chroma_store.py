"""Almacén vectorial local con Chroma (ADR-005). Distancia coseno: relevancia = 1 - distancia.

NOTA: este adaptador no se ejecutó en el esqueleto (requiere el extra "rag"); pruébalo
con tu corpus real antes de confiar en él.
"""
from __future__ import annotations

from typing import Sequence

from ...domain.models import Chunk, ScoredChunk


class ChromaVectorStore:
    def __init__(self, path: str, collection: str = "listi"):
        import chromadb  # import perezoso (extra "rag")
        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        step = 256
        for i in range(0, len(chunks), step):
            part = chunks[i:i + step]
            self._col.add(
                ids=[c.id for c in part],
                embeddings=[list(v) for v in vectors[i:i + step]],
                documents=[c.text for c in part],
                metadatas=[c.to_metadata() for c in part],
            )

    def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        res = self._col.query(query_embeddings=[list(vector)], n_results=k,
                              where={"vigente": True})
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            sim = max(0.0, 1.0 - float(dist))
            out.append(ScoredChunk(Chunk.from_metadata(cid, doc, meta), sim, sim))
        return out
