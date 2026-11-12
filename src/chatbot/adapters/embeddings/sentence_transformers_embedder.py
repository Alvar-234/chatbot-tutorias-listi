"""Embeddings locales con sentence-transformers (BGE-M3, multilingual-e5, EmbeddingGemma...)."""
from __future__ import annotations

from typing import Sequence


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = "",
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer  # import perezoso (extra "rag")
        self.model_id = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._qp, self._pp = query_prefix, passage_prefix

    def embed_query(self, text: str) -> list[float]:
        v = self._model.encode(self._qp + text, normalize_embeddings=True)
        return v.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vs = self._model.encode([self._pp + t for t in texts], normalize_embeddings=True,
                                batch_size=16, show_progress_bar=False)
        return vs.tolist()
