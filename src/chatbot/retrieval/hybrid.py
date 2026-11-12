"""Recuperación híbrida: BM25 + vectorial, fusionados con Reciprocal Rank Fusion (sección 6.3)."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..domain.models import ScoredChunk
from ..domain.ports import Embedder, LexicalIndex, VectorStore


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + pos)
    return scores


class HybridRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, lexical: LexicalIndex, *,
                 k: int = 5, k_candidates: int = 20, rrf_k: int = 60,
                 reranker: Optional[Callable[[str, list[ScoredChunk]], list[ScoredChunk]]] = None):
        self.embedder, self.vector, self.lexical = embedder, vector_store, lexical
        self.k, self.k_candidates, self.rrf_k, self.reranker = k, k_candidates, rrf_k, reranker

    def retrieve(self, query: str) -> list[ScoredChunk]:
        vec_hits = self.vector.search(self.embedder.embed_query(query), self.k_candidates)
        lex_hits = [h for h in self.lexical.search(query, self.k_candidates)]

        by_id = {h.chunk.id: h for h in lex_hits}
        by_id.update({h.chunk.id: h for h in vec_hits})  # la versión vectorial trae `relevance`
        fused = rrf_fuse([[h.chunk.id for h in vec_hits], [h.chunk.id for h in lex_hits]],
                         self.rrf_k)

        out = []
        for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            h = by_id[cid]
            if not h.chunk.vigente:  # nunca citar documentos obsoletos (sección 6.2)
                continue
            out.append(ScoredChunk(h.chunk, score, h.relevance))
        out = out[: self.k]
        return self.reranker(query, out) if self.reranker else out
