"""Reordenamiento con la API de NVIDIA (build.nvidia.com). Ver http_reranker.py.

Cada reranker tiene su propia URL: https://ai.api.nvidia.com/v1/retrieval/<modelo>/reranking
(probado con nvidia/nemotron-3.5-rerank-8b, multilingüe). Los rerankers anteriores
(llama-3.2-nv-rerankqa-1b-v2...) se retiraron el 2026-05-18.
"""
from __future__ import annotations

from .http_reranker import HttpReranker

NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/{model}/reranking"


class NvidiaReranker(HttpReranker):
    service, key_env = "NVIDIA", "NVIDIA_API_KEY"

    def __init__(self, model_name: str = "nvidia/nemotron-3.5-rerank-8b",
                 api_key: str | None = None, timeout_s: float = 15.0, url: str | None = None):
        super().__init__(model_name, url or NVIDIA_RERANK_URL.format(model=model_name),
                         api_key, timeout_s)

    def _body(self, query, documents):
        return {"model": self._model, "query": {"text": query},
                "passages": [{"text": d} for d in documents], "truncate": "END"}

    def _parse(self, data):
        return [(r["index"], r["logit"]) for r in data["rankings"]]
