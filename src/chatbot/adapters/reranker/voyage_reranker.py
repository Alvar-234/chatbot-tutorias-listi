"""Reordenamiento con la API de Voyage AI (POST /v1/rerank). Ver http_reranker.py.

Cuenta sin método de pago: 3 solicitudes y 10 000 tokens por minuto, insuficiente para
reordenar ~30 candidatos por pregunta.
"""
from __future__ import annotations

from .http_reranker import HttpReranker

VOYAGE_URL = "https://api.voyageai.com/v1/rerank"


class VoyageReranker(HttpReranker):
    service, key_env = "Voyage", "VOYAGE_API_KEY"

    def __init__(self, model_name: str = "rerank-2.5", api_key: str | None = None,
                 timeout_s: float = 15.0, url: str = VOYAGE_URL):
        super().__init__(model_name, url, api_key, timeout_s)

    def _body(self, query, documents):
        return {"query": query, "model": self._model, "documents": documents}

    def _parse(self, data):
        return [(d["index"], d["relevance_score"]) for d in data["data"]]
