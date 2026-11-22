"""Reordenamiento con la API de Voyage AI (POST /v1/rerank).

Hace lo mismo que el cross-encoder local, pero en los servidores de Voyage: no descarga
modelos ni ocupa CPU, a cambio de enviar la pregunta (ya sin datos personales: el
redactor corre antes) y los fragmentos del corpus a un servicio externo.

Si la API falla (sin red, clave inválida, cuota agotada), se conserva el orden de la fusión
BM25 + vectorial: el chatbot responde un poco peor, pero no se cae.

Cuenta sin método de pago: 3 solicitudes y 10 000 tokens por minuto, insuficiente para
reordenar ~30 candidatos por pregunta.
"""
from __future__ import annotations

import logging
import os

import httpx

from ...domain.models import ScoredChunk

log = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/rerank"


class VoyageReranker:
    def __init__(self, model_name: str = "rerank-2.5", api_key: str | None = None,
                 timeout_s: float = 15.0, url: str = VOYAGE_URL):
        self.model_id = f"voyage:{model_name}"
        self._model = model_name
        self._key = api_key if api_key is not None else os.environ.get("VOYAGE_API_KEY", "")
        if not self._key:
            raise ValueError("Falta VOYAGE_API_KEY en .env para usar el reranker de Voyage AI.")
        self._timeout = timeout_s
        self._url = url

    def __call__(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        try:
            r = httpx.post(self._url, timeout=self._timeout,
                           headers={"Authorization": f"Bearer {self._key}"},
                           json={"query": query, "model": self._model,
                                 "documents": [c.chunk.text for c in candidates]})
            r.raise_for_status()
            data = r.json()["data"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning("Reranker de Voyage no disponible (%s); se usa el orden de la fusión.", e)
            return candidates
        ranked = sorted(data, key=lambda d: d["relevance_score"], reverse=True)
        # `relevance` (similitud vectorial) se conserva: es la base del umbral de suficiencia.
        return [ScoredChunk(candidates[d["index"]].chunk, float(d["relevance_score"]),
                            candidates[d["index"]].relevance) for d in ranked]
