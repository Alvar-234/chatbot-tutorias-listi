"""Base para rerankers servidos por API (Voyage AI, NVIDIA...).

Hacen lo mismo que el cross-encoder local, pero en servidores externos: no descargan modelos
ni ocupan CPU, a cambio de enviar la pregunta (ya sin datos personales: el redactor corre
antes) y los fragmentos del corpus a un servicio externo.

Si la API falla (sin red, clave inválida, cuota agotada), se conserva el orden de la fusión
BM25 + vectorial: el chatbot responde un poco peor, pero no se cae.
"""
from __future__ import annotations

import logging
import os

import httpx

from ...domain.models import ScoredChunk

log = logging.getLogger(__name__)


class HttpReranker:
    service = ""      # nombre legible para mensajes
    key_env = ""      # variable de entorno con la clave

    def __init__(self, model_name: str, url: str, api_key: str | None = None,
                 timeout_s: float = 15.0):
        self.model_id = f"{self.service.lower()}:{model_name}"
        self._model, self._url, self._timeout = model_name, url, timeout_s
        self._key = api_key if api_key is not None else os.environ.get(self.key_env, "")
        if not self._key:
            raise ValueError(f"Falta {self.key_env} en .env para usar el reranker de {self.service}.")

    def _body(self, query: str, documents: list[str]) -> dict:
        raise NotImplementedError

    def _parse(self, data: dict) -> list[tuple[int, float]]:
        """[(índice del candidato, puntaje)] a partir de la respuesta JSON."""
        raise NotImplementedError

    def __call__(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        try:
            r = httpx.post(self._url, timeout=self._timeout,
                           headers={"Authorization": f"Bearer {self._key}",
                                    "Accept": "application/json"},
                           json=self._body(query, [c.chunk.text for c in candidates]))
            r.raise_for_status()
            scored = self._parse(r.json())
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
            log.warning("Reranker de %s no disponible (%s); se usa el orden de la fusión.",
                        self.service, e)
            return candidates
        ranked = sorted(scored, key=lambda s: s[1], reverse=True)
        # Se conserva `relevance` (similitud vectorial) y se guarda el puntaje en `rerank_score`:
        # la suficiencia usa este último si su umbral está calibrado.
        return [ScoredChunk(candidates[i].chunk, float(s), candidates[i].relevance, float(s))
                for i, s in ranked]
