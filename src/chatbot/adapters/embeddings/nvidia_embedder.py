"""Embeddings por la API de NVIDIA (POST https://integrate.api.nvidia.com/v1/embeddings).

Probado con nvidia/nemotron-3.5-embed-1b (multilingüe, 2048 dimensiones). Los modelos de
NVIDIA distinguen consulta y pasaje con `input_type`, en lugar de prefijos de texto.
Al construir el índice se envían TODOS los fragmentos del corpus a NVIDIA.
"""
from __future__ import annotations

import os
import time
from typing import Sequence

import httpx

NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"


class NvidiaEmbedder:
    def __init__(self, model_name: str = "nvidia/nemotron-3.5-embed-1b",
                 api_key: str | None = None, timeout_s: float = 30.0, batch_size: int = 32,
                 url: str = NVIDIA_EMBED_URL, max_retries: int = 3):
        self.model_id = model_name
        self._key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
        if not self._key:
            raise ValueError("Falta NVIDIA_API_KEY en .env para usar los embeddings de NVIDIA.")
        self._url, self._timeout, self._batch = url, timeout_s, batch_size
        self._retries = max_retries

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        body = {"model": self.model_id, "input": texts, "input_type": input_type,
                "encoding_format": "float", "truncate": "END"}
        for attempt in range(self._retries + 1):
            r = httpx.post(self._url, json=body, timeout=self._timeout,
                           headers={"Authorization": f"Bearer {self._key}",
                                    "Accept": "application/json"})
            if r.status_code == 429 and attempt < self._retries:  # cuota: esperar y reintentar
                time.sleep(float(r.headers.get("retry-after", 5 * (attempt + 1))))
                continue
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        raise RuntimeError("inalcanzable")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.warnings: list[str] = []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = list(texts[i:i + self._batch])
            try:
                out.extend(self._embed(batch, "passage"))
            except httpx.HTTPStatusError as e:
                if not _rejected_by_filter(e):
                    raise
                out.extend(self._embed_one_by_one(batch))  # solo uno del lote suele ser el culpable
        return out

    def _embed_one_by_one(self, texts: list[str]) -> list[list[float]]:
        """El filtro de contenido de NVIDIA rechaza a veces fragmentos inofensivos (p. ej. un
        temario de bases de datos distribuidas). Se intenta con sus dos mitades (promediadas)
        y, si tampoco pasan, con su primera línea (programa y sección). El texto completo
        sigue en el índice léxico (BM25)."""
        out = []
        for t in texts:
            for attempt in (lambda: [t], lambda: _halves(t), lambda: [t[:120]]):
                parts = attempt()
                try:
                    vecs = self._embed(parts, "passage")
                except httpx.HTTPStatusError as e:
                    if not _rejected_by_filter(e):
                        raise
                    continue
                if len(parts) > 1 or parts[0] != t:
                    self.warnings.append(f"filtro de NVIDIA: vector aproximado ({len(parts)} parte(s)) "
                                         f"para el fragmento que empieza {t[:70]!r}")
                out.append(_mean(vecs))
                break
            else:
                raise RuntimeError(f"El filtro de NVIDIA rechaza el fragmento {t[:70]!r} incluso recortado")
        return out


def _rejected_by_filter(e: httpx.HTTPStatusError) -> bool:
    return e.response.status_code == 422 and "safety" in e.response.text.lower()


def _halves(text: str) -> list[str]:
    words = text.split()
    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _mean(vectors: list[list[float]]) -> list[float]:
    if len(vectors) == 1:
        return vectors[0]
    return [sum(xs) / len(vectors) for xs in zip(*vectors)]
