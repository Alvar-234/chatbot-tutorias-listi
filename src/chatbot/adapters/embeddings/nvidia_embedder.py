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
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._embed(list(texts[i:i + self._batch]), "passage"))
        return out
