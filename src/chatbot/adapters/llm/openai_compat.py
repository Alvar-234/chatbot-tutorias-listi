"""Cliente para APIs compatibles con OpenAI /chat/completions (Groq, HF Router, Ollama...)."""
from __future__ import annotations

import time
from typing import Optional

import httpx

from ...domain.models import LLMResponse
from ...domain.ports import ProviderError


class OpenAICompatProvider:
    def __init__(self, *, name: str, base_url: str, model: str, api_key: str = "",
                 timeout_s: float = 30.0, extra_body: Optional[dict] = None,
                 max_retry_wait_s: float = 0.0):
        self.name, self.model = name, model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout_s
        self._extra = dict(extra_body or {})  # p. ej. {"reasoning_effort": "low"}
        self._max_wait = max_retry_wait_s     # espera máxima ante 429 antes de pasar al siguiente

    def generate(self, messages: list[dict], *, temperature: float = 0.0,
                 max_tokens: int = 600) -> LLMResponse:
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens, **self._extra}
        try:
            r = self._post(payload)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage", {}) or {}
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            # Cuota (429), caída, modelo retirado, etc.: el pipeline pasa al siguiente proveedor.
            raise ProviderError(f"{self.name}: {exc}") from exc
        if choice.get("finish_reason") == "length":
            # Respuesta cortada por max_tokens (en modelos de razonamiento el "pensamiento"
            # también consume ese presupuesto): nunca se entrega media respuesta.
            raise ProviderError(f"{self.name}: respuesta truncada por max_tokens={max_tokens}")
        return LLMResponse(text=text.strip(), model_id=self.model, provider=self.name,
                           prompt_tokens=usage.get("prompt_tokens", 0),
                           completion_tokens=usage.get("completion_tokens", 0))

    def _post(self, payload: dict) -> httpx.Response:
        r = httpx.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
        if r.status_code == 429:
            wait = _retry_after(r)
            if wait is not None and wait <= self._max_wait:
                time.sleep(wait)
                r = httpx.post(self._url, json=payload, headers=self._headers,
                               timeout=self._timeout)
        return r


def _retry_after(r: httpx.Response) -> Optional[float]:
    try:
        return max(0.0, float(r.headers["retry-after"]))
    except (KeyError, ValueError):
        return None
