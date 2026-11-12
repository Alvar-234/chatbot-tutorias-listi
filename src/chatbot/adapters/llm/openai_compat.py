"""Cliente para APIs compatibles con OpenAI /chat/completions (Groq, HF Router, Ollama...)."""
from __future__ import annotations

import httpx

from ...domain.models import LLMResponse
from ...domain.ports import ProviderError


class OpenAICompatProvider:
    def __init__(self, *, name: str, base_url: str, model: str, api_key: str = "",
                 timeout_s: float = 30.0):
        self.name, self.model = name, model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout_s

    def generate(self, messages: list[dict], *, temperature: float = 0.0,
                 max_tokens: int = 600) -> LLMResponse:
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        try:
            r = httpx.post(self._url, json=payload, headers=self._headers,
                           timeout=self._timeout)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {}) or {}
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            # Cuota (429), caída, modelo retirado, etc.: el pipeline pasa al siguiente proveedor.
            raise ProviderError(f"{self.name}: {exc}") from exc
        return LLMResponse(text=text.strip(), model_id=self.model, provider=self.name,
                           prompt_tokens=usage.get("prompt_tokens", 0),
                           completion_tokens=usage.get("completion_tokens", 0))
