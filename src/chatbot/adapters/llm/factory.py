from __future__ import annotations

import os

from ...domain.ports import LLMProvider
from .openai_compat import OpenAICompatProvider

# provider -> (base_url por defecto, variable de entorno con la clave)
_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "local": ("http://localhost:11434/v1", None),  # Ollama / llama.cpp / vLLM
}


def build_llm_chain(cfg: dict) -> list[LLMProvider]:
    chain = []
    timeout = cfg.get("timeout_s", 30)
    max_wait = cfg.get("retry_429_max_wait_s", 0)
    for item in cfg["chain"]:
        model, p = item["model"], item["provider"]
        if "REEMPLAZAR" in model:
            raise ValueError(
                f"Define un modelo real para '{p}' en config/settings.yaml "
                "(verifica el catálogo vigente del proveedor).")
        if p not in _PROVIDERS:
            raise ValueError(f"Proveedor desconocido: {p}")
        base_url, key_env = _PROVIDERS[p]
        chain.append(OpenAICompatProvider(
            name=p, base_url=item.get("base_url", base_url), model=model,
            api_key=os.environ.get(key_env, "") if key_env else "",
            timeout_s=item.get("timeout_s", 60 if p == "local" else timeout),
            extra_body=item.get("extra"), max_retry_wait_s=max_wait))
    return chain
