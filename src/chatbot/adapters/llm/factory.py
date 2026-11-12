from __future__ import annotations

from ...domain.ports import LLMProvider
from .groq_provider import GroqProvider
from .hf_provider import HuggingFaceProvider
from .local_provider import LocalProvider


def build_llm_chain(cfg: dict) -> list[LLMProvider]:
    chain = []
    timeout = cfg.get("timeout_s", 30)
    for item in cfg["chain"]:
        model = item["model"]
        if "REEMPLAZAR" in model:
            raise ValueError(
                f"Define un modelo real para '{item['provider']}' en config/settings.yaml "
                "(verifica el catálogo vigente del proveedor).")
        p = item["provider"]
        if p == "groq":
            chain.append(GroqProvider(model, timeout))
        elif p == "huggingface":
            chain.append(HuggingFaceProvider(model, timeout))
        elif p == "local":
            chain.append(LocalProvider(model, item.get("base_url", "http://localhost:11434/v1")))
        else:
            raise ValueError(f"Proveedor desconocido: {p}")
    return chain
