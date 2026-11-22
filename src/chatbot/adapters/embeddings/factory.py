"""Elige el modelo de embeddings según `embeddings.provider` (local | nvidia)."""
from __future__ import annotations

from ...domain.ports import Embedder


def build_embedder(cfg: dict) -> Embedder:
    provider = cfg.get("provider", "local")
    if provider == "local":
        from .sentence_transformers_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(cfg["model"], cfg.get("query_prefix", ""),
                                           cfg.get("passage_prefix", ""))
    if provider == "nvidia":
        from .nvidia_embedder import NvidiaEmbedder
        return NvidiaEmbedder(cfg["nvidia_model"], timeout_s=cfg.get("timeout_s", 30))
    raise ValueError(f"Proveedor de embeddings desconocido: {provider} (usa local | nvidia)")


def collection_name(cfg: dict) -> str:
    """Una colección de Chroma por proveedor: los vectores de modelos distintos no se mezclan
    y se puede alternar entre ellos sin reconstruir el índice ("listi" = índice local)."""
    provider = cfg.get("provider", "local")
    return "listi" if provider == "local" else f"listi-{provider}"
