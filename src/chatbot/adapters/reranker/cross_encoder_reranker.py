"""Reordenamiento con cross-encoder (p. ej. BAAI/bge-reranker-v2-m3, multilingüe).

A diferencia de los embeddings, el cross-encoder lee la pregunta y el fragmento JUNTOS, así que
relaciona "¿cuántas veces puedo cursar?" con "tienen derecho a dos inscripciones" aunque no
compartan palabras. Es más lento: solo se aplica a los candidatos que entrega la fusión.
"""
from __future__ import annotations

from ...domain.models import ScoredChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, device: str | None = None,
                 quantize_int8: bool = False):
        from sentence_transformers import CrossEncoder  # import perezoso (extra "rag")
        self.model_id = model_name
        self._model = CrossEncoder(model_name, max_length=max_length, device=device)
        if quantize_int8:
            # Solo CPU: pesos de las capas lineales a 8 bits. ~2x más rápido, misma calidad
            # en las pruebas con el corpus (ver README).
            import torch
            torch.quantization.quantize_dynamic(self._model, {torch.nn.Linear},
                                                dtype=torch.qint8, inplace=True)

    def __call__(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates],
                                     batch_size=8, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True)
        # `relevance` (similitud vectorial) se conserva: es la base del umbral de suficiencia.
        return [ScoredChunk(c.chunk, float(s), c.relevance, float(s)) for c, s in ranked]
