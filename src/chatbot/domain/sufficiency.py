"""Evaluación de suficiencia de la evidencia (sección 7.2, paso 5).

Un fragmento recuperado NO es necesariamente evidencia suficiente. Se combinan:
  (i)  relevancia máxima >= umbral (calibrado con el subconjunto dev, no fijado a priori).
       Si hubo reranker, se usa SU puntaje: lee pregunta y fragmento juntos y separa mucho
       mejor (p. ej. tolera erratas como "mareias"). Sin reranker, o si su API falló, se usa
       la similitud vectorial. Cada reranker tiene su propia escala: el umbral es por modelo.
  (ii) las entidades de la consulta (claves, siglas, cifras) aparecen en los fragmentos.
Pendiente (opcional): verificación con modelo ligero solo en casos dudosos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .models import ScoredChunk
from .text import normalize

_ENTITY = re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ]{2,}[-]?\d*|\w*\d+\w*)\b")
# Marcadores que deja el redactor ([CORREO], [MATRICULA]...): no son entidades de la consulta.
_REDACTION_MARKER = re.compile(r"\[[A-ZÁÉÍÓÚÑ_]+\]")


@dataclass
class SufficiencyResult:
    sufficient: bool
    reason: str
    max_relevance: float = 0.0
    max_rerank_score: Optional[float] = None


def extract_entities(text: str, ignore: Iterable[str] = ()) -> set[str]:
    ignored = {normalize(i) for i in ignore}
    text = _REDACTION_MARKER.sub(" ", text)
    found = _ENTITY.findall(text)
    if text.isupper():  # consulta escrita toda en mayúsculas: solo cifras y claves con dígitos
        found = [f for f in found if any(c.isdigit() for c in f)]
    return {normalize(f) for f in found if normalize(f) not in ignored}


class SufficiencyChecker:
    def __init__(self, relevance_threshold: float, ignore_entities: Iterable[str] = (),
                 rerank_threshold: Optional[float] = None):
        self.threshold = relevance_threshold
        self.rerank_threshold = rerank_threshold
        self.ignore = tuple(ignore_entities)

    def check(self, query: str, chunks: Sequence[ScoredChunk]) -> SufficiencyResult:
        if not chunks:
            return SufficiencyResult(False, "sin_fragmentos")
        max_rel = max(c.relevance for c in chunks)
        rr = [c.rerank_score for c in chunks if c.rerank_score is not None]
        max_rr = max(rr) if rr else None
        if self.rerank_threshold is not None and max_rr is not None:
            if max_rr < self.rerank_threshold:
                return SufficiencyResult(False, "relevancia_baja", max_rel, max_rr)
        elif max_rel < self.threshold:
            return SufficiencyResult(False, "relevancia_baja", max_rel, max_rr)
        corpus = normalize(" ".join(c.chunk.text for c in chunks))
        missing = [e for e in extract_entities(query, self.ignore) if e not in corpus]
        if missing:
            return SufficiencyResult(False, f"entidades_ausentes:{sorted(missing)}", max_rel, max_rr)
        return SufficiencyResult(True, "ok", max_rel, max_rr)
