"""Evaluación de suficiencia de la evidencia (sección 7.2, paso 5).

Un fragmento recuperado NO es necesariamente evidencia suficiente. Se combinan:
  (i)  relevancia máxima >= umbral (calibrado con el subconjunto dev, no fijado a priori);
  (ii) las entidades de la consulta (claves, siglas, cifras) aparecen en los fragmentos.
Pendiente (opcional): verificación con modelo ligero solo en casos dudosos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

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


def extract_entities(text: str, ignore: Iterable[str] = ()) -> set[str]:
    ignored = {normalize(i) for i in ignore}
    text = _REDACTION_MARKER.sub(" ", text)
    found = _ENTITY.findall(text)
    if text.isupper():  # consulta escrita toda en mayúsculas: solo cifras y claves con dígitos
        found = [f for f in found if any(c.isdigit() for c in f)]
    return {normalize(f) for f in found if normalize(f) not in ignored}


class SufficiencyChecker:
    def __init__(self, relevance_threshold: float, ignore_entities: Iterable[str] = ()):
        self.threshold = relevance_threshold
        self.ignore = tuple(ignore_entities)

    def check(self, query: str, chunks: Sequence[ScoredChunk]) -> SufficiencyResult:
        if not chunks:
            return SufficiencyResult(False, "sin_fragmentos")
        max_rel = max(c.relevance for c in chunks)
        if max_rel < self.threshold:
            return SufficiencyResult(False, "relevancia_baja", max_rel)
        corpus = normalize(" ".join(c.chunk.text for c in chunks))
        missing = [e for e in extract_entities(query, self.ignore) if e not in corpus]
        if missing:
            return SufficiencyResult(False, f"entidades_ausentes:{sorted(missing)}", max_rel)
        return SufficiencyResult(True, "ok", max_rel)
