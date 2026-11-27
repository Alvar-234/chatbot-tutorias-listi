"""Preguntas de seguimiento ("¿y general?", "¿y cuántas horas?").

Solas no dicen de qué tratan: la búsqueda con "y general" no recupera nada útil y el chatbot
se abstiene aunque la respuesta esté en el mismo documento que la pregunta anterior. Para
BUSCAR (y enrutar) se les antepone la consulta anterior; el LLM recibe la pregunta tal cual,
junto con el historial.
"""
from __future__ import annotations

import re

from .text import normalize

# Conectores con los que suele empezar una pregunta que continúa la anterior.
_CONNECTORS = {"y", "e", "o", "tambien", "entonces", "pero", "ademas", "igual"}
_MAX_SHORT_WORDS = 2  # "¿general?", "¿y optativas?": demasiado cortas para buscarlas solas


def is_follow_up(text: str) -> bool:
    words = re.findall(r"\w+", normalize(text))
    if not words:
        return False
    return words[0] in _CONNECTORS or len(words) <= _MAX_SHORT_WORDS


def contextual_query(text: str, previous: str) -> str:
    """Consulta para buscar: la anterior (ya contextualizada) más la actual."""
    return f"{previous} {text}".strip() if previous and is_follow_up(text) else text
