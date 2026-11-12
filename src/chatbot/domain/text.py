"""Utilidades de texto compartidas (sin dependencias externas)."""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    """Minúsculas y sin acentos, para comparar de forma tolerante."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _WORD.findall(normalize(text))
