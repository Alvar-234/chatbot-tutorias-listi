"""Segmentación por estructura (artículos/capítulos/secciones) y luego por ventana con solape."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .loaders import Page

_HEADING = re.compile(
    r"^(art[ií]culo\s+\d+[\w.]*|cap[ií]tulo\s+[\w.]+|t[ií]tulo\s+[\w.]+|\d+(\.\d+)*\.?\s+[A-ZÁÉÍÓÚÑ].{0,80})",
    re.IGNORECASE,
)
TOKENS_PER_WORD = 1.3  # aproximación para español; ajustar con el tokenizador real si se desea


@dataclass
class RawChunk:
    text: str
    section: str
    page: Optional[int]


def chunk_pages(pages: list[Page], size_tokens: int = 450, overlap_pct: float = 12) -> list[RawChunk]:
    size_words = max(20, int(size_tokens / TOKENS_PER_WORD))
    overlap = int(size_words * overlap_pct / 100)

    # 1) bloques por encabezado
    blocks: list[tuple[str, int, list[str]]] = []  # (sección, página inicial, líneas)
    section = ""
    for pg in pages:
        for line in pg.text.splitlines():
            if _HEADING.match(line.strip()):
                section = line.strip()[:80]
                blocks.append((section, pg.number, [line.strip()]))
            else:
                if not blocks:
                    blocks.append((section, pg.number, []))
                blocks[-1][2].append(line)

    # 2) ventanas con solape dentro de cada bloque
    chunks: list[RawChunk] = []
    for sec, page, lines in blocks:
        words = " ".join(lines).split()
        if not words:
            continue
        start = 0
        while start < len(words):
            piece = words[start:start + size_words]
            chunks.append(RawChunk(" ".join(piece), sec, page))
            if start + size_words >= len(words):
                break
            start += size_words - overlap
    return chunks
