"""Segmentación por estructura y luego por tamaño.

1. Se parte el texto en bloques por encabezado.
   - Encabezados FUERTES (Markdown, Título, Capítulo): siempre cortan el fragmento.
   - Encabezados DÉBILES (Artículo N, "1.2 Título numerado"): abren bloque, pero bloques
     consecutivos se empaquetan hasta el tamaño objetivo. Sin esto, un reglamento con
     artículos de 30 palabras genera cientos de fragmentos diminutos que compiten en la
     recuperación con fragmentos que solo contienen un encabezado.
2. Un bloque más grande que el tamaño objetivo se divide en ventanas con solape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .loaders import Page

# Sensibles a mayúsculas a propósito: "título profesional" a inicio de línea no es encabezado.
_MD = re.compile(r"^#{1,6}\s+(.+)")
_STRONG = re.compile(r"^(?:T[ií]tulo|TÍTULO|TITULO|Cap[ií]tulo|CAPÍTULO|CAPITULO)\s+[IVXLC\d]+\b")
_ARTICLE = re.compile(r"^(?:Art[ií]culo|ARTÍCULO|ARTICULO)\s+\d+\w*")
_NUMBERED = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÑ]")
_MAX_HEADING_CHARS = 80
TOKENS_PER_WORD = 1.3  # aproximación para español; ajustar con el tokenizador real si se desea


@dataclass
class RawChunk:
    text: str
    section: str
    page: Optional[int]


@dataclass
class _Block:
    section: str
    page: int
    strong: bool
    lines: list[str] = field(default_factory=list)


def classify_heading(line: str) -> Optional[tuple[str, bool]]:
    """(etiqueta de sección, es_fuerte) si la línea es un encabezado; None si no."""
    m = _MD.match(line)
    if m:
        return m.group(1).strip()[:_MAX_HEADING_CHARS], True
    if _STRONG.match(line):
        return line[:_MAX_HEADING_CHARS], True
    m = _ARTICLE.match(line)
    if m:
        return m.group(0), False
    if len(line) <= _MAX_HEADING_CHARS and _NUMBERED.match(line):
        return line, False
    return None


def _split_blocks(pages: list[Page]) -> list[_Block]:
    blocks: list[_Block] = []
    for pg in pages:
        for raw in pg.text.splitlines():
            line = raw.strip()
            heading = classify_heading(line)
            if heading:
                blocks.append(_Block(heading[0], pg.number, heading[1], [line]))
            else:
                if not blocks:
                    blocks.append(_Block("", pg.number, True))
                blocks[-1].lines.append(line)
    return blocks


def _label(sections: list[str]) -> str:
    named = list(dict.fromkeys(s for s in sections if s))
    if len(named) <= 1:
        return named[0] if named else ""
    return f"{named[0][:38]} – {named[-1][:38]}"


def chunk_pages(pages: list[Page], size_tokens: int = 450, overlap_pct: float = 12) -> list[RawChunk]:
    size_words = max(20, int(size_tokens / TOKENS_PER_WORD))
    overlap = int(size_words * overlap_pct / 100)
    chunks: list[RawChunk] = []

    buf_words: list[str] = []
    buf_sections: list[str] = []
    buf_page: Optional[int] = None

    def flush() -> None:
        nonlocal buf_words, buf_sections, buf_page
        if buf_words:
            chunks.append(RawChunk(" ".join(buf_words), _label(buf_sections), buf_page))
        buf_words, buf_sections, buf_page = [], [], None

    for block in _split_blocks(pages):
        words = " ".join(block.lines).split()
        if not words:
            continue
        if block.strong:
            flush()
        if len(words) > size_words:  # bloque grande: ventanas con solape, solo
            flush()
            start = 0
            while start < len(words):
                chunks.append(RawChunk(" ".join(words[start:start + size_words]),
                                       block.section, block.page))
                if start + size_words >= len(words):
                    break
                start += size_words - overlap
            continue
        if len(buf_words) + len(words) > size_words:
            flush()
        if buf_page is None:
            buf_page = block.page
        buf_words.extend(words)
        buf_sections.append(block.section)
    flush()
    return chunks
