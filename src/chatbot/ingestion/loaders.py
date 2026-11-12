"""Carga de documentos a páginas de texto. PDF nativo vía pypdf; PDF escaneado requiere OCR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class NeedsOCR(Exception):
    """El PDF no tiene capa de texto (escaneado). Aplicar OCR antes de ingerir."""


@dataclass
class Page:
    number: int
    text: str


def load_document(path: str | Path) -> list[Page]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".txt", ".md"):
        return [Page(1, p.read_text(encoding="utf-8"))]
    if suffix == ".pdf":
        from pypdf import PdfReader  # import perezoso (extra "rag")
        pages = [Page(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(PdfReader(str(p)).pages)]
        if not any(pg.text.strip() for pg in pages):
            raise NeedsOCR(f"{p.name} no contiene texto extraíble")
        return pages
    raise ValueError(f"Formato no soportado: {suffix}")
