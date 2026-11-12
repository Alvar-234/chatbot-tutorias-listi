"""Limpieza: encabezados/pies repetidos, espacios y guiones de corte de línea."""
from __future__ import annotations

import re
from collections import Counter

from .loaders import Page


def clean_pages(pages: list[Page]) -> list[Page]:
    if not pages:
        return pages
    counts: Counter = Counter()
    for pg in pages:
        for line in {ln.strip() for ln in pg.text.splitlines() if ln.strip()}:
            counts[line] += 1
    threshold = max(3, int(len(pages) * 0.5))
    repeated = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 120}

    out = []
    for pg in pages:
        lines = [ln for ln in pg.text.splitlines() if ln.strip() not in repeated]
        text = "\n".join(lines)
        text = re.sub(r"-\n(\w)", r"\1", text)      # palabras cortadas al final de línea
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        out.append(Page(pg.number, text.strip()))
    return out
