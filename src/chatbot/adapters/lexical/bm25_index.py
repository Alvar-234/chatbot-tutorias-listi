"""BM25 en Python puro (suficiente para el corpus de cientos de fragmentos)."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

from ...domain.models import Chunk, ScoredChunk
from ...domain.text import tokenize


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = [c for c in chunks if c.vigente]
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(c.text)) for c in self.chunks]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        chunks = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                chunks.append(Chunk.from_metadata(r["id"], r["text"], r["meta"]))
        return cls(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        scored = []
        for chunk, doc, dl in zip(self.chunks, self._docs, self._len):
            s = 0.0
            for t in terms:
                f = doc.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1))
                s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append(ScoredChunk(chunk, s, 0.0))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
