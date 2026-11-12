"""Métricas de la sección 9 (se amplían con fidelidad por juicio humano/LLM)."""
from __future__ import annotations

from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    for pos, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / pos
    return 0.0


def rate(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0
