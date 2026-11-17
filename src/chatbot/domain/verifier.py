"""Verificación posterior de la respuesta (sección 7.2, paso 7).

- Cada cita [n] debe corresponder a un fragmento realmente recuperado.
- Las claves y cifras de la respuesta deben aparecer en las fuentes.
Si falla, el pipeline descarta la respuesta y se abstiene o deriva.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .text import normalize

_CITE = re.compile(r"\[(\d+)\]")
_LIST_NUM = re.compile(r"(?m)^\s*\d+[.)]\s")
_CODE = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]+\b")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class VerificationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    cited_indexes: list[int] = field(default_factory=list)


_ALT_CITE = re.compile(r"[【［\[]\s*(\d+(?:\s*[,;]\s*\d+)*)\s*[】］\]]")


def normalize_citations(text: str) -> str:
    """Unifica variantes de cita a [n]: 【1】 y ［1］ (comunes en gpt-oss) o [1, 2] -> [1][2]."""
    return _ALT_CITE.sub(
        lambda m: "".join(f"[{n}]" for n in re.split(r"\s*[,;]\s*", m.group(1).strip())), text)


def _norm_num(n: str) -> str:
    return n.replace(",", ".")


class AnswerVerifier:
    def verify(self, answer: str, sources: Sequence[str],
               require_citations: bool = True) -> VerificationResult:
        problems: list[str] = []
        if not answer.strip():  # p. ej. un modelo de razonamiento que agotó max_tokens
            return VerificationResult(ok=False, problems=["respuesta_vacia"])
        cited = sorted({int(m) for m in _CITE.findall(answer)})

        bad = [i for i in cited if i < 1 or i > len(sources)]
        if bad:
            problems.append(f"cita_inexistente:{bad}")
        if require_citations and not cited:
            problems.append("sin_citas")

        body = _LIST_NUM.sub("", _CITE.sub("", answer))
        joined = " ".join(sources)
        src_codes = {normalize(c) for c in _CODE.findall(joined)}
        src_nums = {_norm_num(n) for n in _NUM.findall(joined)}

        for code in sorted({c for c in _CODE.findall(body)}):
            if normalize(code) not in src_codes:
                problems.append(f"clave_sin_respaldo:{code}")

        for num in sorted(set(_NUM.findall(_CODE.sub(" ", body)))):
            if _norm_num(num) not in src_nums:
                problems.append(f"cifra_sin_respaldo:{num}")

        valid_cited = [i for i in cited if 1 <= i <= len(sources)]
        return VerificationResult(ok=not problems, problems=problems, cited_indexes=valid_cited)
