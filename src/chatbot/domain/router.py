"""Clasificación de intención y enrutamiento por reglas (sección 7.2, paso 3).

Rutas: structured | documental | handoff | out_of_scope.
Si más adelante las reglas no bastan, se puede añadir un clasificador ligero aquí
sin tocar el resto del pipeline.
"""
from __future__ import annotations

from typing import Iterable

from .handoff import HandoffRules
from .models import Route, RouteDecision
from .ports import StructuredRepo
from .text import normalize

STRUCTURED_HINTS = (
    "requisito", "seriacion", "seriada", "previa", "prerrequisito",
    "creditos", "credito", "periodo", "area", "antecedente",
    # "¿qué necesito para cursar X?", "¿qué requiere X?", "¿en qué semestre va X?"
    "necesito", "necesita", "requiere", "recomend", "semestre", "horas",
)


class Router:
    def __init__(self, handoff: HandoffRules, repo: StructuredRepo,
                 out_of_scope_keywords: Iterable[str] = ()):
        self._handoff = handoff
        self._repo = repo
        self._oos = tuple(normalize(k) for k in out_of_scope_keywords)

    def route(self, text: str) -> RouteDecision:
        kind = self._handoff.check(text)
        if kind:
            return RouteDecision(Route.HANDOFF, reason=f"handoff:{kind}", handoff_kind=kind)

        n = normalize(text)
        if any(k in n for k in self._oos):
            return RouteDecision(Route.OUT_OF_SCOPE, reason="palabra_fuera_de_alcance")

        subjects = tuple(self._repo.find_subjects(text))
        if subjects and any(h in n for h in STRUCTURED_HINTS):
            return RouteDecision(Route.STRUCTURED, reason="asignatura+dato_estructurado",
                                 subjects=subjects)
        return RouteDecision(Route.DOCUMENTAL, reason="por_defecto", subjects=subjects)
