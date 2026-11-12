"""Reglas explícitas de derivación a tutoría (sección 7.3)."""
from __future__ import annotations

import re
from typing import Optional

from .text import normalize

PRIORITY = ("risk", "sensitive", "requested", "administrative")


class HandoffRules:
    def __init__(self, topics: dict, contact: Optional[dict] = None):
        self._patterns: dict[str, list[re.Pattern]] = {}
        for kind in PRIORITY:
            phrases = topics.get(kind, []) or []
            # Coincidencia por inicio de palabra (prefijo): "suicid" cubre "suicidio".
            self._patterns[kind] = [
                re.compile(r"\b" + re.escape(normalize(p))) for p in phrases
            ]
        self.contact = contact or {}

    def check(self, text: str) -> Optional[str]:
        """Devuelve el tipo de derivación (risk > sensitive > requested > administrative)."""
        n = normalize(text)
        for kind in PRIORITY:
            if any(p.search(n) for p in self._patterns[kind]):
                return kind
        return None

    def message(self, kind: str) -> str:
        tutoria = self.contact.get("tutoria", "tu tutor o la coordinación de tutorías")
        ayuda = self.contact.get("ayuda_inmediata", "los servicios de apoyo de la universidad")
        if kind == "risk":
            return (
                "Lamento que estés pasando por esto, y es importante que hables con alguien "
                f"ahora. Puedes acudir de inmediato a: {ayuda}. También puedes buscar a tu "
                f"tutor o a una persona de confianza. Contacto de tutorías: {tutoria}."
            )
        if kind == "administrative":
            return (
                "Este asistente solo brinda información y no puede realizar ni modificar "
                f"trámites. Para gestionarlo, acude con tu tutor o a la instancia "
                f"correspondiente: {tutoria}."
            )
        return (
            "Este tema requiere la atención de una persona. Te recomiendo platicar con tu "
            f"tutor. Contacto de tutorías: {tutoria}."
        )

    def repeated_insufficient_message(self) -> str:
        tutoria = self.contact.get("tutoria", "tu tutor o la coordinación de tutorías")
        return (
            "No he podido encontrar información suficiente para ayudarte con esto. "
            f"Te recomiendo consultarlo con tu tutor. Contacto de tutorías: {tutoria}."
        )
