"""Redacción de datos personales ANTES de generar, registrar o enviar a un proveedor (RNF5)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CURP = re.compile(r"\b[A-Z]{4}\d{6}[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]\d\b", re.IGNORECASE)
# Matrícula UV (formato asumido: z + S + 8 dígitos). CONFIRMAR el formato real con la coordinación.
_MATRICULA = re.compile(r"\bz[sS]\d{8}\b")
_PHONE = re.compile(r"(?<![\w])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}(?![\w])")


@dataclass
class RedactionResult:
    text: str
    types: list[str] = field(default_factory=list)


class Redactor:
    def redact(self, text: str) -> RedactionResult:
        types: list[str] = []

        def sub(pattern, marker, label, s, validate=None):
            def repl(m):
                if validate and not validate(m.group(0)):
                    return m.group(0)
                if label not in types:
                    types.append(label)
                return marker
            return pattern.sub(repl, s)

        def ten_plus_digits(s: str) -> bool:
            return 10 <= sum(c.isdigit() for c in s) <= 13

        out = sub(_EMAIL, "[CORREO]", "correo", text)
        out = sub(_CURP, "[CURP]", "curp", out)
        out = sub(_MATRICULA, "[MATRICULA]", "matricula", out)
        out = sub(_PHONE, "[TELEFONO]", "telefono", out, ten_plus_digits)
        return RedactionResult(out, types)
