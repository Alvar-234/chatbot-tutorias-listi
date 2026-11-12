"""Trazas por consulta en JSONL. NUNCA incluyen el texto de la pregunta ni de la respuesta (RNF5)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from ..domain.models import Trace

log = logging.getLogger("chatbot.trace")


class JsonlTracer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, trace: Trace) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **asdict(trace)}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # la observabilidad no debe tumbar la respuesta
            log.warning("No se pudo escribir la traza: %s", exc)
