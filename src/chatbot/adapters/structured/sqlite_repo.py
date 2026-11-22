"""Datos estructurados del plan de estudios y seriación (ADR-003). Aquí NO hay RAG ni LLM."""
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import Optional

from ...domain.models import Prerequisite, SubjectFact
from ...domain.text import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS asignatura (
    clave TEXT PRIMARY KEY,
    nombre TEXT NOT NULL,
    creditos INTEGER,
    periodo TEXT,
    area TEXT,
    plan_vigente TEXT
);
CREATE TABLE IF NOT EXISTS seriacion (
    clave TEXT NOT NULL REFERENCES asignatura(clave),
    requiere_clave TEXT NOT NULL REFERENCES asignatura(clave),
    tipo TEXT NOT NULL CHECK (tipo IN ('previa', 'simultanea', 'recomendada')),
    PRIMARY KEY (clave, requiere_clave)
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def load_csv_dir(conn: sqlite3.Connection, directory: str | Path) -> None:
    d = Path(directory)
    with (d / "asignaturas.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            conn.execute(
                "INSERT OR REPLACE INTO asignatura VALUES (?,?,?,?,?,?)",
                (r["clave"].strip(), r["nombre"].strip(),
                 int(r["creditos"]) if r["creditos"].strip() else None,
                 r["periodo"].strip() or None, r["area"].strip() or None,
                 r["plan_vigente"].strip() or None))
    with (d / "seriacion.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            conn.execute("INSERT OR REPLACE INTO seriacion VALUES (?,?,?)",
                         (r["clave"].strip(), r["requiere_clave"].strip(), r["tipo"].strip()))
    conn.commit()


class SqliteStructuredRepo:
    def __init__(self, path: str | Path = ":memory:", conn: Optional[sqlite3.Connection] = None):
        self._conn = conn or sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._load_index()

    def _load_index(self) -> None:
        rows = self._conn.execute("SELECT clave, nombre FROM asignatura").fetchall()
        self._index = []
        for r in rows:
            parts = [p for p in re.split(r"[\s\-]+", normalize(r["clave"])) if p]
            pattern = re.compile(r"\b" + r"[\s\-]?".join(re.escape(p) for p in parts) + r"\b")
            self._index.append((r["clave"], pattern, normalize(r["nombre"])))

    def find_subjects(self, text: str) -> list[str]:
        n = normalize(text)
        found: list[str] = []
        for clave, pattern, nombre in self._index:
            if pattern.search(n) or (len(nombre) >= 5 and nombre in n):
                if clave not in found:
                    found.append(clave)
        return found

    def get_facts(self, clave: str) -> Optional[SubjectFact]:
        r = self._conn.execute("SELECT * FROM asignatura WHERE clave = ?", (clave,)).fetchone()
        if not r:
            return None
        reqs = self._conn.execute(
            """SELECT s.requiere_clave AS clave, a.nombre AS nombre, s.tipo AS tipo
               FROM seriacion s JOIN asignatura a ON a.clave = s.requiere_clave
               WHERE s.clave = ? ORDER BY s.requiere_clave""", (clave,)).fetchall()
        return SubjectFact(
            clave=r["clave"], nombre=r["nombre"], creditos=r["creditos"], periodo=r["periodo"],
            area=r["area"], plan_vigente=r["plan_vigente"],
            requisitos=tuple(Prerequisite(x["clave"], x["nombre"], x["tipo"]) for x in reqs))
