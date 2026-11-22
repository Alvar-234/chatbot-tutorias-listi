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
    plan_vigente TEXT,
    caracter TEXT,            -- OB | OP
    horas_teoricas INTEGER,
    horas_practicas INTEGER,
    horas_otras INTEGER,
    horas_totales INTEGER,
    modalidad TEXT,
    nota TEXT
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


_TEXT_COLS = ("clave", "nombre", "periodo", "area", "plan_vigente", "caracter", "modalidad", "nota")
_INT_COLS = ("creditos", "horas_teoricas", "horas_practicas", "horas_otras", "horas_totales")


def _csv(d: Path, base: str) -> Path:
    """asignaturas.csv o asignaturas_ee.csv (la salida de scripts/extract_ee.py)."""
    for name in (f"{base}.csv", f"{base}_ee.csv"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"No hay {base}.csv ni {base}_ee.csv en {d}")


def load_csv_dir(conn: sqlite3.Connection, directory: str | Path) -> None:
    """Carga asignaturas y seriación. Las columnas extra (horas, nota...) son opcionales."""
    d = Path(directory)
    cols = _TEXT_COLS + _INT_COLS
    with _csv(d, "asignaturas").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            vals = {c: (r.get(c) or "").strip() or None for c in cols}
            for c in _INT_COLS:
                vals[c] = int(vals[c]) if vals[c] is not None else None
            conn.execute(f"INSERT OR REPLACE INTO asignatura ({', '.join(cols)}) "
                         f"VALUES ({', '.join('?' * len(cols))})", [vals[c] for c in cols])
    with _csv(d, "seriacion").open(encoding="utf-8-sig", newline="") as f:
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
            nombre = normalize(r["nombre"])
            # Sin \b: falla con nombres que terminan en ")" como "Bases de Datos (EJEMPLO)".
            name_re = (re.compile(r"(?<!\w)" + re.escape(nombre) + r"(?!\w)")
                       if len(nombre) >= 5 else None)
            self._index.append((r["clave"], pattern, name_re))

    def find_subjects(self, text: str) -> list[str]:
        """Claves de las materias mencionadas por clave o por nombre completo.

        - Límites de palabra: "Lengua I" no se encuentra dentro de "Lengua II".
        - Sin solaparse: en "Algoritmos y Programación" no se cuenta también "Programación".
        """
        n = normalize(text)
        hits: list[tuple[str, list[tuple[int, int]]]] = []
        for clave, pattern, name_re in self._index:
            spans = [m.span() for m in pattern.finditer(n)]
            if name_re:
                spans += [m.span() for m in name_re.finditer(n)]
            if spans:
                hits.append((clave, spans))

        def inside_longer(span, own):
            return any(o is not own and s[0] <= span[0] and span[1] <= s[1] and s != span
                       for o in hits for s in o[1])

        return [clave for clave, spans in (h for h in hits)
                if not all(inside_longer(s, (clave, spans)) for s in spans)]

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
            requisitos=tuple(Prerequisite(x["clave"], x["nombre"], x["tipo"]) for x in reqs),
            **{k: r[k] for k in ("caracter", "horas_teoricas", "horas_practicas", "horas_otras",
                                 "horas_totales", "modalidad", "nota") if k in r.keys()})
