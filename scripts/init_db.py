"""Crea la base SQLite del plan de estudios a partir de CSV.

Uso:  python scripts/init_db.py [carpeta_csv] [ruta_db] [--forzar]

Por defecto usa los datos reales de data/structured/extraido/ (los genera
`python scripts/extract_ee.py` a partir de los programas de EE y de validaciones_ee.yaml).
Si no existen, usa los datos FICTICIOS de data/structured/sample y lo avisa.

La base se crea desde cero cada vez. Si alguna fila tiene algo en la columna `revisar`, no
se carga (salvo con --forzar): la seriación es justo donde el chatbot no puede equivocarse.
"""
import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chatbot.adapters.structured.sqlite_repo import _csv, init_db, load_csv_dir  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
forzar = "--forzar" in sys.argv
REAL, DEMO = Path("data/structured/extraido"), Path("data/structured/sample")
csv_dir = Path(args[0]) if args else (REAL if REAL.exists() else DEMO)
db_path = Path(args[1] if len(args) > 1 else "data/structured/plan_estudios.db")
if csv_dir == DEMO:
    print("AVISO: se cargan los datos FICTICIOS de ejemplo (DEMO). Para los reales ejecuta antes "
          "python scripts/extract_ee.py")

pendientes = []
for base in ("asignaturas", "seriacion"):
    with _csv(csv_dir, base).open(encoding="utf-8-sig", newline="") as f:
        pendientes += [f"{base}: {r.get('nombre') or r.get('clave')}: {r['revisar']}"
                       for r in csv.DictReader(f) if (r.get("revisar") or "").strip()]
if pendientes and not forzar:
    print("No se creó la base: hay filas pendientes de revisar (usa --forzar para cargarlas igual).")
    print("\n".join("  " + p for p in pendientes))
    sys.exit(1)

db_path.parent.mkdir(parents=True, exist_ok=True)
tmp = db_path.with_suffix(".db.tmp")
tmp.unlink(missing_ok=True)
conn = sqlite3.connect(tmp)
init_db(conn)
load_csv_dir(conn, csv_dir)
n_asig = conn.execute("SELECT COUNT(*) FROM asignatura").fetchone()[0]
n_seri = conn.execute("SELECT tipo, COUNT(*) FROM seriacion GROUP BY tipo").fetchall()
conn.close()
tmp.replace(db_path)  # se reemplaza solo cuando la carga terminó bien
print(f"Base creada en {db_path} desde {csv_dir}: {n_asig} asignaturas; seriación: "
      + (", ".join(f"{n} {t}" for t, n in n_seri) or "ninguna"))
