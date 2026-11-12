"""Crea la base SQLite del plan de estudios a partir de CSV.

Uso:  python scripts/init_db.py [carpeta_csv] [ruta_db]
Por defecto usa los datos FICTICIOS de data/structured/sample. Reemplázalos con el plan real
(validado con la coordinación) antes de cualquier prueba con estudiantes.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chatbot.adapters.structured.sqlite_repo import init_db, load_csv_dir  # noqa: E402

csv_dir = sys.argv[1] if len(sys.argv) > 1 else "data/structured/sample"
db_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/structured/plan_estudios.db")
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(db_path)
init_db(conn)
load_csv_dir(conn, csv_dir)
print(f"Base creada en {db_path} desde {csv_dir}")
