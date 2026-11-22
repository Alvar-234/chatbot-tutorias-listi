"""Extrae los datos generales de los programas de EE (PDF) a CSV revisables.

Uso:  python scripts/extract_ee.py [carpeta_pdf] [carpeta_salida]
Por defecto lee documentos_ee_pdf/ y escribe en data/structured/extraido/:
  - asignaturas_ee.csv  una fila por EE: programas PDF + EE del mapa curricular sin programa
                       (compatible con scripts/init_db.py); periodo sugerido según el mapa
  - seriacion_ee.csv    prerrequisitos resueltos a claves
Aplica la revisión manual de data/structured/validaciones_ee.yaml. La columna `revisar`
explica qué sigue sin cuadrar. Revisa y valida con la coordinación ANTES de
cargar los datos en la base (la seriación es justo donde el chatbot no puede equivocarse).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml  # noqa: E402

from chatbot.ingestion.ee_programs import (apply_validations, check_period_credits,  # noqa: E402
                                          merge_curricular_map, parse_curricular_map,
                                          parse_program, resolve_prerequisites)

src = Path(sys.argv[1] if len(sys.argv) > 1 else "documentos_ee_pdf")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/structured/extraido")
out.mkdir(parents=True, exist_ok=True)

from pypdf import PdfReader  # noqa: E402

programs = []
for pdf in sorted(src.glob("*.pdf")):  # glob: los nombres traen acentos descompuestos (NFD)
    pages = PdfReader(pdf).pages
    programs.append(parse_program("\n".join(p.extract_text() or "" for p in pages[:3]), pdf.name))
for name in ("asignaturas_ee.csv", "seriacion_ee.csv"):  # Windows bloquea lo abierto en Excel
    try:
        if (out / name).exists():
            (out / name).open("a").close()
    except PermissionError:
        raise SystemExit(f"No se puede escribir {out / name}: ciérralo en Excel y vuelve a ejecutar.")

# Paso 2: periodo sugerido y EE sin programa, desde el mapa curricular.
mapa = Path("Documentos/Extraidos a .MD/mapa_curricular_LISTI.md")
cmap = parse_curricular_map(mapa.read_text(encoding="utf-8"))
merge_curricular_map(programs, cmap)

validaciones = Path("data/structured/validaciones_ee.yaml")
recomendadas = apply_validations(
    programs, yaml.safe_load(validaciones.read_text(encoding="utf-8")) if validaciones.exists() else {})
seriacion = resolve_prerequisites(programs) + recomendadas

dups = {p.clave for p in programs if [q.clave for q in programs].count(p.clave) > 1}
for p in programs:
    if p.clave in dups:
        p.revisar.append("clave_duplicada")

COLS = ["clave", "nombre", "creditos", "periodo", "area", "plan_vigente", "caracter",
        "horas_teoricas", "horas_practicas", "horas_otras", "horas_totales", "agrupacion",
        "modalidad", "oportunidades_evaluacion", "equivalencia", "prerrequisitos_texto",
        "cupo_max", "cupo_min", "nota", "nombre_mapa", "archivo", "revisar"]
# utf-8-sig: Excel muestra bien los acentos; init_db lo lee igual.
with (out / "asignaturas_ee.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, COLS)
    w.writeheader()
    for p in programs:
        row = {c: getattr(p, c, "") for c in COLS}
        row.update(clave=p.clave, plan_vigente=p.plan, revisar="; ".join(p.revisar))
        w.writerow({k: "" if v is None else v for k, v in row.items()})
with (out / "seriacion_ee.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, ["clave", "requiere_clave", "tipo", "requiere_texto_original", "revisar"])
    w.writeheader()
    w.writerows(seriacion)

n_rev = sum(1 for p in programs if p.revisar)
print(f"{len(programs)} programas -> {out}/asignaturas_ee.csv ({n_rev} con algo que revisar)")
print(f"{len(seriacion) - len(recomendadas)} prerrequisitos + {len(recomendadas)} recomendaciones "
      f"-> {out}/seriacion_ee.csv")
for p in programs:
    if p.revisar:
        print(f"  REVISAR {p.nombre or p.archivo}: {'; '.join(p.revisar)}")
print("Nombres distintos entre programa y mapa (emparejados por parecido):")
for p in programs:
    if p.nombre_mapa:
        print(f"  {p.nombre!r}  <->  mapa: {p.nombre_mapa!r}")
for aviso in check_period_credits(programs, cmap):
    print("  CRÉDITOS", aviso)
