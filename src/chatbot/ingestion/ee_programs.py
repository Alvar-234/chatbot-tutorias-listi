"""Extrae los datos generales de los programas de experiencia educativa (formato UV).

Cada programa trae un bloque numerado ("6. Nombre de la Experiencia Educativa", "8. Carácter",
"10. Valores", "15. EE prerrequisito(s)"...). De ahí salen los datos exactos que van a la
base SQLite (créditos, horas, carácter, seriación). El resto del programa (temas,
evaluación, bibliografía) es texto para el índice de búsqueda, no para SQL.

El resultado NO se carga directo: se escribe en CSV con una columna `revisar` que explica
qué no cuadró, para validarlo con la coordinación antes de usarlo con estudiantes.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

AREAS = {  # mismo vocabulario que el mapa curricular
    "basica": "basica", "formacion basica": "basica", "iniciacion a la disciplina": "iniciacion_disciplina",
    "disciplinar": "disciplinar", "area de formacion disciplinaria": "disciplinar",
    "formacion disciplinaria": "disciplinar", "terminal": "terminal", "eleccion libre": "eleccion_libre",
}
SIN_PRERREQUISITO = {"ninguno", "ninguna", "sin prerrequisito", "sin prerrequisitos", "n/a", "na", "no aplica"}
_PAGE_NUMBER = re.compile(r"^\s*\d{1,2}\s*$")
_VALORES = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*(.*)$", re.S)


def fold(text: str) -> str:
    """minúsculas, sin acentos y con espacios simples (para comparar nombres)."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().lower()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")


def _clean(text: str) -> str:
    text = re.sub(r"\s*-\s*\n\s*", "-", text)  # "Curso-\nTaller" -> "Curso-Taller"
    return re.sub(r"\s+", " ", text).strip()


def _between(text: str, start: str, end: str) -> Optional[str]:
    m = re.search(start + r"(.*?)" + end, text, re.S)
    return m.group(1) if m else None


@dataclass
class EEProgram:
    archivo: str
    nombre: str = ""
    plan: str = ""
    area: str = ""
    caracter: str = ""
    agrupacion: str = ""
    horas_teoricas: Optional[int] = None
    horas_practicas: Optional[int] = None
    horas_otras: Optional[int] = None
    horas_totales: Optional[int] = None
    creditos: Optional[int] = None
    equivalencia: str = ""
    modalidad: str = ""
    oportunidades_evaluacion: str = ""
    prerrequisitos_texto: str = ""
    cupo_max: Optional[int] = None
    cupo_min: Optional[int] = None
    nota: str = ""
    periodo: str = ""        # periodo sugerido según el mapa curricular
    nombre_mapa: str = ""    # cómo lo llama el mapa, si difiere del programa
    revisar: list[str] = field(default_factory=list)

    @property
    def clave(self) -> str:
        # PROVISIONAL: el campo "5. Código" viene vacío en todos los programas y el "código"
        # del mapa curricular (FD 3309...) se repite entre materias. Reemplazar por la clave oficial.
        return slug(self.nombre)


def parse_program(text: str, archivo: str = "") -> EEProgram:
    """Lee el bloque de datos generales (texto de las primeras páginas del PDF)."""
    p = EEProgram(archivo)
    # Los números de página quedan sueltos en medio de los campos ("\n 2 \n"): se quitan.
    text = "\n".join(l for l in text.splitlines() if not _PAGE_NUMBER.match(l))
    text = unicodedata.normalize("NFC", text)

    # El año del plan va en la línea siguiente al título, con redacciones distintas:
    # "Licenciatura en ... Información, 2023", "... Información 2023", "... Información, año 2023"
    m = re.search(r"Programa de experiencia educativa\s*\n[^\n]*?(\d{4})", text)
    p.plan = f"LISTI {m.group(1)}" if m else ""

    nombre = _between(text, r"6\.\s*Nombre de la Experiencia\s*Educativa", r"7\.\s*[ÁA]rea de Formaci")
    p.nombre = _clean(nombre or "")
    if not p.nombre:
        p.revisar.append("nombre_no_encontrado")

    ac = _clean(_between(text, r"8\.\s*Car[áa]cter", r"9\.\s*Agrupaci") or "")
    m = re.search(r"\b(Obligatori[ao]|Optativ[ao])\b\s*$", ac, re.I)
    if m:
        p.caracter = "OB" if m.group(1).lower().startswith("obl") else "OP"
        area_txt = fold(ac[:m.start()])
    else:
        p.revisar.append("caracter_no_reconocido")
        area_txt = fold(ac)
    p.area = AREAS.get(area_txt, "")
    if not p.area:
        p.area = area_txt
        p.revisar.append(f"area_no_reconocida:{area_txt or 'vacia'}")

    p.agrupacion = _clean(_between(text, r"9\.\s*Agrupaci[óo]n curricular distintiva", r"10\.\s*Valores") or "")

    valores = _between(text, r"Equivalencia\s*\(s\)", r"11\.\s*Modalidad")
    m = _VALORES.match(valores or "")
    if m:
        (p.horas_teoricas, p.horas_practicas, p.horas_otras,
         p.horas_totales, p.creditos) = (int(g) for g in m.groups()[:5])
        eq = _clean(m.group(6))
        p.equivalencia = "" if fold(eq) in SIN_PRERREQUISITO else eq
        # Reglas UV de los programas revisados: horas totales = (HT + HP) x 15 + otras;
        # créditos = 2 x HT + HP. Las EE especiales (servicio social, experiencia
        # recepcional, prácticas) pueden no seguirlas: se marcan para confirmar, no se corrigen.
        if (p.horas_teoricas + p.horas_practicas) * 15 + p.horas_otras != p.horas_totales:
            p.revisar.append("horas_totales_no_cuadran")
        if 2 * p.horas_teoricas + p.horas_practicas != p.creditos:
            p.revisar.append("creditos_no_cuadran_con_horas")
    else:
        p.revisar.append("valores_no_encontrados")

    # El encabezado 14 llega partido de varias formas ("Oportunidade / s de / evaluación"):
    # se ancla en "14. Oportunidad" y se toma desde "M:".
    mod = _clean(_between(text, r"14\.\s*Oportunidad", r"15\.\s*EE prerrequisito") or "")
    mm = re.search(r"M:\s*(.*?)\s*A:\s*(\S+)", mod)
    if mm:
        # "Curso - Taller", "Curso taller", "Curso – Taller", "Curso-Talles" -> "Curso-Taller"
        m_txt = re.sub(r"(?i)^curso\s*[-–]?\s*tall\w*$", "Curso-Taller", mm.group(1))
        p.modalidad = f"{m_txt}, {mm.group(2)}"
    else:
        p.modalidad = ""
        p.revisar.append("modalidad_no_encontrada")
    mo = re.search(r"(\S+)\s*$", mod)
    p.oportunidades_evaluacion = mo.group(1) if mo else ""

    pre = _between(text, r"15\.\s*EE prerrequisito\(s\)", r"16\.\s*Organizaci")
    p.prerrequisitos_texto = _clean(pre or "")
    if pre is None:
        p.revisar.append("prerrequisitos_no_encontrados")
    elif not p.prerrequisitos_texto:
        p.revisar.append("prerrequisitos_vacio")

    m = re.search(r"M[áa]ximo\s+M[íi]nimo\s+(\d+)\s+(\d+)", text)
    if m:
        p.cupo_max, p.cupo_min = int(m.group(1)), int(m.group(2))
    return p


def _matcher(programs: list[EEProgram]):
    """Busca un programa por nombre: igual (sin acentos ni mayúsculas) o muy parecido."""
    by_name = {fold(p.nombre): p for p in programs if p.nombre}
    names = list(by_name)

    def numerals(f: str) -> set[str]:
        return set(re.findall(r"\b(?:[ivx]+|\d+)\b", f))

    def match(txt: str) -> Optional[EEProgram]:
        f = fold(txt)
        if f in by_name:
            return by_name[f]
        # "Lengua II" se parece mucho a "Lengua I": el parecido no vale si cambia el número.
        close = [c for c in difflib.get_close_matches(f, names, n=3, cutoff=0.85)
                 if numerals(c) == numerals(f)]
        return by_name[close[0]] if close else None
    return match


def apply_validations(programs: list[EEProgram], validations: dict) -> list[dict]:
    """Aplica la revisión manual (data/structured/validaciones_ee.yaml).

    Modifica los programas (prerrequisitos confirmados, marcas revisadas, notas) y devuelve
    las filas de seriación de tipo "recomendada". Una validación que no corresponde a ningún
    programa, o una recomendación desconocida, detiene el proceso: mejor fallar que perder
    una corrección en silencio.
    """
    match = _matcher(programs)
    rows = []
    for nombre, v in (validations or {}).items():
        p = match(nombre)
        if p is None:
            raise ValueError(f"validaciones_ee.yaml: no hay programa llamado {nombre!r}")
        v = v or {}
        if v.get("nombre"):  # corrige el nombre (y con él la clave), p. ej. una errata del PDF
            p.nombre = v["nombre"]
            match = _matcher(programs)
        if "prerrequisitos" in v:
            p.prerrequisitos_texto = "; ".join(v["prerrequisitos"]) or "Ninguno"
            p.revisar = [r for r in p.revisar if not r.startswith("prerrequisito")]
        confirmed = set(v.get("confirmado", []))
        p.revisar = [r for r in p.revisar if r not in confirmed]
        if v.get("nota"):
            p.nota = v["nota"]
        if v.get("area"):  # área confirmada: prevalece sobre el PDF y sobre el mapa curricular
            if v["area"] not in set(AREAS.values()):
                raise ValueError(f"validaciones_ee.yaml: área desconocida {v['area']!r} en {nombre!r}")
            p.area = v["area"]
            p.revisar = [r for r in p.revisar if not r.startswith("area_")]
        if "creditos" in v:  # p. ej. EE sin programa PDF, confirmadas a mano
            p.creditos = int(v["creditos"])
            p.revisar = [r for r in p.revisar if "creditos_por_confirmar" not in r]
        for rec in v.get("recomendadas", []):
            req = match(rec)
            if req is None:
                raise ValueError(f"validaciones_ee.yaml: recomendación desconocida {rec!r} en {nombre!r}")
            rows.append({"clave": p.clave, "requiere_clave": req.clave, "tipo": "recomendada",
                         "requiere_texto_original": rec, "revisar": ""})
    return rows


def resolve_prerequisites(programs: list[EEProgram]) -> list[dict]:
    """Convierte el texto de prerrequisitos en filas de seriación (clave -> requiere_clave).

    Se busca cada prerrequisito entre los nombres de los demás programas: igual (sin acentos
    ni mayúsculas) o muy parecido. Lo que no se encuentra queda marcado para revisión.
    """
    match = _matcher(programs)
    rows = []
    for p in programs:
        txt = p.prerrequisitos_texto
        if not txt or fold(txt) in SIN_PRERREQUISITO:
            continue
        # Primero el texto completo ("Algoritmos y Programación" lleva " y "); si no, por comas.
        whole = match(txt)
        parts = [txt] if whole else [s for s in re.split(r"\s*[,;]\s*", txt) if s]
        for part in parts:
            req = whole or match(part)
            row = {"clave": p.clave, "requiere_clave": req.clave if req else "",
                   "tipo": "previa", "requiere_texto_original": part, "revisar": ""}
            if not req:
                row["revisar"] = "prerrequisito_no_encontrado_entre_los_programas"
                p.revisar.append(f"prerrequisito_no_encontrado:{part}")
            elif fold(req.nombre) != fold(part):
                row["revisar"] = f"coincidencia_aproximada:{req.nombre}"
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# Cruce con el mapa curricular (Documentos/Extraidos a .MD/mapa_curricular_LISTI.md)
# ---------------------------------------------------------------------------------------------
_MAP_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*(OB|OP)\s*\|\s*`?([a-z_]+)`?\s*\|")
_OPTATIVE_SLOT = re.compile(r"(?i)^optativa\s+[ivx]+$")


@dataclass
class MapEntry:
    nombre: str
    codigo: str
    tipo: str     # OB | OP
    area: str
    periodo: str  # "1".."8"; las optativas del pool: "7-8"


@dataclass
class CurricularMap:
    entries: list[MapEntry]
    period_credits: dict[str, int]  # créditos que el mapa declara por periodo
    optative_slots: dict[str, int]  # huecos "Optativa I/II/III" por periodo


def parse_curricular_map(md: str) -> CurricularMap:
    entries, credits, slots = [], {}, {}
    periodo: Optional[str] = None
    for line in md.splitlines():
        h = re.match(r"^###\s+Periodo\s+(\d+)", line)
        if h:
            periodo = h.group(1)
            continue
        if line.startswith("## "):
            # El pool de optativas no tiene periodo propio: se cursa en los huecos del 7.º y 8.º.
            periodo = "7-8" if re.search(r"(?i)pool.*optativ", line) else None
            continue
        if periodo is None:
            continue
        c = re.match(r"^\|\s*Créditos\s*\|\s*(\d+)\s*\|", line)
        if c and periodo != "7-8":
            credits[periodo] = int(c.group(1))
            continue
        m = _MAP_ROW.match(line)
        if not m or m.group(1) in ("Materia", "Materia optativa"):
            continue
        nombre = m.group(1).strip()
        if _OPTATIVE_SLOT.match(nombre):
            slots[periodo] = slots.get(periodo, 0) + 1
            continue
        entries.append(MapEntry(nombre, m.group(2).strip(), m.group(3), m.group(4), periodo))
    return CurricularMap(entries, credits, slots)


def merge_curricular_map(programs: list[EEProgram], cmap: CurricularMap) -> list[EEProgram]:
    """Añade el periodo sugerido a cada programa y crea las EE del mapa que no tienen PDF.

    - El área y el carácter del programa (PDF) prevalecen; si el mapa dice otra cosa se marca.
    - Las EE del mapa sin programa se agregan con créditos vacíos: el código del mapa no permite
      deducirlos con seguridad. Se marcan para confirmar.
    - Un programa que no aparece en el mapa también se marca.
    """
    match = _matcher(programs)
    used: dict[str, str] = {}
    for e in cmap.entries:
        p = match(e.nombre)
        if p is None:
            p = EEProgram(archivo="", nombre=e.nombre, plan=programs[0].plan if programs else "",
                          area=e.area, caracter=e.tipo)
            p.revisar += ["sin_programa_pdf", f"creditos_por_confirmar(codigo del mapa: {e.codigo})"]
            programs.append(p)
            match = _matcher(programs)
        elif p.clave in used:
            p.revisar.append(f"aparece_dos_veces_en_el_mapa:{used[p.clave]}|{e.nombre}")
            continue
        used[p.clave] = e.nombre
        p.periodo = e.periodo
        if fold(p.nombre) != fold(e.nombre):
            p.nombre_mapa = e.nombre
        if p.area != e.area:
            p.revisar.append(f"area_difiere_del_mapa:mapa={e.area}")
        if p.caracter != e.tipo:
            p.revisar.append(f"caracter_difiere_del_mapa:mapa={e.tipo}")
    for p in programs:
        if p.clave not in used:
            p.revisar.append("no_aparece_en_el_mapa_curricular")
    return programs


def check_period_credits(programs: list[EEProgram], cmap: CurricularMap,
                         optative_credits: int = 6) -> list[str]:
    """Compara los créditos de cada periodo (suma de sus EE) con lo que declara el mapa."""
    out = []
    for periodo, declared in sorted(cmap.period_credits.items()):
        ees = [p for p in programs if p.periodo == periodo]
        unknown = [p.nombre for p in ees if p.creditos is None]
        total = sum(p.creditos or 0 for p in ees) + cmap.optative_slots.get(periodo, 0) * optative_credits
        if unknown:
            out.append(f"Periodo {periodo}: el mapa declara {declared} créditos; las EE conocidas suman "
                       f"{total} y faltan los de {', '.join(unknown)} (deberían sumar {declared - total}).")
        elif total != declared:
            out.append(f"Periodo {periodo}: el mapa declara {declared} créditos y sus EE suman {total}.")
    return out
