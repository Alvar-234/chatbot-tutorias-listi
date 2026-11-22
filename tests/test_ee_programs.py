"""Extracción de datos generales de programas de EE (texto como lo entrega pypdf)."""
from chatbot.ingestion.ee_programs import parse_program, resolve_prerequisites

BLOQUE = (
    "Programa de experiencia educativa \nLicenciatura en Ingeniería en Sistemas y Tecnologías de la Información, 2023 \n"
    "5. Código  6. Nombre de la Experiencia \nEducativa \n {nombre} \n \n7. Área de Formación del Modelo \n"
    "Educativo Institucional 8. Carácter  \n{area} \n \n9. Agrupación curricular distintiva  \n"
    "Gestión y Análisis de Datos \n \n10. Valores \nHoras  \nTeóricas \nHoras \nPrácticas \nHoras \nOtras \n"
    "Total de \nhoras  Créditos Equivalencia (s)  \n{valores} \n \n \n11. Modalidad y \nambiente de \n"
    "aprendizaje  \n12. Espacio  13. Relación \ndisciplinaria \n14. Oportunidades \nde evaluación \n"
    "M: Curso-\nTaller \nA: \nPresencial \nIaF: Intrafacultad Interdisciplinaria Todas \n    \n"
    "15. EE prerrequisito(s) \n{pre} \n\n \n \n 2 \n16.  Organización de los estudiantes en el proceso de aprendizaje \n"
    "Máximo Mínimo \n30 10 \n")


def bloque(nombre="Bases de Datos Relacionales", area="Disciplinar Obligatoria",
           valores="3 3 0 90 9 Bases de datos (Plan 2014)", pre="Ninguno"):
    return BLOQUE.format(nombre=nombre, area=area, valores=valores, pre=pre)


def test_lee_los_datos_generales():
    p = parse_program(bloque())
    assert (p.nombre, p.area, p.caracter, p.plan) == ("Bases de Datos Relacionales", "disciplinar", "OB", "LISTI 2023")
    assert (p.horas_teoricas, p.horas_practicas, p.horas_otras, p.horas_totales, p.creditos) == (3, 3, 0, 90, 9)
    assert p.equivalencia == "Bases de datos (Plan 2014)" and p.modalidad == "Curso-Taller, Presencial"
    assert (p.cupo_max, p.cupo_min, p.oportunidades_evaluacion) == (30, 10, "Todas")
    assert p.clave == "bases-de-datos-relacionales" and not p.revisar


def test_numero_de_pagina_suelto_no_se_toma_como_prerrequisito():
    # Caso real (Procesos de Desarrollo de Software): campo vacío y un "2" de número de página.
    p = parse_program(bloque(pre=""))
    assert p.prerrequisitos_texto == "" and "prerrequisitos_vacio" in p.revisar


def test_horas_o_creditos_que_no_cuadran_se_marcan_sin_corregir():
    p = parse_program(bloque(valores="2 2 0 75 6 Ninguna"))  # (2+2) x 15 = 60, no 75
    assert p.horas_totales == 75 and "horas_totales_no_cuadran" in p.revisar
    assert p.equivalencia == ""


def test_area_de_varias_palabras_y_optativa():
    p = parse_program(bloque(area="Iniciación a la Disciplina \nOptativa"))
    assert (p.area, p.caracter) == ("iniciacion_disciplina", "OP")


def test_prerrequisito_con_y_en_el_nombre_se_resuelve_completo():
    progs = [parse_program(bloque(nombre="Algoritmos y Programación")),
             parse_program(bloque(nombre="Programación", pre="Algoritmos y Programación"))]
    (fila,) = resolve_prerequisites(progs)
    assert fila["clave"] == "programacion" and fila["requiere_clave"] == "algoritmos-y-programacion"
    assert fila["revisar"] == ""


def test_prerrequisito_desconocido_queda_para_revision():
    progs = [parse_program(bloque(nombre="Redes Aplicadas", pre="Redes Avanzadas Inexistentes"))]
    (fila,) = resolve_prerequisites(progs)
    assert fila["requiere_clave"] == "" and fila["revisar"]
    assert any("prerrequisito_no_encontrado" in r for r in progs[0].revisar)


def test_encabezado_partido_y_variantes_de_curso_taller():
    txt = bloque().replace("14. Oportunidades \nde evaluación", "14. Oportunidade\ns de \nevaluación") \
                  .replace("M: Curso-\nTaller", "M: Curso - \nTaller")
    assert parse_program(txt).modalidad == "Curso-Taller, Presencial"


def test_validaciones_confirman_vacios_y_agregan_recomendaciones():
    import pytest
    from chatbot.ingestion.ee_programs import apply_validations
    progs = [parse_program(bloque(nombre="Networking")),
             parse_program(bloque(nombre="Administración de Servidores", pre="")),
             parse_program(bloque(nombre="Servicio Social", valores="0 4 0 60 12 Ninguna"))]
    assert "prerrequisitos_vacio" in progs[1].revisar
    recs = apply_validations(progs, {
        "Administración de Servidores": {"prerrequisitos": [], "recomendadas": ["Networking"]},
        "Servicio Social": {"confirmado": ["creditos_no_cuadran_con_horas"], "nota": "480 horas"}})
    assert recs == [{"clave": "administracion-de-servidores", "requiere_clave": "networking",
                     "tipo": "recomendada", "requiere_texto_original": "Networking", "revisar": ""}]
    assert not progs[1].revisar and progs[1].prerrequisitos_texto == "Ninguno"
    assert not progs[2].revisar and progs[2].nota == "480 horas"
    assert resolve_prerequisites(progs) == []  # "Ninguno" no genera seriación
    with pytest.raises(ValueError):
        apply_validations(progs, {"Materia Inexistente": {}})


def test_recomendacion_no_se_presenta_como_requisito():
    from chatbot.domain.models import Prerequisite, SubjectFact
    from chatbot.domain.prompts import facts_text
    f = SubjectFact("administracion-de-servidores", "Administración de Servidores", 8, "6",
                    "disciplinar", "LISTI 2023",
                    (Prerequisite("networking", "Networking", "recomendada"),))
    txt = facts_text([f])
    assert "no tiene requisitos de seriación" in txt
    assert "Recomendación (NO es requisito obligatorio)" in txt and "Networking" in txt


MAPA = """# Mapa
### Periodo 1
| Dato | Valor |
|---|---:|
| Créditos | 20 |
| Materia | Código | Tipo | Área |
|---|---|---|---|
| Lengua I | BG 0064 | OB | `basica` |
| Networking | FD 3309 | OB | `iniciacion_disciplina` |
### Periodo 2
| Créditos | 15 |
| Lengua II | BG 0064 | OB | `basica` |
| Optativa I | FT 2206 | OP | `terminal` | Slot a llenar desde el pool de optativas |
## 4. Pool general de materias optativas
| Materia optativa | Código | Tipo | Área |
| Redes Aplicadas | FT 2206 | OP | `terminal` |
## 5. Leyenda de áreas
"""


def test_mapa_curricular_periodos_huecos_y_pool():
    from chatbot.ingestion.ee_programs import parse_curricular_map
    cmap = parse_curricular_map(MAPA)
    assert [(e.nombre, e.periodo) for e in cmap.entries] == [
        ("Lengua I", "1"), ("Networking", "1"), ("Lengua II", "2"), ("Redes Aplicadas", "7-8")]
    assert cmap.period_credits == {"1": 20, "2": 15} and cmap.optative_slots == {"2": 1}


def test_cruce_con_el_mapa():
    from chatbot.ingestion.ee_programs import (check_period_credits, merge_curricular_map,
                                               parse_curricular_map)
    progs = [parse_program(bloque(nombre="Networking", valores="3 3 0 90 9 Ninguna")),
             parse_program(bloque(nombre="Redes Aplicadas", area="Terminal Optativa",
                                  valores="2 2 0 60 6 Ninguna")),
             parse_program(bloque(nombre="Programa Fuera del Mapa"))]
    cmap = parse_curricular_map(MAPA)
    merge_curricular_map(progs, cmap)
    por_nombre = {p.nombre: p for p in progs}
    # "Lengua II" NO se empareja con "Lengua I" aunque se parezcan: son dos EE sin programa.
    assert por_nombre["Lengua I"].periodo == "1" and por_nombre["Lengua II"].periodo == "2"
    assert "sin_programa_pdf" in por_nombre["Lengua II"].revisar and por_nombre["Lengua II"].creditos is None
    net = por_nombre["Networking"]
    assert net.periodo == "1" and net.area == "disciplinar"  # vale el programa, pero se marca
    assert "area_difiere_del_mapa:mapa=iniciacion_disciplina" in net.revisar
    assert por_nombre["Redes Aplicadas"].periodo == "7-8"
    assert "no_aparece_en_el_mapa_curricular" in por_nombre["Programa Fuera del Mapa"].revisar
    avisos = check_period_credits(progs, cmap)
    assert any("Periodo 1" in a and "Lengua I" in a for a in avisos)
    assert any("Periodo 2" in a for a in avisos)  # 6 de la optativa + Lengua II desconocida
