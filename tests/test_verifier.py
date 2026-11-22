from chatbot.domain.verifier import AnswerVerifier

v = AnswerVerifier()
SRC = ["El examen puede presentarse hasta 2 veces. Clave DEMO-101."]


def test_respuesta_valida():
    assert v.verify("Puedes presentarlo hasta 2 veces [1].", SRC).ok


def test_cita_inexistente():
    assert "cita_inexistente:[3]" in v.verify("Son 2 veces [3].", SRC).problems


def test_sin_citas():
    assert "sin_citas" in v.verify("Son 2 veces.", SRC).problems


def test_cifra_inventada():
    assert any(p.startswith("cifra_sin_respaldo") for p in v.verify("Son 5 veces [1].", SRC).problems)


def test_clave_inventada():
    assert any(p.startswith("clave_sin_respaldo") for p in v.verify("Ver DEMO-999 [1].", SRC).problems)


def test_numeracion_de_lista_no_cuenta():
    assert v.verify("1. Hasta 2 veces [1].\n2. Clave DEMO-101 [1].", SRC).ok


def test_respuesta_vacia_no_es_valida():
    for texto in ("", "   \n"):
        r = v.verify(texto, SRC, require_citations=False)
        assert not r.ok and r.problems == ["respuesta_vacia"]


def test_normaliza_variantes_de_cita():
    from chatbot.domain.verifier import normalize_citations
    assert normalize_citations("Lista【1】 y ［2］.") == "Lista[1] y [2]."
    assert normalize_citations("Ver [1, 3] o [2; 4].") == "Ver [1][3] o [2][4]."
    assert normalize_citations("Sin cita [nota].") == "Sin cita [nota]."


def test_citas_escritas_con_palabras():
    from chatbot.domain.verifier import normalize_citations
    assert normalize_citations("No. Según la fuente 1, el artículo 125 lo prohíbe.") == \
        "No. Según [1], el artículo 125 lo prohíbe."
    assert normalize_citations("Hasta dos veces (fuentes 2 y 3).") == "Hasta dos veces [2][3]."
    assert normalize_citations("Dos inscripciones [1, 2].") == "Dos inscripciones [1][2]."
