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
