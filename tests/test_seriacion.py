"""RNF1: la seriación debe ser 100 % exacta, funcione o no el LLM."""
from conftest import FakeLLM

PREGUNTA = "¿Qué requisitos tiene DEMO-202?"


def test_fallback_determinista_si_el_llm_cae(build):
    ans = build(FakeLLM(fail=True)).answer(PREGUNTA)
    assert ans.decision.value == "answered"
    assert "DEMO-201" in ans.text and "DEMO-102" in ans.text


def test_respuesta_del_llm_con_dato_inventado_se_descarta(build):
    llm = FakeLLM("DEMO-202 requiere DEMO-999 y 40 créditos.")
    ans = build(llm).answer(PREGUNTA)
    assert "DEMO-999" not in ans.text and "DEMO-201" in ans.text
    assert ans.trace.verification_problems


def test_respuesta_correcta_del_llm_se_conserva(build):
    llm = FakeLLM("Ingeniería de Software (EJEMPLO) (DEMO-202) requiere DEMO-201 y DEMO-102.")
    ans = build(llm).answer(PREGUNTA)
    assert ans.text.startswith("Ingeniería de Software") and not ans.trace.verification_problems


def test_cada_asignatura_del_ejemplo(build, repo):
    pipe = build(FakeLLM(fail=True))
    esperado = {"DEMO-102": ["DEMO-101"], "DEMO-201": ["DEMO-102"],
                "DEMO-202": ["DEMO-201", "DEMO-102"]}
    for clave, reqs in esperado.items():
        text = pipe.answer(f"requisitos de {clave}").text
        assert all(r in text for r in reqs)


def test_respuesta_vacia_del_llm_usa_el_fallback(build):
    # Bug 3: un modelo de razonamiento puede agotar max_tokens y devolver "".
    ans = build(FakeLLM("")).answer(PREGUNTA)
    assert ans.decision.value == "answered"
    assert "DEMO-201" in ans.text and "DEMO-102" in ans.text
    assert "respuesta_vacia" in ans.trace.verification_problems
