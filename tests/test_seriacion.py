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


def test_llm_que_omite_un_requisito_usa_el_fallback(build):
    # Bug 4: "DEMO-202 requiere DEMO-201" omite DEMO-102 y antes se entregaba tal cual.
    ans = build(FakeLLM("Ingeniería de Software (EJEMPLO) requiere DEMO-201.")).answer(PREGUNTA)
    assert "DEMO-102" in ans.text and "requisito_omitido:DEMO-102" in ans.trace.verification_problems


def test_recomendacion_omitida_se_agrega_al_final(handoff):
    import sqlite3
    from conftest import FakeRetriever
    from chatbot.adapters.structured.sqlite_repo import SqliteStructuredRepo, init_db
    from chatbot.domain.pipeline import ChatPipeline, PipelineSettings
    from chatbot.domain.router import Router
    from chatbot.domain.sufficiency import SufficiencyChecker
    from chatbot.domain.verifier import AnswerVerifier
    from chatbot.privacy.redactor import Redactor
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    conn.executemany("INSERT INTO asignatura (clave, nombre, creditos, periodo, area, plan_vigente) "
                     "VALUES (?,?,?,?,?,?)", [
        ("networking", "Networking", 9, "3", "disciplinar", "LISTI 2023"),
        ("administracion-de-servidores", "Administración de Servidores", 8, "4", "disciplinar", "LISTI 2023")])
    conn.execute("INSERT INTO seriacion VALUES ('administracion-de-servidores', 'networking', 'recomendada')")
    repo = SqliteStructuredRepo(conn=conn)
    pipe = ChatPipeline(router=Router(handoff, repo), retriever=FakeRetriever([]), repo=repo,
                        llms=[FakeLLM("Administración de Servidores no tiene requisitos de seriación.")],
                        redactor=Redactor(), sufficiency=SufficiencyChecker(0.35),
                        verifier=AnswerVerifier(), handoff=handoff, settings=PipelineSettings())
    ans = pipe.answer("¿Qué necesito para cursar Administración de Servidores?")
    assert ans.route.value == "structured"
    assert ans.text.endswith("Recomendación (no es requisito obligatorio): cursar Networking "
                             "antes de Administración de Servidores.")


def test_pregunta_solo_de_creditos_agrega_el_requisito(build):
    ans = build(FakeLLM("Bases de Datos (EJEMPLO) tiene 6 créditos.")).answer(
        "¿Cuántos créditos tiene DEMO-201?")
    assert ans.text.startswith("Bases de Datos (EJEMPLO) tiene 6 créditos.")
    assert ans.text.endswith("Requisito de seriación: haber cursado Estructuras de Datos (EJEMPLO) (previa).")


def test_negar_requisitos_que_existen_usa_el_fallback(build):
    ans = build(FakeLLM("DEMO-202 no tiene requisitos.")).answer(PREGUNTA)
    assert "DEMO-201" in ans.text and "DEMO-102" in ans.text


def test_nota_validada_omitida_se_agrega():
    from chatbot.domain.models import SubjectFact
    from chatbot.domain.prompts import omitted_notes
    ss = SubjectFact("servicio-social", "Servicio Social", 12, "7", "terminal", "LISTI 2023",
                     horas_totales=60, nota="480 horas comprobables por medio de reportes mensuales.")
    assert omitted_notes("Servicio Social tiene 60 horas en total.", [ss]) == [ss]
    assert omitted_notes("Son 480 horas comprobables con reportes mensuales.", [ss]) == []
