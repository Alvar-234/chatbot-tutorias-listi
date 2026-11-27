from conftest import FakeLLM
from chatbot.domain.followup import is_follow_up
from chatbot.domain.models import Route, Session


def test_detecta_seguimientos():
    assert is_follow_up("y general")
    assert is_follow_up("¿Y cuántas horas tiene?")
    assert is_follow_up("¿optativas?")
    assert not is_follow_up("¿Cuántos créditos tiene la carrera?")


def test_seguimiento_busca_con_la_consulta_anterior(build):
    pipe = build(FakeLLM("Puedes presentarlo hasta 2 veces [1]."))
    s = Session("s-seg")
    pipe.answer("¿Cuántas veces puedo presentar examen?", s)
    ans = pipe.answer("y extraordinario", s)
    consultas = pipe.retriever.queries
    assert consultas[-1] == "¿Cuántas veces puedo presentar examen? y extraordinario"
    assert ans.trace.follow_up
    # al LLM le llega la pregunta tal cual, con el historial
    assert "Pregunta: y extraordinario" in pipe.llms[0].calls[-1][-1]["content"]


def test_pregunta_completa_no_se_contextualiza(build):
    pipe = build(FakeLLM("Puedes presentarlo hasta 2 veces [1]."))
    s = Session("s-noseg")
    pipe.answer("¿Cuántas veces puedo presentar examen?", s)
    ans = pipe.answer("¿Cuál es el plazo para la baja temporal de una materia?", s)
    assert not ans.trace.follow_up


def test_seguimiento_de_materia_va_a_la_base_estructurada(build):
    pipe = build(FakeLLM("Estructuras de Datos (EJEMPLO) tiene 8 créditos."))
    s = Session("s-mat")
    pipe.answer("¿De qué trata DEMO-102?", s)
    ans = pipe.answer("¿y cuántos créditos tiene?", s)
    assert ans.route == Route.STRUCTURED and ans.trace.follow_up
