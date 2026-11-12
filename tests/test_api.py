from fastapi import FastAPI
from fastapi.testclient import TestClient

from chatbot.api.routes import router
from conftest import FakeLLM


def _client(pipeline):
    app = FastAPI()
    app.include_router(router)
    app.state.pipeline = pipeline
    return TestClient(app)


def test_health_y_503_sin_pipeline():
    c = _client(None)
    assert c.get("/health").json()["pipeline_ready"] is False
    assert c.post("/chat", json={"message": "hola"}).status_code == 503


def test_chat_devuelve_fuentes(build):
    c = _client(build(FakeLLM("Hasta 2 veces [1].")))
    r = c.post("/chat", json={"message": "¿Cuántas veces puedo presentar examen?"})
    body = r.json()
    assert r.status_code == 200 and body["decision"] == "answered"
    assert body["sources"][0]["label"] == "Reglamento" and body["session_id"]


def test_valida_longitud():
    assert _client(None).post("/chat", json={"message": ""}).status_code == 422
