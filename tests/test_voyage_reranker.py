"""Reranker de Voyage AI: sin red (se sustituye httpx.post)."""
import httpx
import pytest

from chatbot.adapters.reranker import http_reranker, voyage_reranker
from chatbot.adapters.reranker.voyage_reranker import VoyageReranker
from chatbot.domain.models import Chunk, ScoredChunk

REQ = httpx.Request("POST", voyage_reranker.VOYAGE_URL)
A = ScoredChunk(Chunk("a", "baja temporal hasta cinco veces", "d"), 0.03, 0.64)
B = ScoredChunk(Chunk("b", "derecho a dos inscripciones", "d"), 0.01, 0.59)


@pytest.fixture
def fake_post(monkeypatch):
    calls, queue = [], []

    def post(url, json, headers, timeout):
        calls.append((json, headers))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(http_reranker.httpx, "post", post)
    return calls, queue


def test_ordena_por_puntaje_y_conserva_relevancia(fake_post):
    calls, queue = fake_post
    queue.append(httpx.Response(200, request=REQ, json={"object": "list", "data": [
        {"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.12}]}))
    out = VoyageReranker("rerank-2.5", api_key="k")("¿cuántas veces puedo cursar?", [A, B])
    assert [o.chunk.id for o in out] == ["b", "a"]
    assert out[0].score == 0.91 and out[0].relevance == 0.59
    body, headers = calls[0]
    assert body["documents"] == [A.chunk.text, B.chunk.text] and body["model"] == "rerank-2.5"
    assert headers["Authorization"] == "Bearer k"


def test_error_de_la_api_conserva_el_orden_de_la_fusion(fake_post):
    _, queue = fake_post
    queue.append(httpx.Response(429, request=REQ))
    assert VoyageReranker(api_key="k")("q", [A, B]) == [A, B]


def test_sin_red_conserva_el_orden(fake_post):
    _, queue = fake_post
    queue.append(httpx.ConnectError("sin red", request=REQ))
    assert VoyageReranker(api_key="k")("q", [A, B]) == [A, B]


def test_sin_candidatos_no_llama_a_la_api(fake_post):
    calls, _ = fake_post
    assert VoyageReranker(api_key="k")("q", []) == [] and not calls


def test_sin_clave_falla_al_arrancar(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageReranker()
