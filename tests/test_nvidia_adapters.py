"""Adaptadores de NVIDIA (embeddings, reranker, LLM): sin red (se sustituye httpx.post)."""
import httpx
import pytest

from chatbot.adapters.embeddings import nvidia_embedder
from chatbot.adapters.embeddings.factory import collection_name
from chatbot.adapters.embeddings.nvidia_embedder import NvidiaEmbedder
from chatbot.adapters.llm.factory import build_llm_chain
from chatbot.adapters.reranker import http_reranker
from chatbot.adapters.reranker.nvidia_reranker import NvidiaReranker
from chatbot.domain.models import Chunk, ScoredChunk

REQ = httpx.Request("POST", "https://x")
A = ScoredChunk(Chunk("a", "baja temporal hasta cinco veces", "d"), 0.03, 0.64)
B = ScoredChunk(Chunk("b", "derecho a dos inscripciones", "d"), 0.01, 0.59)


@pytest.fixture
def fake_post(monkeypatch):
    calls, queue, sleeps = [], [], []

    def post(url, json, headers, timeout):
        calls.append((url, json))
        return queue.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(nvidia_embedder.time, "sleep", sleeps.append)
    return calls, queue, sleeps


def emb_response(vectors):
    # NVIDIA puede devolver los vectores desordenados: se reordenan por `index`.
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)][::-1]
    return httpx.Response(200, request=REQ, json={"object": "list", "data": data})


def test_embeddings_distinguen_consulta_y_pasaje_y_van_por_lotes(fake_post):
    calls, queue, _ = fake_post
    queue.extend([emb_response([[1.0], [2.0]]), emb_response([[3.0]]), emb_response([[9.0]])])
    e = NvidiaEmbedder("m", api_key="k", batch_size=2)
    assert e.embed_documents(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]
    assert e.embed_query("q") == [9.0]
    assert [c[1]["input_type"] for c in calls] == ["passage", "passage", "query"]
    assert [len(c[1]["input"]) for c in calls] == [2, 1, 1]


def test_embeddings_reintentan_ante_429(fake_post):
    calls, queue, sleeps = fake_post
    queue.extend([httpx.Response(429, request=REQ, headers={"retry-after": "1"}),
                  emb_response([[1.0]])])
    assert NvidiaEmbedder("m", api_key="k").embed_query("q") == [1.0]
    assert sleeps == [1.0] and len(calls) == 2


def test_reranker_nvidia_usa_logit_y_url_del_modelo(fake_post):
    calls, queue, _ = fake_post
    queue.append(httpx.Response(200, request=REQ, json={"rankings": [
        {"index": 1, "logit": -1.79}, {"index": 0, "logit": -1.96}]}))
    out = NvidiaReranker("nvidia/nemotron-3.5-rerank-8b", api_key="k")("q", [A, B])
    assert [o.chunk.id for o in out] == ["b", "a"] and out[0].relevance == 0.59
    url, body = calls[0]
    assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking"
    assert body["query"] == {"text": "q"} and body["passages"][1] == {"text": B.chunk.text}


def test_reranker_nvidia_retirado_conserva_el_orden(fake_post):
    _, queue, _ = fake_post
    queue.append(httpx.Response(410, request=REQ))
    assert NvidiaReranker(api_key="k")("q", [A, B]) == [A, B]


def test_sin_clave_falla_al_arrancar(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NvidiaEmbedder()
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NvidiaReranker()


def test_cada_proveedor_de_embeddings_tiene_su_coleccion():
    assert collection_name({}) == "listi"  # el índice local existente no cambia
    assert collection_name({"provider": "nvidia"}) == "listi-nvidia"


def test_fabrica_de_llm_conoce_nvidia():
    (nv,) = build_llm_chain({"chain": [{"provider": "nvidia", "model": "m"}]})
    assert nv._url.startswith("https://integrate.api.nvidia.com/v1")
