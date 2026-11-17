"""Cliente OpenAI-compatible: sin red (se sustituye httpx.post)."""
import httpx
import pytest

from chatbot.adapters.llm import openai_compat
from chatbot.adapters.llm.factory import build_llm_chain
from chatbot.adapters.llm.openai_compat import OpenAICompatProvider
from chatbot.domain.ports import ProviderError

REQ = httpx.Request("POST", "https://x/chat/completions")


def ok(text="Hola [1].", finish="stop"):
    return httpx.Response(200, request=REQ, json={
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


def limited(retry_after="2"):
    return httpx.Response(429, request=REQ, headers={"retry-after": retry_after})


@pytest.fixture
def fake_http(monkeypatch):
    calls, sleeps, queue = [], [], []

    def post(url, json, headers, timeout):
        calls.append(json)
        return queue.pop(0)

    monkeypatch.setattr(openai_compat.httpx, "post", post)
    monkeypatch.setattr(openai_compat.time, "sleep", sleeps.append)
    return calls, sleeps, queue


def provider(**kw):
    return OpenAICompatProvider(name="t", base_url="https://x", model="m", **kw)


def test_respuesta_normal(fake_http):
    _, _, queue = fake_http
    queue.append(ok())
    assert provider().generate([]).text == "Hola [1]."


def test_respuesta_truncada_es_error(fake_http):
    _, _, queue = fake_http
    queue.append(ok("1. Primer paso 2. **Presen", finish="length"))
    with pytest.raises(ProviderError, match="truncada"):
        provider().generate([])


def test_429_con_espera_corta_reintenta(fake_http):
    calls, sleeps, queue = fake_http
    queue.extend([limited("2"), ok()])
    assert provider(max_retry_wait_s=10).generate([]).text == "Hola [1]."
    assert sleeps == [2.0] and len(calls) == 2


def test_429_con_espera_larga_pasa_al_siguiente(fake_http):
    calls, sleeps, queue = fake_http
    queue.append(limited("40"))
    with pytest.raises(ProviderError):
        provider(max_retry_wait_s=10).generate([])
    assert not sleeps and len(calls) == 1


def test_parametros_extra_se_envian(fake_http):
    calls, _, queue = fake_http
    queue.append(ok())
    provider(extra_body={"reasoning_effort": "low"}).generate([], max_tokens=1500)
    assert calls[0]["reasoning_effort"] == "low" and calls[0]["max_tokens"] == 1500


def test_fabrica_lee_extra_y_espera():
    groq, local = build_llm_chain({"retry_429_max_wait_s": 7, "chain": [
        {"provider": "groq", "model": "g", "extra": {"reasoning_effort": "low"}},
        {"provider": "local", "model": "llama3.2"}]})
    assert groq._extra == {"reasoning_effort": "low"} and groq._max_wait == 7
    assert groq._url.startswith("https://api.groq.com") and local._url.startswith("http://localhost:11434")
