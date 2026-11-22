import sqlite3
from pathlib import Path

import pytest

from chatbot.adapters.structured.sqlite_repo import SqliteStructuredRepo, init_db, load_csv_dir
from chatbot.domain.handoff import HandoffRules
from chatbot.domain.models import Chunk, ScoredChunk
from chatbot.domain.pipeline import ChatPipeline, PipelineSettings
from chatbot.domain.ports import ProviderError
from chatbot.domain.models import LLMResponse
from chatbot.domain.router import Router
from chatbot.domain.sufficiency import SufficiencyChecker
from chatbot.domain.verifier import AnswerVerifier
from chatbot.privacy.redactor import Redactor

SAMPLE = Path(__file__).resolve().parents[1] / "data/structured/sample"
TOPICS = {
    "risk": ["suicid"], "sensitive": ["ansiedad", "baja temporal"],
    "requested": ["hablar con mi tutor"], "administrative": ["inscribeme"],
}


class FakeLLM:
    def __init__(self, text="", fail=False):
        self.name, self.text, self.fail, self.calls = "fake", text, fail, []

    def generate(self, messages, *, temperature=0.0, max_tokens=600):
        self.calls.append(messages)
        if self.fail:
            raise ProviderError("caído")
        # text puede ser una lista: una respuesta por llamada (la última se repite)
        text = self.text[min(len(self.calls), len(self.text)) - 1] if isinstance(self.text, list) else self.text
        return LLMResponse(text, "fake-model", "fake", 10, 5)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query):
        return self.chunks


def make_chunk(cid="reg-0001", text="El estudiante puede presentar el examen hasta 2 veces.",
               rel=0.8):
    return ScoredChunk(Chunk(id=cid, text=text, doc_id="reg", doc_title="Reglamento",
                             valid_from="2023-08-01"), 0.03, rel)


@pytest.fixture
def repo():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    load_csv_dir(conn, SAMPLE)
    return SqliteStructuredRepo(conn=conn)


@pytest.fixture
def handoff():
    return HandoffRules(TOPICS, {"tutoria": "tutorias@ejemplo.mx"})


@pytest.fixture
def build(repo, handoff):
    def _build(llm, chunks=None, **kw):
        return ChatPipeline(
            router=Router(handoff, repo, ["horoscopo"]),
            retriever=FakeRetriever(chunks if chunks is not None else [make_chunk()]),
            repo=repo, llms=llm if isinstance(llm, list) else [llm], redactor=Redactor(),
            sufficiency=SufficiencyChecker(0.35, ["LISTI"]), verifier=AnswerVerifier(),
            handoff=handoff, settings=PipelineSettings(**kw))
    return _build
