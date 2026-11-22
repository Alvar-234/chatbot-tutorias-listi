"""Suficiencia: umbral del reranker cuando lo hay; vectorial cuando no."""
from chatbot.domain.models import Chunk, ScoredChunk
from chatbot.domain.sufficiency import SufficiencyChecker

TXT = "Los alumnos tienen derecho a dos inscripciones por experiencia educativa."


def sc(rel, rr=None):
    return ScoredChunk(Chunk("c", TXT, "d"), 0.0, rel, rr)


def test_reranker_alto_rescata_similitud_vectorial_baja():
    # "cuantas mareias hay": la errata hunde la similitud vectorial, no al reranker.
    res = SufficiencyChecker(0.35, rerank_threshold=-2.9).check("q", [sc(0.12, -2.7)])
    assert res.sufficient and res.max_rerank_score == -2.7


def test_reranker_bajo_se_abstiene_aunque_la_similitud_sea_alta():
    res = SufficiencyChecker(0.35, rerank_threshold=-2.9).check("q", [sc(0.60, -3.4)])
    assert not res.sufficient and res.reason == "relevancia_baja"


def test_si_el_reranker_fallo_se_usa_la_similitud_vectorial():
    # La API del reranker cayó: los fragmentos llegan sin rerank_score.
    checker = SufficiencyChecker(0.35, rerank_threshold=-2.9)
    assert not checker.check("q", [sc(0.12)]).sufficient
    assert checker.check("q", [sc(0.50)]).sufficient


def test_reranker_sin_umbral_calibrado_usa_la_similitud_vectorial():
    assert not SufficiencyChecker(0.35).check("q", [sc(0.12, 5.0)]).sufficient
