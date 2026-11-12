from chatbot.adapters.lexical.bm25_index import BM25Index
from chatbot.domain.models import Chunk, ScoredChunk
from chatbot.ingestion.chunker import chunk_pages
from chatbot.ingestion.loaders import Page
from chatbot.retrieval.hybrid import HybridRetriever, rrf_fuse


def test_rrf_premia_coincidencia_en_ambos_rankings():
    s = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]])
    assert s["a"] > s["b"] and s["a"] > s["d"]


def test_bm25_encuentra_clave_exacta_y_filtra_obsoletos():
    chunks = [Chunk("1", "La asignatura DEMO-201 se cursa en tercer periodo", "d"),
              Chunk("2", "Texto sobre becas y apoyos", "d"),
              Chunk("3", "DEMO-201 versión anterior", "d", vigente=False)]
    hits = BM25Index(chunks).search("DEMO-201", 5)
    assert [h.chunk.id for h in hits] == ["1"]


class _Emb:
    model_id = "fake"
    def embed_query(self, t): return [0.0]
    def embed_documents(self, ts): return [[0.0]] * len(ts)


class _Vec:
    def __init__(self, hits): self.hits = hits
    def search(self, v, k): return self.hits


class _Lex(_Vec):
    def search(self, q, k): return self.hits


def test_hibrido_conserva_relevancia_vectorial_y_descarta_obsoletos():
    a, b = Chunk("a", "x", "d"), Chunk("b", "y", "d", vigente=False)
    vec = [ScoredChunk(a, 0.9, 0.9), ScoredChunk(b, 0.8, 0.8)]
    lex = [ScoredChunk(a, 5.0, 0.0)]
    out = HybridRetriever(_Emb(), _Vec(vec), _Lex(lex), k=5).retrieve("q")
    assert [o.chunk.id for o in out] == ["a"] and out[0].relevance == 0.9


def test_chunker_respeta_articulos_y_solape():
    texto = "Artículo 1\n" + "palabra " * 300 + "\nArtículo 2\nTexto corto del segundo."
    chunks = chunk_pages([Page(1, texto)], size_tokens=100, overlap_pct=10)
    assert any(c.section.startswith("Artículo 2") for c in chunks)
    largos = [c for c in chunks if c.section.startswith("Artículo 1")]
    assert len(largos) > 1
