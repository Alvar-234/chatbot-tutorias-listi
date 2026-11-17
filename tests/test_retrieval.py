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


def test_chunker_respeta_encabezados_markdown():
    from chatbot.ingestion.chunker import chunk_pages
    from chatbot.ingestion.loaders import Page
    md = "# Mapa\n\n### Periodo 1\n| Lengua I | BG 0064 |\n\n### Periodo 2\n| Lengua II | BG 0064 |"
    chunks = chunk_pages([Page(1, md)])
    by_section = {c.section: c.text for c in chunks}
    assert "Lengua I |" in by_section["Periodo 1"] and "Lengua II" not in by_section["Periodo 1"]
    assert "Lengua II" in by_section["Periodo 2"]


def test_chunker_empaqueta_articulos_cortos():
    from chatbot.ingestion.chunker import chunk_pages
    from chatbot.ingestion.loaders import Page
    texto = ("Capítulo I De la inscripción\n"
             "Artículo 24. Texto breve del artículo veinticuatro.\n"
             "Artículo 25. Los alumnos tienen derecho a dos inscripciones por experiencia educativa.\n"
             "Artículo 26. Otro texto breve.\n"
             "Capítulo II De las bajas\n"
             "Artículo 27. Texto del capítulo dos.")
    chunks = chunk_pages([Page(1, texto)], size_tokens=450)
    assert len(chunks) == 2  # un fragmento por capítulo, no uno por artículo
    assert "dos inscripciones" in chunks[0].text and "Capítulo I De" in chunks[0].text
    assert chunks[0].section == "Capítulo I De la inscripción – Artículo 26"
    assert chunks[1].text.startswith("Capítulo II")


def test_chunker_no_toma_lineas_de_texto_como_encabezado():
    from chatbot.ingestion.chunker import classify_heading
    assert classify_heading("15 días hábiles para inscribirse en el periodo") is None
    assert classify_heading("título profesional, en los términos del reglamento") is None
    assert classify_heading("Artículo 31 . Los alumnos que hayan seleccionado") == ("Artículo 31", False)
    assert classify_heading("Título I Disposiciones generales")[1] is True
    assert classify_heading("### Periodo 3") == ("Periodo 3", True)


def test_reranker_rescata_un_candidato_que_la_fusion_dejo_fuera_de_k():
    chunks = [Chunk(str(i), f"texto {i}", "d") for i in range(6)]
    vec = [ScoredChunk(c, 0.9 - i / 10, 0.9 - i / 10) for i, c in enumerate(chunks)]
    vistos = []

    def reranker(query, cands):  # prefiere el último candidato de la fusión
        vistos.append([c.chunk.id for c in cands])
        return list(reversed(cands))

    sin = HybridRetriever(_Emb(), _Vec(vec), _Lex([]), k=2).retrieve("q")
    con = HybridRetriever(_Emb(), _Vec(vec), _Lex([]), k=2, reranker=reranker,
                          rerank_candidates=4).retrieve("q")
    assert [o.chunk.id for o in sin] == ["0", "1"]
    assert vistos == [["0", "1", "2", "3"]]           # reordena 4 candidatos, no solo k
    assert [o.chunk.id for o in con] == ["3", "2"]    # y luego corta a k
    assert con[0].relevance == vec[3].relevance       # conserva la similitud vectorial


def test_cross_encoder_ordena_por_puntaje_y_conserva_relevancia():
    from chatbot.adapters.reranker.cross_encoder_reranker import CrossEncoderReranker

    class _Modelo:
        def predict(self, pares, **kw):
            return [1.0 if "inscripciones" in texto else 0.1 for _, texto in pares]

    rr = CrossEncoderReranker.__new__(CrossEncoderReranker)  # sin cargar el modelo real
    rr._model = _Modelo()
    a = ScoredChunk(Chunk("a", "baja temporal hasta cinco veces", "d"), 0.03, 0.64)
    b = ScoredChunk(Chunk("b", "derecho a dos inscripciones", "d"), 0.01, 0.59)
    out = rr("¿cuántas veces puedo cursar?", [a, b])
    assert [o.chunk.id for o in out] == ["b", "a"] and out[0].relevance == 0.59
    assert rr("q", []) == []
