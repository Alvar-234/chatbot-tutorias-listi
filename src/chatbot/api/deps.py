"""Ensamblado del pipeline real a partir de config/settings.yaml (composition root)."""
from __future__ import annotations

import json

import yaml

from ..adapters.llm.factory import build_llm_chain
from ..adapters.structured.sqlite_repo import SqliteStructuredRepo
from ..config import Settings
from ..domain.handoff import HandoffRules
from ..domain.pipeline import ChatPipeline, PipelineSettings
from ..domain.router import Router
from ..domain.sufficiency import SufficiencyChecker
from ..domain.verifier import AnswerVerifier
from ..observability.tracing import JsonlTracer
from ..privacy.redactor import Redactor


def build_pipeline(settings: Settings) -> ChatPipeline:
    # Importes pesados aquí para que el resto de la app (y las pruebas) no los necesiten.
    from ..adapters.embeddings.factory import build_embedder, collection_name
    from ..adapters.lexical.bm25_index import BM25Index
    from ..adapters.vectorstore.chroma_store import ChromaVectorStore
    from ..retrieval.hybrid import HybridRetriever

    topics = yaml.safe_load(settings.path("sensitive_topics").read_text(encoding="utf-8"))
    handoff = HandoffRules(topics, topics.get("contact"))
    repo = SqliteStructuredRepo(settings.path("structured_db"))
    router = Router(handoff, repo, settings["routing"].get("out_of_scope_keywords", []))

    emb = settings["embeddings"]
    embedder = build_embedder(emb)
    store = ChromaVectorStore(str(settings.path("chroma_dir")), collection_name(emb))
    if store.count() == 0:
        raise RuntimeError(
            f"No hay índice para los embeddings '{emb.get('provider', 'local')}' "
            f"({embedder.model_id}). Ejecuta: python -m chatbot.ingestion.build_index")
    retr = settings["retrieval"]
    rr = retr.get("reranker") or {}
    reranker = None
    provider = rr.get("provider", "local")
    if provider == "voyage" and rr.get("voyage_model"):
        from ..adapters.reranker.voyage_reranker import VoyageReranker
        reranker = VoyageReranker(rr["voyage_model"], timeout_s=rr.get("timeout_s", 15))
    elif provider == "nvidia" and rr.get("nvidia_model"):
        from ..adapters.reranker.nvidia_reranker import NvidiaReranker
        reranker = NvidiaReranker(rr["nvidia_model"], timeout_s=rr.get("timeout_s", 15))
    elif provider == "local" and rr.get("model"):
        from ..adapters.reranker.cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(rr["model"], rr.get("max_length", 512),
                                        quantize_int8=rr.get("quantize_int8", False))
    elif provider not in ("local", "voyage", "nvidia"):
        raise ValueError(f"Reranker desconocido: {provider} (usa local | voyage | nvidia)")
    retriever = HybridRetriever(
        embedder, store,
        BM25Index.from_jsonl(settings.path("chunks")),
        k=retr["k"], k_candidates=retr["k_candidates"], rrf_k=retr["rrf_k"],
        reranker=reranker, rerank_candidates=rr.get("candidates", retr["k_candidates"]))

    llm_cfg = settings["llm"]
    pl = settings["pipeline"]
    info_path = settings.path("index_info")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    return ChatPipeline(
        router=router, retriever=retriever, repo=repo, llms=build_llm_chain(llm_cfg),
        redactor=Redactor(),
        sufficiency=SufficiencyChecker(
            settings["sufficiency"]["relevance_threshold"],
            settings["sufficiency"].get("ignore_entities", []),
            # umbral del reranker en uso; si no está calibrado, se usa el vectorial
            (settings["sufficiency"].get("rerank_threshold") or {}).get(
                getattr(reranker, "model_id", ""))),
        verifier=AnswerVerifier(), handoff=handoff,
        tracer=JsonlTracer(settings.path("traces")),
        settings=PipelineSettings(
            max_query_chars=pl["max_query_chars"], history_turns=pl["history_turns"],
            max_insufficient_before_handoff=pl["max_insufficient_before_handoff"],
            temperature=llm_cfg["temperature"], max_tokens=llm_cfg["max_tokens"]),
        versions={"embedding_model": embedder.model_id,
                  "reranker_model": getattr(reranker, "model_id", ""),
                  "corpus_hash": info.get("corpus_hash", ""),
                  "chunking": info.get("chunking", {})})
