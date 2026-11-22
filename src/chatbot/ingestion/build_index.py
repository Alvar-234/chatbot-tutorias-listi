"""Construye el índice (offline): manifiesto -> carga -> limpieza -> chunks -> embeddings -> Chroma.

Uso:  python -m chatbot.ingestion.build_index
Escribe data/processed/chunks.jsonl (para BM25) e index_info.json (versiones, RNF8/ADR-007).
Requiere el extra "rag". Cambiar de modelo de embeddings obliga a reconstruir el índice.
"""
from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from pathlib import Path

from ..config import load_settings
from ..domain.models import Chunk
from ..tls import use_system_certificates
from .chunker import chunk_pages
from .cleaner import clean_pages
from .loaders import NeedsOCR, load_document
from .metadata import load_manifest


def resolve_path(path: Path) -> Path:
    """La ruta del manifiesto, o el archivo de esa carpeta con el mismo nombre en Unicode NFC.

    Los PDF de documentos_ee_pdf/ traen los acentos descompuestos (NFD) en el nombre: el
    manifiesto puede escribirlos normalmente y aun así encontrarlos.
    """
    if path.exists() or not path.parent.is_dir():
        return path
    want = unicodedata.normalize("NFC", path.name)
    for p in path.parent.iterdir():
        if unicodedata.normalize("NFC", p.name) == want:
            return p
    return path


def build_chunks(settings) -> list[Chunk]:
    raw_dir = settings.path("manifest").parent
    cfg = settings["chunking"]
    chunks: list[Chunk] = []
    for doc in load_manifest(settings.path("manifest")):
        try:
            pages = clean_pages(load_document(resolve_path(raw_dir / doc["file"])))
        except NeedsOCR as exc:
            print(f"[OMITIDO] {exc}")
            continue
        # `contexto`: se antepone a cada fragmento. En los programas de EE, un fragmento de
        # "Saberes" o "Evaluación" no nombra la materia; sin esto no se recupera al preguntar
        # "¿qué temas ve Bases de Datos Relacionales?".
        prefix = f'{doc["contexto"].strip()} ' if doc.get("contexto") else ""
        for i, rc in enumerate(chunk_pages(pages, cfg["size_tokens"], cfg["overlap_pct"])):
            chunks.append(Chunk(
                id=f'{doc["doc_id"]}-{i:04d}', text=prefix + rc.text, doc_id=doc["doc_id"],
                doc_title=doc["title"], version=str(doc.get("version", "")),
                valid_from=str(doc.get("valid_from", "")), section=rc.section,
                page=rc.page, url=doc.get("url", ""), vigente=bool(doc.get("vigente", True))))
    return chunks


def main() -> None:
    from ..adapters.embeddings.factory import build_embedder, collection_name
    from ..adapters.vectorstore.chroma_store import ChromaVectorStore

    use_system_certificates()
    settings = load_settings()
    chunks = build_chunks(settings)
    if not chunks:
        raise SystemExit("No se generaron fragmentos: revisa data/raw/ y el manifiesto.")

    emb_cfg = settings["embeddings"]
    embedder = build_embedder(emb_cfg)
    # Primero los embeddings (lo que puede fallar: red, cuota, filtros de la API) y solo después
    # se tocan el índice y chunks.jsonl: si algo falla, el índice anterior queda intacto.
    vectors = embedder.embed_documents([c.text for c in chunks])

    out = settings.path("chunks")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "text": c.text, "meta": c.to_metadata()},
                               ensure_ascii=False) + "\n")
    store = ChromaVectorStore(str(settings.path("chroma_dir")), collection_name(emb_cfg))
    store.reset()
    store.add(chunks, vectors)
    for aviso in getattr(embedder, "warnings", []):
        print(f"[AVISO] {aviso}")

    corpus_hash = hashlib.sha256("".join(c.text for c in chunks).encode()).hexdigest()[:12]
    settings.path("index_info").write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_chunks": len(chunks),
        "embedding_model": embedder.model_id, "chunking": settings["chunking"],
        "corpus_hash": corpus_hash}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Índice listo: {len(chunks)} fragmentos (corpus {corpus_hash}).")


if __name__ == "__main__":
    main()
