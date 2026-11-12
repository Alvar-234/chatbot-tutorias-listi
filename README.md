# Chatbot de tutorías LISTI (RAG)

Esqueleto del proyecto. La lógica propia (enrutamiento, suficiencia de evidencia, verificación,
derivación) vive en `src/chatbot/domain/` y **no importa** LangChain, Groq ni Chroma: todo lo
externo entra por los puertos de `domain/ports.py` (RNF6).

## Arranque rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # núcleo + pruebas
pytest                             # 38 pruebas, sin red ni modelos
python scripts/init_db.py          # SQLite con datos FICTICIOS de ejemplo
```

Para la parte RAG real:

```bash
pip install -e ".[rag]"            # sentence-transformers, chromadb, pypdf (fija versiones: pip freeze)
cp .env.example .env               # agrega GROQ_API_KEY
# 1) edita config/settings.yaml: modelos (marcados REEMPLAZAR) y umbrales
# 2) coloca PDFs autorizados en data/raw/ y describe cada uno en data/raw/manifest.yaml
python -m chatbot.ingestion.build_index
uvicorn chatbot.api.main:app --reload   # abre http://localhost:8000
```

Si falta el índice o la configuración, la API arranca igual y `/chat` responde 503 (`/health` lo indica).

## Flujo de una consulta (`domain/pipeline.py`)

1. **Sanitizar**: recorte y redacción de PII (`privacy/redactor.py`).
2. **Enrutar** (`router.py`): derivación › fuera de alcance › estructurada › documental.
3. **Evidencia**: SQLite (estructurada) o híbrido BM25 + vectorial con RRF (documental).
4. **Suficiencia** (`sufficiency.py`): relevancia ≥ umbral y entidades presentes; si no, abstención.
5. **Generar** con la cadena de contingencia de proveedores (`adapters/llm/`).
6. **Verificar** (`verifier.py`): citas reales y cifras/claves respaldadas por las fuentes.
7. **Presentar** con fuentes y vigencia; abstenciones repetidas derivan al tutor.

La seriación **nunca depende del LLM**: si falla la generación o la verificación, se responde con
el dato exacto de la base (`deterministic_structured_answer`).

## Qué está probado y qué no

| Componente | Estado |
|---|---|
| dominio, redactor, BM25, RRF, chunker, repo SQLite, API (con fakes) | cubierto por `pytest` |
| `OpenAICompatProvider` (Groq / HF / local) | escrito, **no probado contra la red** |
| `SentenceTransformerEmbedder`, `ChromaVectorStore`, `loaders` PDF, `build_index` | escritos, **no ejecutados** (requieren extra `rag` y corpus real) |
| `eval/run_experiment.py` | escrito, no ejecutado de punta a punta |

## Pendientes que requieren decisión humana

- [ ] Modelos reales en `settings.yaml` (catálogo y límites vigentes de Groq; modelo local de contingencia).
- [ ] Plan de estudios y seriación reales en CSV (los de `data/structured/sample` son ficticios).
- [ ] Lista de temas sensibles y contactos de ayuda en `config/sensitive_topics.yaml`, validada con tutores.
- [ ] Formato real de matrícula en `privacy/redactor.py` (se asumió `zS` + 8 dígitos).
- [ ] Autorización y manifiesto de los documentos del corpus (`data/raw/manifest.yaml`).
- [ ] Umbral de suficiencia: calibrar con el subconjunto `dev`; `holdout` solo al final.
- [ ] Aviso de privacidad definitivo en `web/index.html` y política de retención de `logs/`.
- [ ] Fijar versiones exactas de las dependencias `rag` (riesgo R7) y registrar ADR-001…007.
