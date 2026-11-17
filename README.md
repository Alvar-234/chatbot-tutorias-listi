# Chatbot de tutorías LISTI (RAG)

Asistente informativo de apoyo a las tutorías académicas de la Licenciatura en Ingeniería en
Sistemas y Tecnologías de la Información (LISTI) de la Universidad Veracruzana.

Responde preguntas sobre plan de estudios, reglamentos y trámites **usando únicamente documentos
oficiales**, cita la fuente de cada respuesta, se abstiene cuando no tiene evidencia suficiente y
deriva al tutor los temas sensibles. No sustituye al tutor ni realiza trámites.

> **Estado:** prototipo funcional en pruebas con un corpus parcial. Aún no está listo para
> estudiantes; ver [Pendientes](#pendientes).

---

## Qué hace

- **Responde con fuentes.** Cada afirmación lleva una cita `[n]` a un fragmento real de un
  documento (título, sección, página y vigencia).
- **Se abstiene** si la evidencia recuperada no es suficiente, en lugar de inventar.
- **Verifica su propia respuesta:** descarta respuestas con citas inexistentes, cifras o claves
  que no aparecen en las fuentes, respuestas vacías o cortadas.
- **Deriva al tutor** temas de riesgo, sensibles, solicitudes de trámites o cuando no puede
  ayudar tras varios intentos. Los mensajes de riesgo se responden sin pasar por el LLM.
- **Protege datos personales:** correo, CURP, matrícula y teléfono se redactan antes de llegar al
  LLM o a los registros. Las trazas no guardan el texto de preguntas ni respuestas.

## Cómo funciona

Cada consulta recorre este flujo (`src/chatbot/domain/pipeline.py`):

```
pregunta
  │
  1. Sanitizar ──── recorta y redacta datos personales          privacy/redactor.py
  2. Enrutar ────── derivación › fuera de alcance ›             domain/router.py
  │                 estructurada › documental
  3. Evidencia
  │    ├─ estructurada: consulta exacta a SQLite                adapters/structured/
  │    └─ documental:   BM25 + vectorial ─ fusión RRF ─         retrieval/hybrid.py
  │                     reranker (cross-encoder) ─ top k        adapters/reranker/
  4. Suficiencia ── relevancia ≥ umbral y entidades presentes   domain/sufficiency.py
  │                 (si no: abstención)
  5. Generar ────── LLM con cadena de contingencia              adapters/llm/
  │                 (Groq → modelo local)
  6. Verificar ──── citas reales, cifras y claves respaldadas   domain/verifier.py
  │                 (si falla: abstención)
  7. Presentar ──── respuesta + fuentes; abstenciones
                    repetidas derivan al tutor
```

Dos rutas de evidencia:

- **Documental (RAG):** los documentos se dividen en fragmentos, se indexan de dos formas
  (léxica con BM25 y semántica con embeddings) y, al preguntar, se fusionan ambos resultados, se
  reordenan con un reranker y los 5 mejores se envían al LLM como fuentes.
- **Estructurada:** datos exactos del plan de estudios y la seriación en SQLite. Si el LLM falla
  o su respuesta no pasa la verificación, se entrega el dato de la base tal cual.
  *Hoy solo contiene datos ficticios de ejemplo (`DEMO-…`)*; ver [Pendientes](#pendientes).

La ingesta de documentos es un proceso aparte, que se ejecuta una vez y cada vez que cambia el
corpus (`src/chatbot/ingestion/build_index.py`):

```
manifest.yaml → cargar (PDF/MD/TXT) → limpiar encabezados y pies repetidos
             → fragmentar por estructura (títulos, capítulos, artículos, encabezados Markdown)
             → embeddings → Chroma  +  chunks.jsonl (para BM25)
```

## Tecnologías

| Capa | Tecnología | Notas |
|---|---|---|
| Lenguaje | Python ≥ 3.11 | probado con 3.11.9 |
| API | FastAPI + Uvicorn | `POST /chat`, `GET /health` |
| Interfaz | HTML + JavaScript sin framework | `web/index.html`, servida por la misma API |
| LLM principal | Groq, modelo `openai/gpt-oss-20b` | API compatible con OpenAI; nivel gratuito |
| LLM de contingencia | Ollama, modelo `llama3.2` | local; opcional |
| Embeddings | `BAAI/bge-m3` con sentence-transformers | local, multilingüe |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) | local; cuantizado a 8 bits en CPU |
| Almacén vectorial | ChromaDB (persistente en disco) | distancia coseno |
| Búsqueda léxica | BM25 en Python puro | sin dependencias |
| Datos estructurados | SQLite | plan de estudios y seriación |
| Lectura de PDF | pypdf | solo PDF con texto; los escaneados requieren OCR previo |
| Cliente HTTP | httpx | llamadas a los LLM |
| Pruebas | pytest | 56 pruebas, sin red ni modelos |

No se usa LangChain ni otro framework de orquestación: la lógica es código propio.

## Estructura del proyecto

```
chatbot-tutorias-listi/
├── config/
│   ├── settings.yaml            # modelos, umbrales, rutas: se cambia sin tocar código
│   └── sensitive_topics.yaml    # frases que disparan derivación y contactos de ayuda
├── Documentos/                  # corpus fuente: .md extraídos a mano (versionados) y PDFs (no versionados)
├── data/
│   ├── raw/manifest.yaml        # qué documentos se indexan y sus metadatos
│   ├── processed/               # chunks.jsonl e index_info.json (generados)
│   ├── chroma/                  # índice vectorial (generado)
│   └── structured/
│       ├── sample/*.csv         # asignaturas y seriación FICTICIAS de ejemplo
│       └── plan_estudios.db     # SQLite (generado)
├── src/chatbot/
│   ├── domain/                  # lógica propia; no importa librerías externas
│   │   ├── pipeline.py          #   orquestador del flujo de consulta
│   │   ├── router.py            #   decide la ruta de cada pregunta
│   │   ├── handoff.py           #   reglas de derivación al tutor
│   │   ├── sufficiency.py       #   ¿la evidencia alcanza para responder?
│   │   ├── verifier.py          #   validación de la respuesta generada
│   │   ├── prompts.py           #   prompts y mensajes fijos
│   │   ├── models.py, text.py   #   tipos y utilidades
│   │   └── ports.py             #   contratos que implementan los adaptadores
│   ├── adapters/                # implementaciones intercambiables de los puertos
│   │   ├── llm/                 #   cliente OpenAI-compatible y cadena de proveedores
│   │   ├── embeddings/          #   sentence-transformers
│   │   ├── reranker/            #   cross-encoder
│   │   ├── vectorstore/         #   Chroma
│   │   ├── lexical/             #   BM25
│   │   └── structured/          #   SQLite
│   ├── retrieval/hybrid.py      # fusión RRF + reranker
│   ├── ingestion/               # carga, limpieza, fragmentación y construcción del índice
│   ├── privacy/                 # redacción de datos personales y límite de solicitudes
│   ├── observability/           # trazas JSONL por consulta (sin texto del usuario)
│   ├── api/                     # FastAPI: rutas, esquemas y ensamblado del pipeline
│   ├── config.py                # carga de settings.yaml y .env
│   └── tls.py                   # certificados del sistema (ver Problemas frecuentes)
├── web/index.html               # interfaz de chat
├── scripts/init_db.py           # crea la base SQLite desde CSV
├── eval/                        # métricas y ejecución de experimentos
├── tests/                       # pruebas unitarias
├── logs/                        # traces.jsonl (generado)
├── docs/adr/                    # decisiones de arquitectura (pendientes de redactar)
└── Requerimientos/              # documento de alcance y comparación de componentes
```

La arquitectura es de **puertos y adaptadores**: `domain/` define qué necesita (`ports.py`) y
`adapters/` lo implementa. Cambiar de proveedor de LLM, modelo de embeddings o almacén vectorial
es editar `settings.yaml` o escribir un adaptador nuevo, sin tocar la lógica.

---

## Puesta en marcha

### Qué se necesita descargar

| Qué | Tamaño aprox. | Cómo se obtiene |
|---|---|---|
| Python 3.11 o superior | — | [python.org](https://www.python.org/downloads/) |
| Dependencias de Python (incluye PyTorch) | 1–2 GB | `pip install` (paso 1) |
| Modelo de embeddings `BAAI/bge-m3` | 4.3 GB | automático al construir el índice (paso 5) |
| Reranker `BAAI/bge-reranker-v2-m3` | 2.2 GB | automático al arrancar la API (paso 6) |
| Clave de API de Groq (gratuita) | — | [console.groq.com](https://console.groq.com) |
| *Opcional:* Ollama + `llama3.2` | 2 GB | [ollama.com](https://ollama.com), luego `ollama pull llama3.2` |

Total en disco: unos 9 GB (11 GB con Ollama). Los modelos se guardan en
`~/.cache/huggingface` y solo se descargan la primera vez. No se necesita GPU, pero sin ella las
respuestas tardan unos 20 s (ver [Rendimiento](#rendimiento-y-límites-conocidos)).

### Pasos

Los comandos se ejecutan desde la carpeta del proyecto.

**1. Crear el entorno e instalar dependencias**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev,rag]"
```

**2. Comprobar la instalación**

```bash
pytest                            # deben pasar 56 pruebas; no usan red ni modelos
```

**3. Configurar la clave de Groq**

```bash
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

Edita `.env` y escribe tu clave en `GROQ_API_KEY=`. Este archivo no se sube al repositorio.

**4. Crear la base de datos estructurada**

```bash
python scripts/init_db.py
```

Es obligatorio aunque los datos sean de ejemplo: sin la base, la API no arranca el asistente.

**5. Preparar el corpus y construir el índice**

Los PDF institucionales **no están en el repositorio**: hay que conseguirlos y colocarlos en
`Documentos/` con los nombres que indica `data/raw/manifest.yaml`. Ese archivo lista qué
documentos se indexan, una entrada por documento, con la ruta relativa a `data/raw/`:

```yaml
documents:
  - file: ../../Documentos/05_E_de-los-Alumnos_19-SEP-25_1123h_Version-Compendio-2025.pdf
    doc_id: estatuto_alumnos            # identificador corto y único
    title: "Estatuto de los Alumnos 2008 (compendio 2025)"
    version: "compendio 2025"           # opcional
    valid_from: "2025-09-19"            # opcional; solo si la fecha está confirmada
    vigente: true                       # false = se excluye de las respuestas
    url: "https://..."                  # opcional; enlace que se muestra en la cita
```

Después:

```bash
python -m chatbot.ingestion.build_index
```

La primera vez descarga el modelo de embeddings. Con el corpus actual genera unos 276 fragmentos
y tarda de 3 a 5 minutos en CPU. **Hay que repetir este paso cada vez que se agregue o modifique
un documento**, o se cambie `chunking` o el modelo de embeddings en `settings.yaml`.

**6. Arrancar el chatbot**

```bash
uvicorn chatbot.api.main:app --reload
```

Abre <http://localhost:8000>. El primer arranque descarga el reranker; los siguientes tardan
unos 35 s en cargar los modelos. Si algo falta (índice, base, configuración), la API arranca
igual pero `/chat` responde 503; `http://localhost:8000/health` indica si el asistente está listo.

**7. (Opcional) Modelo local de contingencia**

Con Ollama instalado y en ejecución: `ollama pull llama3.2`. Se usa solo cuando Groq falla.

### Agregar o corregir documentos

1. Coloca el archivo en `Documentos/` (PDF con texto, `.md` o `.txt`).
2. Añade su entrada en `data/raw/manifest.yaml`.
3. Ejecuta `python -m chatbot.ingestion.build_index` y reinicia la API.

Recomendaciones aprendidas en las pruebas:

- **Los datos agregados deben estar escritos.** El asistente solo ve 5 fragmentos por pregunta y
  no sabe sumar sobre todo un documento. Si se espera una pregunta como "¿cuántas materias hay
  en total?", el total debe aparecer como texto (así se hizo en `mapa_curricular_LISTI.md`).
- **En Markdown, usa encabezados (`##`, `###`)** para separar temas: cada encabezado inicia un
  fragmento nuevo y se muestra como sección en la cita.
- **Redacta con las palabras que usaría un estudiante** ("materias", "inscripción"), además de
  los términos oficiales.
- **Los PDF escaneados se omiten** con un aviso `[OMITIDO]`; necesitan OCR antes de indexarse.

---

## Configuración

Todo se ajusta en `config/settings.yaml`. Lo más relevante:

| Clave | Para qué sirve |
|---|---|
| `llm.chain` | Proveedores en orden de uso. `extra` envía parámetros propios del proveedor. |
| `llm.max_tokens` | Presupuesto de la respuesta. En modelos de razonamiento incluye el "pensamiento". |
| `llm.retry_429_max_wait_s` | Espera máxima ante límite de cuota antes de pasar al siguiente proveedor. |
| `embeddings.model` | Modelo de embeddings. Cambiarlo obliga a reconstruir el índice. |
| `chunking.size_tokens` | Tamaño de fragmento. Cambiarlo obliga a reconstruir el índice. |
| `retrieval.k` | Fragmentos que se envían al LLM. |
| `retrieval.reranker.model` | Modelo de reordenamiento. `""` lo desactiva. |
| `sufficiency.relevance_threshold` | Similitud mínima para intentar responder (provisional, sin calibrar). |
| `privacy.rate_limit` | Solicitudes permitidas por ventana de tiempo. |

`config/sensitive_topics.yaml` contiene las frases que disparan la derivación y los contactos que
se muestran al estudiante.

## Rendimiento y límites conocidos

Medido con el corpus actual en un equipo sin GPU (6 hilos):

- **Latencia: unos 20 s por respuesta**, casi toda del reranker. Sin reranker
  (`retrieval.reranker.model: ""`) baja a 2–3 s, pero empeoran las preguntas donde el estudiante
  y el reglamento usan palabras distintas: "¿cuántas veces puedo cursar una experiencia
  educativa?" debe encontrar el artículo que habla de "dos inscripciones". Con GPU el costo del
  reranker sería despreciable.
  Alternativas medidas y descartadas: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (4–7 s, no
  encuentra ese artículo) y `BAAI/bge-reranker-base` (8–10 s, ordena mal en español).
- **Cuota de Groq (nivel gratuito): 8 000 tokens por minuto**, unas 4 consultas por minuto en
  total, no por usuario. Al agotarse, el cliente espera hasta 10 s y después pasa al modelo local.
- **El modelo local casi no sirve de respaldo en CPU:** `llama3.2` supera el tiempo límite de
  60 s en respuestas largas y con frecuencia no cita, por lo que su respuesta se descarta.
- **La verificación no detecta errores de interpretación.** Comprueba citas, cifras y claves,
  pero no si el LLM entendió mal una fuente real.
- **Sin memoria de tema entre preguntas:** una pregunta de seguimiento ("¿y cuántos créditos
  tiene?") no hereda la asignatura de la anterior.

## Problemas frecuentes

**`CERTIFICATE_VERIFY_FAILED` al descargar modelos, llamar a Groq o usar `pip`.**
Ocurre cuando un antivirus inspecciona el tráfico HTTPS (por ejemplo, Avast). Windows confía en
el certificado del antivirus, pero Python no. Solución: instalar `truststore`; el proyecto lo
detecta y usa los certificados del sistema (`src/chatbot/tls.py`), sin desactivar la verificación.

```bash
pip install truststore
```

Si `pip` falla por el mismo motivo, descarga el archivo `.whl` de
<https://pypi.org/project/truststore/#files> con el navegador e instálalo con
`pip install ruta\al\archivo.whl`. A partir de ahí, `pip install --use-feature=truststore <paquete>`
funciona con normalidad.

**`/chat` responde 503.** El asistente no pudo construirse. Revisa el mensaje
`No se pudo construir el pipeline` en la consola de Uvicorn; lo habitual es que falte el índice
(paso 5), la base SQLite (paso 4) o un modelo en `settings.yaml`.

**"El servicio no está disponible en este momento".** Fallaron todos los proveedores de LLM;
normalmente es la cuota de Groq. Espera un minuto.

**El asistente se abstiene demasiado.** Consulta `logs/traces.jsonl`: cada línea registra la
decisión, el motivo (`sufficiency_reason`, `verification_problems`), los fragmentos recuperados y
los tiempos por etapa.

---

## Pruebas y evaluación

```bash
pytest                                                # pruebas unitarias
python eval/run_experiment.py eval/dataset/dev.jsonl  # evaluación con el pipeline real
```

Las pruebas unitarias cubren el dominio, el redactor, BM25, la fusión, el reranker (con un modelo
simulado), el fragmentador, el cliente de LLM (sin red), el repositorio SQLite y la API.
El pipeline real (Groq, embeddings, Chroma, reranker, ingesta de PDF) se ha probado a mano con
9 preguntas sobre el corpus actual. `eval/run_experiment.py` aún no se ha ejecutado: falta
construir el conjunto de evaluación (`eval/dataset/dev.example.jsonl` muestra el formato).

## Pendientes

- [ ] **Corpus:** faltan documentos; el Reglamento de Movilidad 2015 está escaneado y requiere OCR.
- [ ] **Autorización** de cada documento y fechas de vigencia (`valid_from`) en el manifiesto.
- [ ] **Plan de estudios y seriación reales** en CSV con claves únicas; hoy la ruta estructurada
      solo tiene datos ficticios y el mapa curricular se consulta como documento.
- [ ] **Contactos de ayuda** en `config/sensitive_topics.yaml` (siguen como `REEMPLAZAR`) y lista
      de temas sensibles validada con tutores. Algunas frases derivan de más (p. ej. "tramita"
      deriva preguntas informativas sobre trámites).
- [ ] **Latencia:** decidir entre GPU, servicio de reranking externo o prescindir del reranker.
- [ ] **Umbral de suficiencia:** calibrarlo con un conjunto `dev`; reservar `holdout` para el final.
- [ ] **Conjunto de evaluación** y primera ejecución de `eval/run_experiment.py`.
- [ ] **Límite de solicitudes:** se puede evadir cambiando el `session_id`; debe basarse en la IP.
- [ ] **Seriación:** el verificador no detecta que el LLM omita o niegue un requisito.
- [ ] **Formato real de matrícula** en `privacy/redactor.py` (se asumió `zS` + 8 dígitos).
- [ ] **Aviso de privacidad** definitivo en `web/index.html` y política de retención de `logs/`.
- [ ] **Versiones exactas** de las dependencias `rag` y de `truststore` en `pyproject.toml`.
- [ ] **Dockerfile:** no incluye `data/` ni `Documentos/`; requiere volúmenes para funcionar.
- [ ] **Decisiones de arquitectura** ADR-001 a ADR-007 en `docs/adr/`.
