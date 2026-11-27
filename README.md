# Chatbot de tutorías LISTI (RAG)

Asistente informativo de apoyo a las tutorías académicas de la Licenciatura en Ingeniería en
Sistemas y Tecnologías de la Información (LISTI) de la Universidad Veracruzana.

Responde preguntas sobre plan de estudios, seriación, contenido de las experiencias educativas
(EE), reglamentos y trámites **usando únicamente documentos oficiales**, cita la fuente de cada
respuesta, se abstiene cuando no tiene evidencia suficiente y deriva al tutor los temas
sensibles. No sustituye al tutor ni realiza trámites.

> **Estado:** prototipo funcional con el plan de estudios 2023 completo (50 EE validadas) y un
> corpus de 55 documentos. Acierta 54/54 en el conjunto de evaluación de desarrollo, pero aún no
> está listo para estudiantes; ver [Pendientes](#pendientes).

---

## Qué hace

- **Responde con fuentes.** Cada afirmación lleva una cita `[n]` a un fragmento real de un
  documento (título, sección, página y vigencia).
- **Conoce el plan de estudios exacto:** créditos, horas, periodo sugerido, área, carácter
  (obligatoria u optativa), prerrequisitos y recomendaciones de las 50 EE, desde una base SQLite
  validada con la coordinación.
- **Responde sobre el contenido de cada EE:** temas, unidad de competencia, evaluación,
  bibliografía y perfil docente, a partir de los 44 programas oficiales.
- **Se abstiene** si la evidencia recuperada no es suficiente, en lugar de inventar.
- **Verifica su propia respuesta:** descarta citas inexistentes, cifras o claves que no están en
  las fuentes, respuestas vacías o cortadas, y afirmaciones de que algo "no existe" basadas solo
  en que las fuentes no lo mencionan. Si encuentra un dato sin respaldo, le pide al LLM una
  segunda versión antes de abstenerse.
- **Nunca omite la seriación:** si el LLM omite o niega un requisito, se entrega el dato exacto
  de la base; las recomendaciones ("se recomienda cursar antes Networking") se presentan aparte
  y como no obligatorias.
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
  │                 estructurada (materia + "requisito", "créditos", "semestre"...) › documental
  │                 (seguimientos como "¿y general?" se buscan y enrutan con la    domain/followup.py
  │                  consulta anterior antepuesta; el LLM recibe la pregunta tal cual)
  3. Evidencia
  │    ├─ estructurada: consulta exacta a SQLite                adapters/structured/
  │    └─ documental:   BM25 + vectorial ─ fusión RRF ─         retrieval/hybrid.py
  │                     reranker ─ top 5                        adapters/reranker/
  │                     (si la pregunta nombra una materia, se descartan los
  │                      programas de OTRAS materias)
  4. Suficiencia ── puntaje del reranker ≥ umbral (o similitud  domain/sufficiency.py
  │                 vectorial si no hay reranker) y entidades presentes
  │                 (si no: abstención)
  5. Generar ────── LLM con cadena de contingencia              adapters/llm/
  │                 (NVIDIA → Groq → modelo local)
  6. Verificar ──── citas reales, cifras y claves respaldadas   domain/verifier.py
  │                 (si falla: un reintento señalando el dato; si vuelve a fallar: abstención)
  │                 estructurada: requisitos, recomendaciones y notas completos
  7. Presentar ──── respuesta + fuentes; abstenciones
                    repetidas derivan al tutor
```

Dos rutas de evidencia:

- **Documental (RAG):** los documentos se dividen en fragmentos, se indexan de dos formas
  (léxica con BM25 y semántica con embeddings) y, al preguntar, se fusionan ambos resultados, se
  reordenan con un reranker y los 5 mejores se envían al LLM como fuentes.
- **Estructurada:** datos exactos del plan de estudios y la seriación en SQLite. Si el LLM falla
  o su respuesta no pasa la verificación, se entrega el dato de la base tal cual. Si el LLM omite
  un requisito obligatorio después de hablar de requisitos, también; si solo omitió un dato que
  no le preguntaron, una recomendación o una nota de la materia, se añade al final. Ver
  [Datos del plan de estudios](#datos-del-plan-de-estudios).

La ingesta de documentos es un proceso aparte, que se ejecuta una vez y cada vez que cambia el
corpus (`src/chatbot/ingestion/build_index.py`):

```
manifest.yaml → cargar (PDF/MD/TXT) → limpiar encabezados y pies repetidos
             → fragmentar por estructura (títulos, capítulos, artículos, encabezados Markdown)
             → anteponer `contexto` (p. ej. "Programa de la EE Networking (LISTI 2023).")
             → embeddings → Chroma  +  chunks.jsonl (para BM25)
```

## Tecnologías

| Capa | Tecnología | Notas |
|---|---|---|
| Lenguaje | Python ≥ 3.11 | probado con 3.11.9 |
| API | FastAPI + Uvicorn | `POST /chat`, `GET /health` |
| Interfaz | HTML + JavaScript sin framework | `web/index.html`, servida por la misma API |
| LLM principal | NVIDIA `nvidia/nemotron-3.5-lightning-30b-a3b` (sin razonamiento) | API compatible con OpenAI (build.nvidia.com) |
| LLM de contingencia | Groq `openai/gpt-oss-20b`, después Ollama `llama3.2` | API gratuita; local opcional |
| Embeddings | NVIDIA `nvidia/nemotron-3.5-embed-1b` (por defecto) o `BAAI/bge-m3` local | ambos multilingües |
| Reranker | NVIDIA `nvidia/nemotron-3.5-rerank-8b` (por defecto), Voyage AI `rerank-2.5` o `BAAI/bge-reranker-v2-m3` local | el local, cuantizado a 8 bits en CPU |
| Almacén vectorial | ChromaDB (persistente en disco) | distancia coseno; una colección por proveedor de embeddings |
| Búsqueda léxica | BM25 en Python puro | sin dependencias |
| Datos estructurados | SQLite | plan de estudios, seriación y recomendaciones |
| Lectura de PDF | pypdf | solo PDF con texto; los escaneados requieren OCR previo |
| Cliente HTTP | httpx | LLM, embeddings y rerankers por API |
| Pruebas | pytest | 104 pruebas, sin red ni modelos |

No se usa LangChain ni otro framework de orquestación: la lógica es código propio.

## Estructura del proyecto

```
chatbot-tutorias-listi/
├── config/
│   ├── settings.yaml            # modelos, umbrales, rutas: se cambia sin tocar código
│   └── sensitive_topics.yaml    # frases que disparan derivación y contactos de ayuda
├── Documentos/                  # corpus: .md extraídos a mano (versionados) y PDF (no versionados)
├── documentos_ee_pdf/           # los 44 programas de EE en PDF (no versionados)
├── data/
│   ├── raw/manifest.yaml        # qué documentos se indexan y sus metadatos
│   ├── processed/               # chunks.jsonl e index_info.json (generados)
│   ├── chroma/                  # índice vectorial (generado)
│   └── structured/
│       ├── validaciones_ee.yaml # revisión manual de los programas de EE (se versiona)
│       ├── extraido/*.csv       # datos extraídos de los programas (generado)
│       ├── sample/*.csv         # asignaturas y seriación FICTICIAS para las pruebas
│       └── plan_estudios.db     # SQLite (generado)
├── src/chatbot/
│   ├── domain/                  # lógica propia; no importa librerías externas
│   │   ├── pipeline.py          #   orquestador del flujo de consulta
│   │   ├── router.py            #   decide la ruta de cada pregunta
│   │   ├── handoff.py           #   reglas de derivación al tutor
│   │   ├── sufficiency.py       #   ¿la evidencia alcanza para responder?
│   │   ├── verifier.py          #   validación de la respuesta generada
│   │   ├── prompts.py           #   prompts, mensajes fijos y controles de omisión
│   │   ├── models.py, text.py   #   tipos y utilidades
│   │   └── ports.py             #   contratos que implementan los adaptadores
│   ├── adapters/                # implementaciones intercambiables de los puertos
│   │   ├── llm/                 #   cliente OpenAI-compatible y cadena de proveedores
│   │   ├── embeddings/          #   sentence-transformers local o API de NVIDIA
│   │   ├── reranker/            #   cross-encoder local o API (NVIDIA, Voyage AI)
│   │   ├── vectorstore/         #   Chroma
│   │   ├── lexical/             #   BM25
│   │   └── structured/          #   SQLite
│   ├── retrieval/hybrid.py      # fusión RRF + reranker
│   ├── ingestion/               # carga, limpieza, fragmentación, índice y
│   │   └── ee_programs.py       #   extracción de los programas de EE + cruce con el mapa
│   ├── privacy/                 # redacción de datos personales y límite de solicitudes
│   ├── observability/           # trazas JSONL por consulta (sin texto del usuario)
│   ├── api/                     # FastAPI: rutas, esquemas y ensamblado del pipeline
│   ├── config.py                # carga de settings.yaml y .env
│   └── tls.py                   # certificados del sistema (ver Problemas frecuentes)
├── web/index.html               # interfaz de chat
├── scripts/extract_ee.py        # programas de EE (PDF) + mapa curricular -> CSV revisables
├── scripts/init_db.py           # crea la base SQLite desde esos CSV
├── eval/
│   ├── dataset/dev.jsonl        # 54 preguntas de evaluación con su resultado esperado
│   ├── run_experiment.py        # ejecuta el conjunto y mide (resultados en eval/results/)
│   └── metrics.py
├── tests/                       # pruebas unitarias
├── logs/                        # traces.jsonl (generado)
├── docs/adr/                    # decisiones de arquitectura (pendientes de redactar)
└── Requerimientos/              # documento de alcance y comparación de componentes
```

La arquitectura es de **puertos y adaptadores**: `domain/` define qué necesita (`ports.py`) y
`adapters/` lo implementa. Cambiar de proveedor de LLM, modelo de embeddings, reranker o almacén
vectorial es editar `settings.yaml` o escribir un adaptador nuevo, sin tocar la lógica.

---

## Puesta en marcha

### Qué se necesita descargar

| Qué | Tamaño aprox. | Cómo se obtiene |
|---|---|---|
| Python 3.11 o superior | — | [python.org](https://www.python.org/downloads/) |
| Dependencias de Python (incluye PyTorch) | 1–2 GB | `pip install` (paso 1) |
| Clave de API de NVIDIA (gratuita) | — | [build.nvidia.com](https://build.nvidia.com/) |
| Clave de API de Groq (gratuita, respaldo) | — | [console.groq.com](https://console.groq.com) |
| *Solo si usas modelos locales:* embeddings `BAAI/bge-m3` | 4.3 GB | automático al construir el índice |
| *Solo si usas modelos locales:* reranker `BAAI/bge-reranker-v2-m3` | 2.2 GB | automático al arrancar la API |
| *Opcional:* Ollama + `llama3.2` | 2 GB | [ollama.com](https://ollama.com), luego `ollama pull llama3.2` |

Con la configuración por defecto (NVIDIA) no se descarga ningún modelo: basta con las
dependencias (1–2 GB). Con todo local son unos 9 GB (11 GB con Ollama), que se guardan en
`~/.cache/huggingface` la primera vez. No se necesita GPU.

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
pytest                            # deben pasar 104 pruebas; no usan red ni modelos
```

**3. Configurar las claves de API**

```bash
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

Edita `.env` y escribe tus claves: `NVIDIA_API_KEY=` (usada por el LLM, los embeddings y el
reranker con la configuración por defecto) y `GROQ_API_KEY=` (LLM de respaldo).
`VOYAGE_API_KEY=` solo hace falta si eliges el reranker de Voyage. Este archivo no se sube al
repositorio.

**4. Crear la base de datos estructurada**

Coloca los programas de EE (PDF) en `documentos_ee_pdf/` (no están en el repositorio) y ejecuta:

```bash
python scripts/extract_ee.py      # PDF + mapa curricular + validaciones -> data/structured/extraido/
python scripts/init_db.py         # CSV -> data/structured/plan_estudios.db (se crea desde cero)
```

`extract_ee.py` aplica la revisión de `data/structured/validaciones_ee.yaml` y marca en la
columna `revisar` lo que no cuadra; `init_db.py` se niega a cargar filas pendientes de revisar
(salvo con `--forzar`). Detén el servidor antes de `init_db.py`: Windows no deja reemplazar la
base mientras está abierta. Sin los PDF, `init_db.py` carga los datos ficticios de
`data/structured/sample/` y lo avisa. La API no arranca el asistente sin la base.

**5. Preparar el corpus y construir el índice**

Los PDF institucionales **no están en el repositorio**: hay que conseguirlos y colocarlos en
`Documentos/` (y los 44 programas de EE en `documentos_ee_pdf/`) con los nombres que indica
`data/raw/manifest.yaml`. Ese archivo lista qué documentos se indexan, una entrada por
documento, con la ruta relativa a `data/raw/`:

```yaml
documents:
  - file: ../../Documentos/05_E_de-los-Alumnos_19-SEP-25_1123h_Version-Compendio-2025.pdf
    doc_id: estatuto_alumnos            # identificador corto y único
    title: "Estatuto de los Alumnos 2008 (compendio 2025)"
    version: "compendio 2025"           # opcional
    valid_from: "2025-09-19"            # opcional; solo si la fecha está confirmada
    vigente: true                       # false = se excluye de las respuestas
    url: "https://..."                  # opcional; enlace que se muestra en la cita
    contexto: "Programa de la EE ..."   # opcional; se antepone a cada fragmento
```

`contexto` se usa en los programas de EE: un fragmento de "Saberes" o "Evaluación" no nombra
la materia, y sin el prefijo no se encuentra al preguntar "¿qué temas ve Networking?". El
`doc_id` de un programa debe ser `programa_ee_<clave>` (con `_` en vez de `-`): si la pregunta
nombra una materia, el asistente descarta los fragmentos de los programas de **otras** materias,
porque temarios y bibliografías se parecen y el LLM los mezclaba. Los nombres de archivo se
encuentran aunque sus acentos estén guardados en otra forma Unicode (los PDF de
`documentos_ee_pdf/` traen acentos descompuestos).

Después:

```bash
python -m chatbot.ingestion.build_index
```

Con el corpus actual genera 591 fragmentos (315 de los programas de EE). Con los embeddings de
NVIDIA tarda menos de un minuto; con `bge-m3` local, la primera vez descarga el modelo y tarda
varios minutos en CPU. Los embeddings se calculan antes de tocar el índice: si fallan, el índice
anterior queda intacto. **Hay que repetir este paso cada vez que se agregue o modifique un
documento**, o se cambie `chunking` o el modelo de embeddings en `settings.yaml`.

**6. Arrancar el chatbot**

```bash
uvicorn chatbot.api.main:app --reload
```

Abre <http://localhost:8000>. Con NVIDIA arranca en unos segundos; con modelos locales, el primer
arranque descarga el reranker y los siguientes tardan unos 35 s en cargar los modelos. Si algo
falta (índice, base, configuración), la API arranca igual pero `/chat` responde 503;
`http://localhost:8000/health` indica si el asistente está listo.

**7. (Opcional) Modelo local de contingencia**

Con Ollama instalado y en ejecución: `ollama pull llama3.2`. Se usa solo cuando NVIDIA y Groq
fallan.

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

## Datos del plan de estudios

La ruta estructurada no depende del LLM para los datos: salen de una base SQLite construida a
partir de los programas oficiales y revisada a mano.

```
documentos_ee_pdf/*.pdf ──┐
mapa_curricular_LISTI.md ─┼─ scripts/extract_ee.py ─→ data/structured/extraido/*.csv ─ scripts/init_db.py ─→ plan_estudios.db
validaciones_ee.yaml ─────┘          (columna `revisar`)                      (rechaza pendientes)
```

1. **Extracción** (`src/chatbot/ingestion/ee_programs.py`): de cada programa se leen los datos
   generales del formato UV: nombre, créditos, horas teóricas, prácticas y otras, área,
   carácter, agrupación, modalidad, equivalencia, prerrequisitos y cupo. Se comprueba que las
   horas cuadren (`(HT + HP) × 15 + otras`) y los créditos (`2 × HT + HP`); lo que no cuadra se
   marca en `revisar`, no se corrige solo.
2. **Cruce con el mapa curricular:** añade el periodo sugerido de cada EE, las 6 EE sin programa
   (las 5 de tronco común y Acreditación del Idioma Inglés) y avisa si el área o el carácter
   difieren o si los créditos por periodo no suman lo que dice el mapa.
3. **Validaciones** (`data/structured/validaciones_ee.yaml`): la revisión con la coordinación,
   que se aplica en cada extracción y sobrevive a una nueva.

   | Clave | Efecto |
   |---|---|
   | `prerrequisitos: []` | confirma que no tiene prerrequisitos (aunque el PDF venga vacío) |
   | `recomendadas: [EE]` | recomendación de cursar antes; **no** es requisito obligatorio |
   | `confirmado: [marca]` | marca de `revisar` revisada y correcta tal como está |
   | `creditos: N` / `area: x` / `nombre: "..."` | dato confirmado que prevalece sobre el PDF y el mapa |
   | `nota: "..."` | aclaración para el estudiante que el chatbot muestra siempre (p. ej. las 480 horas del servicio social) |

   Si una validación nombra una EE que no existe, el proceso se detiene en lugar de ignorarla.

Resultado: **50 EE** (42 obligatorias y un grupo de 8 optativas), 6 prerrequisitos y 2
recomendaciones. Los créditos cuadran con los totales oficiales por área: básica 20, iniciación
a la disciplina 57, disciplinar 213, terminal 52 y elección libre 18, **360 en total**. Las
claves son provisionales (salen del nombre: `programacion`, `desarrollo-web`) porque el campo
"Código" de los programas está vacío.

---

## Configuración

Todo se ajusta en `config/settings.yaml`. Lo más relevante:

| Clave | Para qué sirve |
|---|---|
| `llm.chain` | Proveedores en orden de uso (`nvidia`, `groq`, `huggingface`, `local`). `extra` envía parámetros propios del proveedor (p. ej. `enable_thinking: false` en Nemotron). |
| `llm.max_tokens` | Presupuesto de la respuesta. En modelos de razonamiento incluye el "pensamiento". |
| `llm.retry_429_max_wait_s` | Espera máxima ante límite de cuota antes de pasar al siguiente proveedor. |
| `embeddings.provider` | `nvidia` (API, `nvidia_model`) o `local` (`model`). Cada proveedor tiene su colección en Chroma: al cambiarlo hay que construir su índice una vez (paso 5); después se puede alternar sin reconstruir. |
| `embeddings.model` / `nvidia_model` | Modelo de embeddings. Cambiarlo obliga a reconstruir el índice. |
| `chunking.size_tokens` | Tamaño de fragmento. Cambiarlo obliga a reconstruir el índice. |
| `retrieval.k` | Fragmentos que se envían al LLM. |
| `retrieval.reranker.provider` | `nvidia` o `voyage` (API; requieren `NVIDIA_API_KEY` o `VOYAGE_API_KEY`) o `local` (cross-encoder en este equipo). Con una API no se descarga el modelo local y, si la API falla, se usa el orden de la fusión. |
| `retrieval.reranker.model` | Modelo local de reordenamiento. `""` lo desactiva. |
| `retrieval.reranker.voyage_model` | Modelo de Voyage: `rerank-2.5` o `rerank-2.5-lite`. Sin método de pago, Voyage limita a 3 solicitudes y 10 000 tokens por minuto: insuficiente (cada pregunta envía ~10 000 tokens). |
| `retrieval.reranker.nvidia_model` | Modelo de NVIDIA: `nvidia/nemotron-3.5-rerank-8b`. |
| `sufficiency.rerank_threshold` | Puntaje mínimo del reranker para intentar responder, por modelo (cada uno tiene su escala). Tiene prioridad sobre la similitud vectorial porque tolera erratas y sinónimos. NVIDIA: −2.9 (provisional). |
| `sufficiency.relevance_threshold` | Similitud vectorial mínima; se usa si no hay reranker, si el suyo no tiene umbral o si su API falla (provisional, sin calibrar). |
| `pipeline.verification_retries` | Reintentos cuando la respuesta trae cifras, claves o citas sin respaldo (1 por defecto; 0 = abstenerse directamente). |
| `privacy.rate_limit` | Solicitudes permitidas por ventana de tiempo. |

`config/sensitive_topics.yaml` contiene las frases que disparan la derivación y los contactos que
se muestran al estudiante.

## Rendimiento y límites conocidos

Medido con el corpus actual (55 documentos, 591 fragmentos) en un equipo sin GPU (6 hilos):

- **Con los tres modelos de NVIDIA (configuración por defecto): 2–7 s por respuesta**, p95 ~7 s
  en el conjunto de evaluación (el reranker tarda 1.5–2.5 s). Construir el índice tarda menos de
  un minuto. Contrapartida: la pregunta (ya sin datos personales) y los fragmentos del corpus se
  envían a NVIDIA, que además no publica sus límites de uso gratuito y retira modelos con
  frecuencia (dos de los que aparecían en su documentación ya estaban retirados desde el
  2026-05-18 cuando se probaron).
- **El filtro de contenido de NVIDIA rechaza a veces fragmentos inofensivos** (error 422 "content
  safety screening"; p. ej. un temario de Bases de Datos Distribuidas). Embeddings: el fragmento
  recibe un vector aproximado y `build_index` lo avisa. Reranker: se aísla el pasaje rechazado y
  se reordenan los demás. En ambos casos el texto sigue en la búsqueda léxica.
- **Todo local: unos 20 s por respuesta**, casi toda del reranker. Sin reranker
  (`retrieval.reranker.model: ""`) baja a 2–3 s, pero empeoran las preguntas donde el estudiante
  y el reglamento usan palabras distintas: "¿cuántas veces puedo cursar una experiencia
  educativa?" debe encontrar el artículo que habla de "dos inscripciones". Con GPU el costo del
  reranker sería despreciable.
  Alternativas medidas y descartadas: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (4–7 s, no
  encuentra ese artículo) y `BAAI/bge-reranker-base` (8–10 s, ordena mal en español).
  **El índice local (`bge-m3`) quedó desactualizado** respecto a los 591 fragmentos: hay que
  reconstruirlo antes de volver a `provider: local`.
- **Cuota de Groq (nivel gratuito): 8 000 tokens por minuto**, unas 4 consultas por minuto en
  total, no por usuario. Al agotarse, el cliente espera hasta 10 s y pasa al siguiente.
- **El modelo local casi no sirve de respaldo en CPU:** `llama3.2` supera el tiempo límite de
  60 s en respuestas largas y con frecuencia no cita, por lo que su respuesta se descarta.
- **Las respuestas sobre programas de EE son largas:** el LLM tiende a volcar el temario
  completo aunque el prompt pide brevedad.
- **El LLM no es del todo determinista** aunque la temperatura sea 0: entre dos ejecuciones del
  mismo conjunto puede cambiar alguna respuesta (ver [Pruebas](#pruebas-y-evaluación)).
- **La verificación no detecta errores de interpretación.** Comprueba citas, cifras y claves,
  pero no si el LLM entendió mal una fuente real.
- **Sin memoria de tema entre preguntas:** una pregunta de seguimiento ("¿y cuántos créditos
  tiene?") no hereda la asignatura de la anterior.

## Problemas frecuentes

**`CERTIFICATE_VERIFY_FAILED` al descargar modelos, llamar a una API o usar `pip`.**
Ocurre cuando un antivirus inspecciona el tráfico HTTPS (por ejemplo, Avast). Windows confía en
el certificado del antivirus, pero Python no. Solución: instalar `truststore`; el proyecto lo
detecta y usa los certificados del sistema (`src/chatbot/tls.py`), sin desactivar la verificación.

```bash
pip install truststore
```

Si `pip` falla por el mismo motivo, descarga el archivo `.whl` de
<https://pypi.org/project/truststore/#files> con el navegador e instálalo con
`pip install ruta\al\archivo.whl`. A partir de ahí, `pip install --use-feature=truststore <paquete>`
funciona con normalidad. Para `git push`/`git pull`: `git config --global http.sslBackend schannel`.

**`/chat` responde 503.** El asistente no pudo construirse. Revisa el mensaje
`No se pudo construir el pipeline` en la consola de Uvicorn; lo habitual es que falte el índice
del proveedor de embeddings elegido (paso 5), la base SQLite (paso 4) o una clave en `.env`.

**`PermissionError` / "Acceso denegado" al ejecutar `init_db.py` o `extract_ee.py`.** Windows
bloquea los archivos abiertos: detén el servidor (la base) o cierra Excel (los CSV).

**"El servicio no está disponible en este momento".** Fallaron todos los proveedores de LLM;
normalmente es una cuota agotada. Espera un minuto.

**El asistente se abstiene demasiado.** Consulta `logs/traces.jsonl`: cada línea registra la
decisión, el motivo (`sufficiency_reason`, `verification_problems`, `max_rerank_score`), los
fragmentos recuperados y los tiempos por etapa. `corregido_en_reintento:...` indica que la
primera versión traía un dato sin respaldo y la segunda se aceptó.

---

## Pruebas y evaluación

```bash
pytest                                                        # pruebas unitarias (sin red)
python eval/run_experiment.py eval/dataset/dev.jsonl etiqueta  # evaluación con el pipeline real
```

**Pruebas unitarias (100):** dominio (pipeline, enrutador, suficiencia, verificador, seriación),
redactor de datos personales, BM25, fusión, fragmentador, extractor de programas de EE y cruce
con el mapa, repositorio SQLite, clientes de LLM, embeddings y rerankers por API (con respuestas
simuladas, incluidos los errores 429, 410 y 422 de filtro de contenido) y la API.

**Evaluación:** `eval/dataset/dev.jsonl` tiene 54 preguntas con su resultado esperado:

| Categoría | n | Ejemplos |
|---|---|---|
| Seriación | 9 | "¿Qué necesito para cursar Programación?", requisitos de Administración de Servidores (recomendación) |
| Datos de las materias | 12 | créditos de Sistemas Operativos, horas de Servicio Social, semestre de Ciencia de Datos |
| Programas de EE | 14 | temas de Bases de Datos Relacionales, evaluación de Networking, libros de Desarrollo Web |
| Mapa curricular | 7 | créditos totales, optativas, "cuantas mareias hay" (con errata) |
| Reglamentos y trámites | 6 | inscripciones, PAFI, centro de cómputo, inglés |
| Sin evidencia (debe abstenerse) | 4 | costo del comedor, becas de transporte, "el mejor profesor" |
| Derivación / fuera de alcance | 2 | ansiedad, receta de cocina |

Cada pregunta indica la decisión y la ruta esperadas, los hechos que deben aparecer en la
respuesta (sin distinguir mayúsculas ni acentos; se admiten alternativas como `["6", "sexto"]`)
y los documentos que deberían recuperarse. Los hechos esperados salen de las fuentes
(programas, validaciones y mapa), no de las respuestas del chatbot. El script imprime OK/MAL por
pregunta, resume por categoría y guarda respuestas y trazas en `eval/results/`.

| Ejecución | Todo bien | Notas |
|---|---|---|
| Primera | 46/54 (85 %) | 4 fallos eran expectativas mal planteadas; 4 eran fallos reales |
| Tras las correcciones | 54/54, 53/54, 54/54 | el 53/54 fue "No hay becas de transporte" (corregido después) |
| Con preguntas de seguimiento | 54/54 | sin regresiones; los seguimientos se probaron a mano (el conjunto es de una sola pregunta por caso) |

Recuperación del documento correcto: 100 %; ruta correcta: 100 %. **Ojo:** el sistema se ajustó
viendo este conjunto (dev). Para la tesis hace falta un conjunto *holdout* con preguntas
redactadas por otras personas (tutores, estudiantes) que solo se ejecute al final, sin ajustar
nada después.

---

## Historial de pruebas y cambios

| Fecha | Commit | Qué se probó o cambió | Resultado |
|---|---|---|---|
| 2026-11-11 | `0592efe` | Esqueleto del proyecto (pipeline, puertos y adaptadores) | — |
| 2026-11-16 | `b3510ba` | Primer corpus real (11 documentos). Correcciones: marcadores de datos personales tomados como entidades, vigencia rechazada por el verificador, citas 【1】 de gpt-oss, respuestas cortadas, cuota de Groq. Fragmentador reescrito. Reranker local `bge-reranker-v2-m3` | 9/9 correctas, ~20 s por respuesta |
| 2026-11-22 | `3920740` (rama `pruebas-voyage`) | Reranker por API de Voyage AI | Rápido (1–2 s) y buen orden, pero sin método de pago solo admite 3 solicitudes y 10 000 tokens por minuto: no alcanza |
| 2026-11-22 | `8305035` (rama `pruebas-nvidia`) | LLM, embeddings y reranker de NVIDIA. Umbral de suficiencia sobre el puntaje del reranker (tolera erratas). Prompt con ejemplo de cita | 9/9 correctas, 4–7 s; **se adopta como configuración por defecto** |
| 2026-11-22 | `0f27ad3` | Datos reales del plan: extracción de los 44 programas, cruce con el mapa, validaciones con la coordinación, recomendaciones de seriación. Mapa corregido: total 360 créditos (no 392), áreas y créditos por periodo | Créditos cuadran con los totales oficiales por área |
| 2026-11-22 | `70122f0` | Base real cargada; 44 programas indexados con contexto y filtro por materia; reintento de verificación; abstenciones del propio LLM; manejo del filtro de contenido de NVIDIA; conjunto de evaluación de 54 preguntas | 54/54 en desarrollo |
| 2026-11-23 | — | Preguntas de seguimiento: "¿y general?" se abstenía porque se buscaba sin contexto. Ahora se busca con la consulta anterior antepuesta. Al mapa se le añadió una oración con los créditos de cada área, porque el LLM confundía las filas de la tabla (respondía 18 en vez de 20 para el área general) | 54/54; seguimientos correctos ("¿y general?" → 20, "¿y disciplinar?" → 213, "¿y cuántas horas?" tras Networking → 90) |

Las ramas `pruebas-voyage` y `pruebas-nvidia` conservan el estado de cada prueba; `main` incluye
todo.

## Pendientes

- [ ] **Conjunto holdout** de evaluación, redactado por tutores o estudiantes y ejecutado solo al
      final.
- [ ] **Casos de conversación en la evaluación** (varias preguntas seguidas), para medir los
      seguimientos; hoy cada caso es una sola pregunta.
- [ ] **Recalibrar `sufficiency.rerank_threshold`** (−2.9) con más preguntas: se fijó con 21,
      antes de añadir los 315 fragmentos de los programas.
- [ ] **Corpus:** faltan documentos; el Reglamento de Movilidad 2015 está escaneado y requiere OCR.
- [ ] **Autorización** de cada documento y fechas de vigencia (`valid_from`) en el manifiesto.
- [ ] **Claves oficiales de las EE:** el campo "Código" de los programas está vacío y el del mapa
      se repite entre materias; hoy la clave sale del nombre (`programacion`, `desarrollo-web`).
- [ ] **Contactos de ayuda** en `config/sensitive_topics.yaml` (siguen como `REEMPLAZAR`) y lista
      de temas sensibles validada con tutores. Algunas frases derivan de más (p. ej. "tramita"
      deriva preguntas informativas sobre trámites).
- [ ] **Dependencia de NVIDIA:** decidir si es aceptable enviar el corpus y las preguntas a un
      servicio externo, y tener un plan si cambian los modelos o los límites gratuitos.
- [ ] **Nombres de académicos:** las respuestas sobre programas pueden mencionar a quienes los
      elaboraron (sección 27); decidir si se excluye esa sección del índice.
- [ ] **Brevedad** de las respuestas sobre programas de EE.
- [ ] **Límite de solicitudes:** se puede evadir cambiando el `session_id`; debe basarse en la IP.
- [ ] **Formato real de matrícula** en `privacy/redactor.py` (se asumió `zS` + 8 dígitos).
- [ ] **Aviso de privacidad** definitivo en `web/index.html` y política de retención de `logs/`.
- [ ] **Versiones exactas** de las dependencias `rag` y de `truststore` en `pyproject.toml`.
- [ ] **Dockerfile:** no incluye `data/` ni `Documentos/`; requiere volúmenes para funcionar.
- [ ] **Decisiones de arquitectura** ADR-001 a ADR-007 en `docs/adr/`.
