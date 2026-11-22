"""Orquestador del flujo de consulta (sección 7.2).

sanitizar -> enrutar -> evidencia -> suficiencia -> generar -> verificar -> presentar.
Depende solo de puertos; no conoce Groq, Chroma ni LangChain.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from . import prompts
from .handoff import HandoffRules
from .models import (Answer, Citation, Decision, LLMResponse, Route, ScoredChunk,
                     Session, SubjectFact, Trace)
from .ports import LLMProvider, ProviderError, Retriever, StructuredRepo
from .router import Router
from .sufficiency import SufficiencyChecker
from .verifier import AnswerVerifier, normalize_citations


@dataclass
class PipelineSettings:
    max_query_chars: int = 500
    history_turns: int = 4
    max_insufficient_before_handoff: int = 2
    temperature: float = 0.0
    max_tokens: int = 600
    verification_retries: int = 1  # reintentos si la respuesta trae datos sin respaldo


PROGRAM_PREFIX = "programa_ee_"  # doc_id de los programas de EE: programa_ee_<clave con _>


def program_doc_id(clave: str) -> str:
    return PROGRAM_PREFIX + clave.replace("-", "_")


def focus_on_subjects(chunks: Sequence[ScoredChunk], subjects: Sequence[str]) -> list[ScoredChunk]:
    """Si la pregunta nombra materias y hay fragmentos de SUS programas, descarta los de los
    programas de otras materias (las bibliografías y temarios se parecen y el LLM los mezcla).
    Los demás documentos (reglamentos, mapa curricular...) no se tocan."""
    wanted = {program_doc_id(s) for s in subjects}
    if not wanted or not any(c.chunk.doc_id in wanted for c in chunks):
        return list(chunks)
    return [c for c in chunks
            if not c.chunk.doc_id.startswith(PROGRAM_PREFIX) or c.chunk.doc_id in wanted]


class ChatPipeline:
    def __init__(self, *, router: Router, retriever: Retriever, repo: StructuredRepo,
                 llms: Sequence[LLMProvider], redactor, sufficiency: SufficiencyChecker,
                 verifier: AnswerVerifier, handoff: HandoffRules, tracer=None,
                 settings: Optional[PipelineSettings] = None,
                 versions: Optional[dict] = None):
        self.router, self.retriever, self.repo = router, retriever, repo
        self.llms, self.redactor = list(llms), redactor
        self.sufficiency, self.verifier, self.handoff = sufficiency, verifier, handoff
        self.tracer = tracer
        self.s = settings or PipelineSettings()
        self.versions = versions or {}

    # ---------------------------------------------------------------- API pública
    def answer(self, message: str, session: Optional[Session] = None) -> Answer:
        session = session or Session(session_id=uuid.uuid4().hex)
        t0 = time.perf_counter()
        trace = Trace(trace_id=uuid.uuid4().hex, session_id=session.session_id,
                      versions=dict(self.versions))
        clean = {"q": ""}
        try:
            ans = self._run(message, session, trace, clean)
        finally:
            trace.total_ms = round((time.perf_counter() - t0) * 1000, 1)
        trace.decision = ans.decision.value
        trace.route = ans.route.value
        ans.trace = trace
        if self.tracer:
            self.tracer.emit(trace)
        if ans.decision == Decision.ANSWERED:
            session.insufficient_count = 0
        # Se guarda el texto ya redactado (sin PII) para dar contexto a la siguiente pregunta.
        session.history.append((clean["q"], ans.text))
        return ans

    # ---------------------------------------------------------------- flujo interno
    def _run(self, message: str, session: Session, trace: Trace, clean: dict) -> Answer:
        with self._stage(trace, "sanitize"):
            text = message.strip()[: self.s.max_query_chars]
            red = self.redactor.redact(text)
            clean["q"] = red.text
            trace.pii_types = list(red.types)
        query = red.text

        with self._stage(trace, "route"):
            decision = self.router.route(query)

        if decision.route == Route.HANDOFF:
            return Answer(self.handoff.message(decision.handoff_kind), Decision.HANDOFF,
                          Route.HANDOFF)
        if decision.route == Route.OUT_OF_SCOPE:
            return Answer(prompts.OUT_OF_SCOPE_MSG, Decision.OUT_OF_SCOPE, Route.OUT_OF_SCOPE)
        if decision.route == Route.STRUCTURED:
            return self._structured(query, decision.subjects, trace)
        return self._documental(query, session, trace, decision.subjects)

    def _structured(self, query: str, subjects: Sequence[str], trace: Trace) -> Answer:
        with self._stage(trace, "evidence"):
            facts = [f for f in (self.repo.get_facts(c) for c in subjects) if f]
        if not facts:
            return Answer(prompts.ABSTAIN_MSG, Decision.ABSTAINED, Route.STRUCTURED)

        source = prompts.facts_text(facts)
        fallback = prompts.deterministic_structured_answer(facts)
        citation = [Citation(label="Plan de estudios (base de datos estructurada)",
                             doc_id="plan_estudios")]
        try:
            with self._stage(trace, "generate"):
                resp = self._generate(prompts.build_structured_messages(query, facts), trace)
            with self._stage(trace, "verify"):
                ver = self.verifier.verify(resp.text, [source], require_citations=False)
            trace.verification_problems = ver.problems
            text = resp.text if ver.ok else fallback  # la seriación nunca depende del LLM
            if ver.ok:
                omitidos, recomendadas = prompts.missing_requirements(resp.text, facts)
                if omitidos and prompts.mentions_requirements(resp.text):
                    # Habló de requisitos y dejó uno fuera (o dijo que no hay): tan grave como
                    # inventarlo. Se entrega la respuesta determinista.
                    trace.verification_problems = [f"requisito_omitido:{r.clave}" for _, r in omitidos]
                    text = fallback
                else:
                    # No habló de requisitos (p. ej. preguntaron solo los créditos): se añaden.
                    text += "".join(
                        f"\n\nRequisito de seriación: haber cursado {r.nombre} ({r.tipo})."
                        for _, r in omitidos)
                    text += "".join(
                        f"\n\nRecomendación (no es requisito obligatorio): cursar {r.nombre} "
                        f"antes de {f.nombre}." for f, r in recomendadas)
                    # Las notas validadas (p. ej. "480 horas comprobables" del servicio social)
                    # corrigen o matizan el dato del programa: nunca se omiten.
                    text += "".join(f"\n\nNota sobre {f.nombre}: {f.nota}"
                                    for f in prompts.omitted_notes(resp.text, facts))
        except ProviderError:
            text = fallback
        return Answer(text, Decision.ANSWERED, Route.STRUCTURED, citation)

    def _documental(self, query: str, session: Session, trace: Trace,
                    subjects: Sequence[str] = ()) -> Answer:
        with self._stage(trace, "evidence"):
            chunks = focus_on_subjects(self.retriever.retrieve(query), subjects)
        trace.retrieved_ids = [c.chunk.id for c in chunks]

        with self._stage(trace, "sufficiency"):
            suff = self.sufficiency.check(query, chunks)
        trace.sufficiency_reason, trace.max_relevance = suff.reason, suff.max_relevance
        trace.max_rerank_score = suff.max_rerank_score
        if not suff.sufficient:
            return self._abstain_or_handoff(session)

        history = list(session.history)[-self.s.history_turns:]
        messages = prompts.build_messages(query, chunks, history)
        sources = [prompts.verification_source(c) for c in chunks]
        try:
            with self._stage(trace, "generate"):
                resp = self._generate(messages, trace)
            with self._stage(trace, "verify"):
                ver = self.verifier.verify(resp.text, sources)
            trace.verification_problems = ver.problems
            for _ in range(self.s.verification_retries):
                if ver.ok:
                    break
                # Suele ser un detalle (un subtotal calculado, una cifra reescrita): se le
                # señala al LLM y se le pide de nuevo. Si vuelve a fallar, se abstiene.
                retry = messages + [{"role": "assistant", "content": resp.text},
                                    {"role": "user", "content": prompts.retry_instruction(ver.problems, len(chunks))}]
                with self._stage(trace, "generate_retry"):
                    resp = self._generate(retry, trace)
                with self._stage(trace, "verify_retry"):
                    first, ver = ver.problems, self.verifier.verify(resp.text, sources)
                trace.verification_problems = (
                    [f"corregido_en_reintento:{p}" for p in first] if ver.ok
                    else first + [f"reintento:{p}" for p in ver.problems])
        except ProviderError:
            return Answer(prompts.UNAVAILABLE_MSG, Decision.UNAVAILABLE, Route.DOCUMENTAL)
        if not ver.ok:
            return self._abstain_or_handoff(session)
        if prompts.is_self_abstention(resp.text):
            # El propio LLM dice que las fuentes no lo cubren ("No cuento con información...").
            # Cuenta como abstención (y para la derivación), conservando su explicación.
            ans = self._abstain_or_handoff(session)
            if ans.decision == Decision.ABSTAINED:
                ans.text = resp.text
            return ans

        cited = [chunks[i - 1] for i in ver.cited_indexes]
        trace.cited_ids = [c.chunk.id for c in cited]
        citations = [Citation(label=c.chunk.doc_title or c.chunk.doc_id, doc_id=c.chunk.doc_id,
                              section=c.chunk.section, page=c.chunk.page, url=c.chunk.url,
                              valid_from=c.chunk.valid_from) for c in cited]
        return Answer(resp.text, Decision.ANSWERED, Route.DOCUMENTAL, citations)

    def _abstain_or_handoff(self, session: Session) -> Answer:
        session.insufficient_count += 1
        if session.insufficient_count >= self.s.max_insufficient_before_handoff:
            return Answer(self.handoff.repeated_insufficient_message(), Decision.HANDOFF,
                          Route.DOCUMENTAL)
        return Answer(prompts.ABSTAIN_MSG, Decision.ABSTAINED, Route.DOCUMENTAL)

    def _generate(self, messages: list[dict], trace: Trace) -> LLMResponse:
        """Recorre la cadena de contingencia (sección 3.4) hasta que un proveedor responda."""
        last: Optional[Exception] = None
        for llm in self.llms:
            try:
                resp = llm.generate(messages, temperature=self.s.temperature,
                                    max_tokens=self.s.max_tokens)
            except ProviderError as exc:
                last = exc
                continue
            resp.text = normalize_citations(resp.text)
            trace.model_id, trace.provider = resp.model_id, resp.provider
            trace.prompt_tokens += resp.prompt_tokens
            trace.completion_tokens += resp.completion_tokens
            return resp
        raise ProviderError(f"Todos los proveedores fallaron: {last}")

    # ---------------------------------------------------------------- utilidades
    def _stage(self, trace: Trace, name: str):
        return _Stage(trace, name)


class _Stage:
    def __init__(self, trace: Trace, name: str):
        self.trace, self.name = trace, name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        self.trace.stage_ms[self.name] = round((time.perf_counter() - self.t) * 1000, 1)
