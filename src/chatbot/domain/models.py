"""Modelos del dominio. Solo biblioteca estándar: nada de LangChain, Groq ni Chroma."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Route(str, Enum):
    STRUCTURED = "structured"
    DOCUMENTAL = "documental"
    HANDOFF = "handoff"
    OUT_OF_SCOPE = "out_of_scope"


class Decision(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    HANDOFF = "handoff"
    OUT_OF_SCOPE = "out_of_scope"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    doc_id: str
    doc_title: str = ""
    version: str = ""
    valid_from: str = ""
    section: str = ""
    page: Optional[int] = None
    url: str = ""
    vigente: bool = True

    def to_metadata(self) -> dict:
        """Metadatos planos (sin None) para almacenes vectoriales."""
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "version": self.version,
            "valid_from": self.valid_from,
            "section": self.section,
            "page": -1 if self.page is None else int(self.page),
            "url": self.url,
            "vigente": bool(self.vigente),
        }

    @classmethod
    def from_metadata(cls, chunk_id: str, text: str, meta: dict) -> "Chunk":
        page = meta.get("page", -1)
        return cls(
            id=chunk_id,
            text=text,
            doc_id=meta.get("doc_id", ""),
            doc_title=meta.get("doc_title", ""),
            version=meta.get("version", ""),
            valid_from=meta.get("valid_from", ""),
            section=meta.get("section", ""),
            page=None if page in (-1, None) else int(page),
            url=meta.get("url", ""),
            vigente=bool(meta.get("vigente", True)),
        )


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float            # puntaje de fusión (RRF) o del índice que lo devolvió
    relevance: float = 0.0  # similitud vectorial [0-1]; base de la evaluación de suficiencia
    rerank_score: Optional[float] = None  # puntaje del reranker, si lo hubo (escala del modelo)


@dataclass(frozen=True)
class Prerequisite:
    clave: str
    nombre: str
    tipo: str  # "previa" | "simultanea"


@dataclass(frozen=True)
class SubjectFact:
    clave: str
    nombre: str
    creditos: Optional[int]
    periodo: Optional[str]
    area: Optional[str]
    plan_vigente: Optional[str]
    requisitos: tuple[Prerequisite, ...] = ()


@dataclass(frozen=True)
class Citation:
    label: str
    doc_id: str = ""
    section: str = ""
    page: Optional[int] = None
    url: str = ""
    valid_from: str = ""


@dataclass
class LLMResponse:
    text: str
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RouteDecision:
    route: Route
    reason: str = ""
    subjects: tuple[str, ...] = ()
    handoff_kind: str = ""  # risk | sensitive | requested | administrative


@dataclass
class Trace:
    """Traza por consulta SIN texto del usuario (RNF5, RNF8)."""
    trace_id: str
    session_id: str = ""
    route: str = ""
    decision: str = ""
    pii_types: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    model_id: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sufficiency_reason: str = ""
    max_relevance: float = 0.0
    max_rerank_score: Optional[float] = None
    verification_problems: list[str] = field(default_factory=list)
    stage_ms: dict = field(default_factory=dict)
    total_ms: float = 0.0
    versions: dict = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    decision: Decision
    route: Route
    citations: list[Citation] = field(default_factory=list)
    trace: Optional[Trace] = None


@dataclass
class Session:
    session_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=8))  # (usuario, asistente)
    insufficient_count: int = 0
