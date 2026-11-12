from __future__ import annotations

import uuid
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Request

from ..domain.models import Session
from ..privacy.rate_limit import SlidingWindowLimiter
from .schemas import ChatRequest, ChatResponse, SourceOut

router = APIRouter()
_MAX_SESSIONS = 1000
_sessions: "OrderedDict[str, Session]" = OrderedDict()  # en memoria; se pierde al reiniciar
_limiter = SlidingWindowLimiter()


def configure_limiter(max_calls: int, window_s: float) -> None:
    global _limiter
    _limiter = SlidingWindowLimiter(max_calls, window_s)


def _get_session(session_id: str | None) -> Session:
    sid = session_id or uuid.uuid4().hex
    if sid not in _sessions:
        _sessions[sid] = Session(session_id=sid)
        if len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
    _sessions.move_to_end(sid)
    return _sessions[sid]


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "pipeline_ready": getattr(request.app.state, "pipeline", None) is not None}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "El asistente no está listo (índice o configuración pendientes).")
    client = request.client.host if request.client else "anon"
    if not _limiter.allow(f"{client}:{req.session_id or ''}"):
        raise HTTPException(429, "Demasiadas solicitudes. Intenta de nuevo en un momento.")

    session = _get_session(req.session_id)
    ans = pipeline.answer(req.message, session)
    return ChatResponse(
        session_id=session.session_id, answer=ans.text, decision=ans.decision.value,
        route=ans.route.value,
        sources=[SourceOut(label=c.label, section=c.section, page=c.page, url=c.url,
                           valid_from=c.valid_from) for c in ans.citations])
