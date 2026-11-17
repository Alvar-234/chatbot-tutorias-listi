"""Punto de entrada: uvicorn chatbot.api.main:app"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import load_settings
from ..tls import use_system_certificates
from .routes import configure_limiter, router

log = logging.getLogger("chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    use_system_certificates()
    try:
        settings = load_settings()
        rl = settings["privacy"]["rate_limit"]
        configure_limiter(rl["max_calls"], rl["window_s"])
        from .deps import build_pipeline
        app.state.pipeline = build_pipeline(settings)
    except Exception as exc:  # la API arranca y /health informa; /chat responde 503
        log.error("No se pudo construir el pipeline: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Chatbot de tutorías LISTI", lifespan=lifespan)
    app.include_router(router)
    web = Path("web")
    if web.exists():
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")
    return app


app = create_app()
