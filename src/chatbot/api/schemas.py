from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    label: str
    section: str = ""
    page: Optional[int] = None
    url: str = ""
    valid_from: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    decision: str
    route: str
    sources: list[SourceOut] = []
