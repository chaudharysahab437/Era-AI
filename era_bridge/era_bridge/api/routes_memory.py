"""/v1/memory/* -- ERA's persistent-memory API.

Protected two ways: the same X-Era-Bridge-Key header as every other Bridge
route, and (unless MEMORY_LOCAL_ONLY=false) a check that the caller is on
this machine. The Gemini Live frontend uses two calls per turn:

    POST /v1/memory/context   before generating a reply  -> context text
    POST /v1/memory/ingest    after the reply            -> extracts + saves
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.auth import verify_api_key
from era_memory import (
    Category, MemoryManager, MemoryNotFoundError, MemoryType, MemoryUnavailableError, SecretContentError,
    SensitiveContentError,
)

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}  # "testclient" = Starlette's in-process TestClient


async def require_local(request: Request) -> None:
    from config import settings

    if not getattr(settings, "MEMORY_LOCAL_ONLY", True):
        return
    host = request.client.host if request.client else ""
    if host not in _LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="Memory endpoints are only available from this machine.")


router = APIRouter(prefix="/v1/memory", dependencies=[Depends(verify_api_key), Depends(require_local)], tags=["memory"])


def get_memory_manager() -> MemoryManager:
    from bridge_server import memory_manager  # lazy: avoids a circular import, mirrors the other routes

    return memory_manager


# ------------------------------------------------------------------ schemas --
class ContextRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    project: Optional[str] = None


class IngestRequest(BaseModel):
    user_message: str
    assistant_message: Optional[str] = None
    conversation_id: str
    project: Optional[str] = None


class SaveRequest(BaseModel):
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    category: Category = Category.NOTE
    importance: float = Field(0.6, ge=0, le=1)
    confidence: float = Field(0.95, ge=0, le=1)
    project: Optional[str] = None
    conversation_id: Optional[str] = None
    allow_sensitive: bool = False
    share_with_model: Optional[bool] = None


class UpdateRequest(BaseModel):
    content: Optional[str] = None
    type: Optional[MemoryType] = None
    category: Optional[Category] = None
    importance: Optional[float] = Field(None, ge=0, le=1)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    confidence_level: Optional[str] = None
    share_with_model: Optional[bool] = None
    allow_sensitive: bool = False


class SettingsRequest(BaseModel):
    auto_memory: bool


def _wrap(exc: Exception) -> HTTPException:
    if isinstance(exc, MemoryNotFoundError):
        return HTTPException(404, detail=f"Not found: {exc}")
    if isinstance(exc, SecretContentError):
        return HTTPException(422, detail={"code": "secret_not_storable", "kinds": exc.kinds,
                                          "message": "Credentials, keys and tokens are never stored."})
    if isinstance(exc, SensitiveContentError):
        return HTTPException(422, detail={"code": "sensitive_requires_opt_in", "categories": exc.categories,
                                          "message": "This looks sensitive. Repeat the request with allow_sensitive=true to store it."})
    if isinstance(exc, MemoryUnavailableError):
        return HTTPException(503, detail={"code": "memory_unavailable", "error": str(exc),
                                          "message": "Memory storage is unavailable. Everything else in ERA is unaffected."})
    if isinstance(exc, ValueError):
        return HTTPException(422, detail=str(exc))
    raise exc


# ------------------------------------------------------ per-turn (frontend) --
@router.post("/context")
async def get_context(req: ContextRequest, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    """Retrieve relevant memories for a message and return the budgeted context package."""
    pkg = await mm.abuild_context(req.message, req.conversation_id, req.project)
    return pkg.to_dict()


@router.post("/ingest")
async def ingest(req: IngestRequest, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    """Extract + persist from a finished turn. Safe to call fire-and-forget."""
    res = await mm.aingest_turn(req.user_message, req.assistant_message, conversation_id=req.conversation_id, project=req.project)
    return res.to_dict()


# --------------------------------------------------------------- settings ----
@router.get("/settings")
async def get_settings(mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        return mm.settings()
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


@router.post("/settings")
async def set_settings(req: SettingsRequest, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        mm.set_auto(req.auto_memory)
        return mm.settings()
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


@router.get("/stats")
async def stats(mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        return mm.stats()
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


@router.post("/reembed")
async def reembed(limit: int = Query(500, ge=1, le=5000), mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    """Re-embed memories whose vectors came from a different provider/model/dimension."""
    try:
        return await __import__("asyncio").to_thread(mm.reembed, limit)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


# --------------------------------------------------------------- projects ----
@router.get("/projects")
async def projects(mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        return {"projects": [{"id": p.id, "name": p.name, "updated_at": p.updated_at} for p in mm.list_projects()]}
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


@router.get("/projects/{project}/summary")
async def project_summary(project: str, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        return mm.project_summary(project)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


# ---------------------------------------------------------- conversations ----
@router.get("/conversations/{conversation_id}")
async def conversation(conversation_id: str, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        c = mm.conversation(conversation_id)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    return {k: getattr(c, k) for k in ("id", "project_id", "active_task", "summary", "turn_count", "created_at", "updated_at", "closed_at")}


@router.post("/conversations/{conversation_id}/close")
async def close_conversation(conversation_id: str, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        return mm.close_conversation(conversation_id)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


# ------------------------------------------------------------- CRUD/search ---
@router.get("/search")
async def search(
    q: str, limit: int = Query(10, ge=1, le=50), project: Optional[str] = None, reveal: bool = False,
    mm: MemoryManager = Depends(get_memory_manager),
) -> Dict[str, Any]:
    try:
        hits = mm.search(q, limit=limit, project=project)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    return {"results": [{**h.memory.to_dict(reveal), "score": round(h.score, 3), "relevance": round(h.relevance, 3)} for h in hits]}


@router.post("", status_code=201)
async def save(req: SaveRequest, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        out = mm.save(
            req.content, type=req.type, category=req.category, importance=req.importance, confidence=req.confidence,
            project=req.project, conversation_id=req.conversation_id, allow_sensitive=req.allow_sensitive,
            share_with_model=req.share_with_model,
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    return {"action": out.action, "memory": out.memory.to_dict(), "superseded": out.superseded_ids}


@router.get("")
async def list_memories(
    type: Optional[MemoryType] = None, category: Optional[Category] = None, project: Optional[str] = None,
    q: Optional[str] = None, include_history: bool = False, reveal: bool = False,
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
    mm: MemoryManager = Depends(get_memory_manager),
) -> Dict[str, Any]:
    try:
        pid = None
        if project:
            p = mm.get_or_create_project(project, create=False)
            if p is None:
                return {"memories": [], "count": 0}
            pid = p.id
        items = mm.list(
            types=[type.value] if type else None, categories=[category.value] if category else None, project_id=pid,
            text=q, include_history=include_history, limit=limit, offset=offset,
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    return {"memories": [m.to_dict(reveal) for m in items], "count": len(items)}


@router.delete("")
async def clear_all(confirm: bool = False, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    if not confirm:
        raise HTTPException(400, detail="This deletes ALL memories, conversations and projects. Repeat with ?confirm=true.")
    try:
        return {"deleted": mm.clear_all()}
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


@router.get("/{memory_id}")
async def get_one(memory_id: str, reveal: bool = False, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        m = mm.get(memory_id)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    if reveal and m.is_sensitive:
        logger.info("Sensitive memory revealed: id=%s", memory_id)
    return m.to_dict(reveal)


@router.patch("/{memory_id}")
async def update(memory_id: str, req: UpdateRequest, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        m = mm.update(memory_id, **req.model_dump(exclude_unset=True))
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    return m.to_dict()


@router.post("/{memory_id}/resolve")
async def resolve(memory_id: str, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    """Confirm this memory as the correct side of a flagged conflict."""
    try:
        return mm.resolve_conflict(memory_id).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)


@router.delete("/{memory_id}")
async def delete(memory_id: str, mm: MemoryManager = Depends(get_memory_manager)) -> Dict[str, Any]:
    try:
        ok = mm.delete(memory_id)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc)
    if not ok:
        raise HTTPException(404, detail=f"Not found: {memory_id}")
    return {"deleted": memory_id}
