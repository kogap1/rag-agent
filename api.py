from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from agentic_rag import AgenticRAG, Settings
from agentic_rag.document_catalog import RollbackUnavailableError
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.mcp_server import create_mcp_server
from agentic_rag.models import ModelRuntime
from agentic_rag.service_metrics import ServiceMetrics


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


class EvidenceResponse(BaseModel):
    citation: str
    chunk_id: str | None = None
    content: str
    score: float
    source: str
    page: int


class QueryResponse(BaseModel):
    request_id: str
    run_id: str | None = None
    answer: str
    grounded: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    evidence: list[EvidenceResponse]
    trace: list[dict[str, str]]


class RollbackRequest(BaseModel):
    source: str = Field(min_length=1, max_length=255)
    version_id: str | None = Field(default=None, max_length=64)


class ServiceContainer:
    def __init__(self, settings: Settings):
        settings.ensure_directories()
        runtime = ModelRuntime(settings)
        self.settings = settings
        self.kb = KnowledgeBase(settings, runtime)
        self.agent = AgenticRAG(runtime, self.kb, settings)
        self.capacity = threading.BoundedSemaphore(settings.api_max_concurrency)
        self.ingestion_lock = threading.Lock()

    def query(self, payload: QueryRequest, request_id: str):
        if not self.capacity.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="服务繁忙，请稍后重试")
        try:
            return self.agent.run(payload.question.strip(), payload.history, request_id=request_id)
        finally:
            self.capacity.release()

    def ingest(self, path: Path, source_name: str) -> int:
        if not self.ingestion_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="已有文档入库任务正在执行")
        try:
            return self.kb.ingest(path, source_name=source_name)
        finally:
            self.ingestion_lock.release()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    if settings.api_max_concurrency < 1:
        raise RuntimeError("API_MAX_CONCURRENCY 必须大于 0")
    return settings


@lru_cache(maxsize=1)
def get_services() -> ServiceContainer:
    return ServiceContainer(get_settings())


def _mcp_query(question: str, history: list[dict[str, str]], request_id: str):
    payload = QueryRequest(question=question, history=history)
    return get_services().query(payload, request_id)


def _mcp_search(query: str, top_k: int | None = None) -> list[dict]:
    hits = get_services().kb.search(query)
    if top_k:
        hits = hits[:top_k]
    return [
        {
            "citation": hit.citation,
            "chunk_id": hit.chunk_id,
            "content": hit.content,
            "score": hit.score,
            "source": hit.source,
            "page": hit.page,
        }
        for hit in hits
    ]


def _mcp_documents() -> list[dict]:
    return [record.__dict__ for record in get_services().kb.document_records()]


def _mcp_health() -> dict:
    services = get_services()
    return {
        "status": "ready" if services.kb.health()["integrity"] == "ok" else "degraded",
        "catalog": services.kb.health(),
        "llm_backend": services.settings.llm_backend,
        "orchestration": "langgraph",
    }


metrics = ServiceMetrics()
mcp_server = create_mcp_server(_mcp_query, _mcp_search, _mcp_documents, _mcp_health)
_mcp_settings = get_settings()
mcp_asgi = mcp_server.streamable_http_app(
    streamable_http_path="/",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(_mcp_settings.mcp_allowed_hosts),
        allowed_origins=list(_mcp_settings.mcp_allowed_origins),
    ),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(
    title="Agentic RAG Service",
    version="3.0.0",
    description="LangGraph编排、FastAPI与MCP双入口的可解释知识库Agent服务",
    lifespan=lifespan,
)
app.mount("/mcp", mcp_asgi, name="mcp")


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _api_key_valid(provided: str | None) -> bool:
    configured = get_settings().api_key
    return not configured or bool(provided and secrets.compare_digest(configured, provided))


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    if not _api_key_valid(x_api_key or _bearer_token(authorization)):
        raise HTTPException(status_code=401, detail="API key 无效")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    metrics.request_started()
    status_code = 500
    try:
        if request.url.path.startswith("/mcp"):
            provided = request.headers.get("X-API-Key") or _bearer_token(
                request.headers.get("Authorization")
            )
            if not _api_key_valid(provided):
                response = JSONResponse(status_code=401, content={"detail": "API key 无效"})
                status_code = response.status_code
                return response
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        metrics.record_error(type(exc).__name__)
        raise
    finally:
        metrics.request_finished(
            request.url.path,
            status_code,
            (time.perf_counter() - started) * 1000,
        )
        # response is not guaranteed to exist on an unhandled exception.
        if "response" in locals():
            response.headers["X-Request-ID"] = request_id


@app.get("/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/health/ready", dependencies=[Depends(require_api_key)])
def ready() -> dict:
    services = get_services()
    catalog = services.kb.health()
    ready_state = catalog["integrity"] == "ok"
    if not ready_state:
        raise HTTPException(status_code=503, detail={"catalog": catalog})
    return {
        "status": "ready",
        "catalog": catalog,
        "llm_backend": services.settings.llm_backend,
        "auth_enabled": bool(services.settings.api_key),
        "orchestration": "langgraph",
        "mcp_endpoint": "/mcp/",
        "mcp_tools": [
            "search_knowledge_base",
            "query_knowledge_base",
            "list_knowledge_documents",
            "knowledge_base_health",
        ],
        "retrieval_recall_k": services.settings.recall_k,
        "retrieval_final_k": services.settings.final_k,
        "reranker_enabled": services.settings.enable_reranker,
    }


@app.get("/v1/documents", dependencies=[Depends(require_api_key)])
def list_documents() -> dict:
    records = get_services().kb.document_records()
    return {"documents": [record.__dict__ for record in records]}


@app.post("/v1/documents", dependencies=[Depends(require_api_key)])
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    services = get_services()
    safe_name = Path(file.filename or "document.pdf").name
    if Path(safe_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="仅支持 PDF 文件")
    incoming_dir = services.settings.upload_dir / ".incoming"
    object_dir = services.settings.upload_dir / "objects"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)
    temporary = incoming_dir / f"{uuid4().hex}.upload"
    size = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > services.settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="文件超过上传大小限制")
                digest.update(chunk)
                output.write(chunk)
        with temporary.open("rb") as uploaded:
            if uploaded.read(5) != b"%PDF-":
                raise HTTPException(status_code=415, detail="文件内容不是有效 PDF")
        file_hash = digest.hexdigest()
        target = object_dir / f"{file_hash}.pdf"
        if target.exists():
            temporary.unlink()
        else:
            temporary.replace(target)
        chunks = await asyncio.to_thread(services.ingest, target, safe_name)
    except Exception as exc:
        metrics.record_error(type(exc).__name__)
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "request_id": request.state.request_id,
        "source": safe_name,
        "bytes": size,
        "chunks_added": chunks,
        "idempotent_skip": chunks == 0,
    }


@app.get("/v1/documents/versions", dependencies=[Depends(require_api_key)])
def document_versions(source: str) -> dict:
    records = get_services().kb.history_versions(source)
    return {
        "source": Path(source).name,
        "versions": [record.__dict__ for record in records],
    }


@app.post("/v1/documents/rollback", dependencies=[Depends(require_api_key)])
async def rollback_document(request: Request, payload: RollbackRequest) -> dict:
    services = get_services()
    try:
        record = await asyncio.to_thread(
            services.kb.rollback,
            payload.source,
            payload.version_id,
        )
    except RollbackUnavailableError as exc:
        metrics.record_error(type(exc).__name__)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        metrics.record_error(type(exc).__name__)
        raise
    return {
        "request_id": request.state.request_id,
        "source": Path(payload.source).name,
        "active_version_id": record.version_id,
        "chunk_count": record.chunk_count,
    }


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    try:
        result = await asyncio.to_thread(get_services().query, payload, request.state.request_id)
    except Exception as exc:
        metrics.record_error(type(exc).__name__)
        raise
    return QueryResponse(
        request_id=request.state.request_id,
        run_id=result.run_id,
        answer=result.answer,
        grounded=result.grounded,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        evidence=[
            EvidenceResponse(
                citation=hit.citation,
                chunk_id=hit.chunk_id,
                content=hit.content,
                score=hit.score,
                source=hit.source,
                page=hit.page,
            )
            for hit in result.hits
        ],
        trace=[step.__dict__ for step in result.steps],
    )


@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_api_key)])
def prometheus_metrics() -> str:
    return metrics.render_prometheus()
