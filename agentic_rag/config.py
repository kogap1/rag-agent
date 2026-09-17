from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str, default: bool) -> bool:
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _as_csv(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    upload_dir: Path
    vector_db_dir: Path
    model_cache_dir: Path
    metadata_db_path: Path
    embedding_model: str
    reranker_model: str
    llm_model: str
    llm_backend: str
    llm_api_base: str
    llm_api_key: str
    llm_api_timeout_seconds: float
    device: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    recall_k: int
    final_k: int
    max_agent_steps: int
    max_context_chars: int
    max_history_messages: int
    tool_timeout_seconds: float
    tool_max_retries: int
    max_tool_output_chars: int
    enable_reranker: bool
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    api_key: str
    api_max_concurrency: int
    max_upload_bytes: int
    ingestion_lease_seconds: int
    mcp_allowed_hosts: tuple[str, ...]
    mcp_allowed_origins: tuple[str, ...]
    max_retrieval_rounds: int
    keep_history_versions: int

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        load_dotenv(root / ".env")
        settings = cls(
            project_root=root,
            upload_dir=root / os.getenv("UPLOAD_DIR", "uploaded_files"),
            vector_db_dir=root / os.getenv("VECTOR_DB_DIR", "server_vector_db"),
            model_cache_dir=root / os.getenv("MODEL_CACHE_DIR", "models"),
            metadata_db_path=root / os.getenv("METADATA_DB_PATH", "state/documents.sqlite3"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Xorbits/bge-m3"),
            reranker_model=os.getenv("RERANKER_MODEL", "Xorbits/bge-reranker-base"),
            llm_model=os.getenv("LLM_MODEL", "qwen/Qwen2.5-1.5B-Instruct"),
            llm_backend=os.getenv("LLM_BACKEND", "local").strip().lower(),
            llm_api_base=os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            llm_api_timeout_seconds=float(os.getenv("LLM_API_TIMEOUT_SECONDS", "120")),
            device=os.getenv("DEVICE", "auto"),
            collection_name=os.getenv("COLLECTION_NAME", "agentic_rag"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            recall_k=int(os.getenv("RECALL_K", "20")),
            final_k=int(os.getenv("FINAL_K", "6")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "2")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "6000")),
            max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "6")),
            tool_timeout_seconds=float(os.getenv("TOOL_TIMEOUT_SECONDS", "30")),
            tool_max_retries=int(os.getenv("TOOL_MAX_RETRIES", "1")),
            max_tool_output_chars=int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "12000")),
            enable_reranker=_as_bool(os.getenv("ENABLE_RERANKER", ""), True),
            cost_per_1m_input_tokens=float(os.getenv("COST_PER_1M_INPUT_TOKENS", "0")),
            cost_per_1m_output_tokens=float(os.getenv("COST_PER_1M_OUTPUT_TOKENS", "0")),
            api_key=os.getenv("API_KEY", ""),
            api_max_concurrency=int(os.getenv("API_MAX_CONCURRENCY", "1")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
            ingestion_lease_seconds=int(os.getenv("INGESTION_LEASE_SECONDS", "900")),
            mcp_allowed_hosts=_as_csv(
                os.getenv("MCP_ALLOWED_HOSTS", ""),
                ("127.0.0.1:*", "localhost:*", "[::1]:*"),
            ),
            mcp_allowed_origins=_as_csv(
                os.getenv("MCP_ALLOWED_ORIGINS", ""),
                ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"),
            ),
            max_retrieval_rounds=int(os.getenv("MAX_RETRIEVAL_ROUNDS", "2")),
            keep_history_versions=int(os.getenv("KEEP_HISTORY_VERSIONS", "2")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.llm_backend not in {"local", "openai_compatible"}:
            raise ValueError("LLM_BACKEND 只能是 local 或 openai_compatible")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_SIZE 必须大于 0，且 CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.recall_k < 1 or self.final_k < 1 or self.final_k > self.recall_k:
            raise ValueError("检索参数必须满足 1 <= FINAL_K <= RECALL_K")
        if self.max_agent_steps < 1 or self.max_context_chars < 1 or self.max_history_messages < 0:
            raise ValueError("Agent 与上下文预算配置不合法")
        if self.tool_timeout_seconds <= 0 or self.tool_max_retries < 0:
            raise ValueError("工具超时必须大于 0，重试次数不能为负数")
        if self.llm_api_timeout_seconds <= 0:
            raise ValueError("LLM_API_TIMEOUT_SECONDS 必须大于 0")
        if self.api_max_concurrency < 1 or self.max_upload_bytes < 1:
            raise ValueError("API 并发数与上传大小限制必须大于 0")
        if self.ingestion_lease_seconds < 0:
            raise ValueError("INGESTION_LEASE_SECONDS 不能为负数")
        if not self.mcp_allowed_hosts:
            raise ValueError("MCP_ALLOWED_HOSTS 不能为空")
        if self.max_retrieval_rounds < 1:
            raise ValueError("MAX_RETRIEVAL_ROUNDS 必须大于 0")
        if self.keep_history_versions < 1:
            raise ValueError("KEEP_HISTORY_VERSIONS 必须大于 0")

    @property
    def run_log_path(self) -> Path:
        return self.project_root / "runs" / "agent_runs.jsonl"

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
