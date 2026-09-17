from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class IngestionConflictError(RuntimeError):
    """Raised when the same document version is already being built."""


class RollbackUnavailableError(RuntimeError):
    """Raised when a document has no healthy previous version to restore."""


@dataclass(frozen=True)
class IngestionTicket:
    document_id: str
    version_id: str
    previous_version_id: str | None
    already_active: bool = False


@dataclass(frozen=True)
class VersionRecord:
    version_id: str
    document_id: str
    file_hash: str
    status: str
    chunk_count: int
    created_at: str
    activated_at: str | None


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source: str
    source_path: str
    active_version_id: str | None
    file_hash: str | None
    status: str
    chunk_count: int
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: str, length: int) -> str:
    value = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:length]


class DocumentCatalog:
    """Transactional metadata catalog for versioned, idempotent ingestion.

    Chroma stores vectors; this SQLite catalog is the source of truth for which
    document version is active. A version becomes searchable only after every
    chunk has been written successfully.
    """

    def __init__(self, path: Path, lease_seconds: int = 900):
        self.path = path
        self.lease_seconds = lease_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL,
                    active_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('building', 'active', 'inactive', 'failed')),
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    UNIQUE(document_id, file_hash),
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_document_versions_status
                    ON document_versions(status);
                """
            )

    def begin_ingestion(self, source: str, source_path: str, file_hash: str) -> IngestionTicket:
        normalized_source = Path(source).name
        document_id = _stable_id(normalized_source.casefold(), length=24)
        version_id = _stable_id(document_id, file_hash, length=32)
        now_datetime = datetime.now(timezone.utc)
        now = now_datetime.isoformat()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT active_version_id FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            previous_version_id = row["active_version_id"] if row else None
            connection.execute(
                """
                INSERT INTO documents(document_id, source, source_path, active_version_id, created_at, updated_at)
                VALUES (?, ?, ?, NULL, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    updated_at = excluded.updated_at
                """,
                (document_id, normalized_source, source_path, now, now),
            )
            version = connection.execute(
                "SELECT version_id, status, created_at FROM document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if version and version["status"] == "active" and previous_version_id == version_id:
                return IngestionTicket(document_id, version_id, previous_version_id, already_active=True)
            building = connection.execute(
                """
                SELECT version_id, created_at FROM document_versions
                WHERE document_id = ? AND status = 'building'
                ORDER BY created_at DESC LIMIT 1
                """,
                (document_id,),
            ).fetchone()
            if building:
                created_at = datetime.fromisoformat(building["created_at"])
                age_seconds = (now_datetime - created_at).total_seconds()
                if age_seconds < self.lease_seconds:
                    raise IngestionConflictError(f"文档版本正在构建：{normalized_source}")
                connection.execute(
                    """
                    UPDATE document_versions SET status = 'failed', error = 'stale ingestion lease expired'
                    WHERE version_id = ? AND status = 'building'
                    """,
                    (building["version_id"],),
                )

            connection.execute(
                """
                INSERT INTO document_versions(
                    version_id, document_id, file_hash, status, chunk_count, error, created_at, activated_at
                ) VALUES (?, ?, ?, 'building', 0, NULL, ?, NULL)
                ON CONFLICT(version_id) DO UPDATE SET
                    status = 'building', chunk_count = 0, error = NULL, created_at = excluded.created_at,
                    activated_at = NULL
                """,
                (version_id, document_id, file_hash, now),
            )
            return IngestionTicket(document_id, version_id, previous_version_id)

    def activate(self, ticket: IngestionTicket, chunk_count: int) -> None:
        if chunk_count <= 0:
            raise ValueError("空文档版本不能激活")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'inactive'
                WHERE document_id = ? AND status = 'active' AND version_id <> ?
                """,
                (ticket.document_id, ticket.version_id),
            )
            cursor = connection.execute(
                """
                UPDATE document_versions
                SET status = 'active', chunk_count = ?, error = NULL, activated_at = ?
                WHERE version_id = ? AND status = 'building'
                """,
                (chunk_count, now, ticket.version_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"无法激活未处于 building 状态的版本：{ticket.version_id}")
            connection.execute(
                "UPDATE documents SET active_version_id = ?, updated_at = ? WHERE document_id = ?",
                (ticket.version_id, now, ticket.document_id),
            )

    def mark_failed(self, ticket: IngestionTicket, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'failed', error = ?
                WHERE version_id = ? AND status = 'building'
                """,
                (error[:1000], ticket.version_id),
            )

    def active_version_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT version_id FROM document_versions WHERE status = 'active' ORDER BY version_id"
            ).fetchall()
        return [row["version_id"] for row in rows]

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.source, d.source_path, d.active_version_id,
                       v.file_hash, COALESCE(v.status, 'empty') AS status,
                       COALESCE(v.chunk_count, 0) AS chunk_count, d.updated_at
                FROM documents d
                LEFT JOIN document_versions v ON v.version_id = d.active_version_id
                ORDER BY d.source
                """
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def health(self) -> dict[str, int | str]:
        with self._connect() as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM document_versions GROUP BY status"
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        return {
            "integrity": str(integrity),
            "active_versions": counts.get("active", 0),
            "inactive_versions": counts.get("inactive", 0),
            "building_versions": counts.get("building", 0),
            "failed_versions": counts.get("failed", 0),
        }

    def version(self, version_id: str) -> VersionRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT version_id, document_id, file_hash, status, chunk_count,
                       created_at, activated_at
                FROM document_versions WHERE version_id = ?
                """,
                (version_id,),
            ).fetchone()
        if row is None:
            raise RollbackUnavailableError(f"版本不存在：{version_id}")
        return VersionRecord(**dict(row))

    def activated_versions(self, document_id: str) -> list[VersionRecord]:
        """Versions that finished ingestion, newest activation first."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT version_id, document_id, file_hash, status, chunk_count,
                       created_at, activated_at
                FROM document_versions
                WHERE document_id = ? AND status IN ('active', 'inactive')
                ORDER BY COALESCE(activated_at, created_at) DESC, created_at DESC
                """,
                (document_id,),
            ).fetchall()
        return [VersionRecord(**dict(row)) for row in rows]

    def versions_for_source(self, source: str) -> list[VersionRecord]:
        return self.activated_versions(self._document_id(source))

    def mark_pruned(self, version_id: str) -> None:
        """Record that a version's vectors were reclaimed and it is no longer restorable."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_versions
                SET chunk_count = 0, error = COALESCE(error, 'pruned: vectors reclaimed')
                WHERE version_id = ? AND status <> 'active'
                """,
                (version_id,),
            )

    def _document_id(self, source: str) -> str:
        return _stable_id(Path(source).name.casefold(), length=24)

    def rollback(self, source: str, target_version_id: str | None = None) -> VersionRecord:
        """Switch the active version back to a previously activated one.

        The old version stays queryable because its vectors are retained inside
        KEEP_HISTORY_VERSIONS; this only flips catalog state inside one
        transaction, so a failed upgrade never leaves the knowledge base empty.
        """
        normalized_source = Path(source).name
        document_id = _stable_id(normalized_source.casefold(), length=24)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT active_version_id FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if current is None:
                raise RollbackUnavailableError(f"文档不在目录中：{normalized_source}")
            current_version_id = current["active_version_id"]

            if target_version_id is None:
                candidate = connection.execute(
                    """
                    SELECT version_id FROM document_versions
                    WHERE document_id = ? AND status = 'inactive' AND chunk_count > 0
                    ORDER BY COALESCE(activated_at, created_at) DESC, created_at DESC
                    LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
                if candidate is None:
                    raise RollbackUnavailableError(f"没有可回退的历史版本：{normalized_source}")
                target_version_id = candidate["version_id"]

            if target_version_id == current_version_id:
                raise RollbackUnavailableError("目标版本已是当前生效版本")

            target = connection.execute(
                """
                SELECT status, chunk_count FROM document_versions
                WHERE version_id = ? AND document_id = ?
                """,
                (target_version_id, document_id),
            ).fetchone()
            if target is None:
                raise RollbackUnavailableError(f"目标版本不属于该文档：{target_version_id}")
            if target["status"] == "building":
                raise RollbackUnavailableError("目标版本仍在构建中，拒绝回退")
            if int(target["chunk_count"]) <= 0:
                raise RollbackUnavailableError("目标版本没有可检索的文本块，拒绝回退")

            now = _utc_now()
            if current_version_id:
                connection.execute(
                    """
                    UPDATE document_versions SET status = 'inactive'
                    WHERE version_id = ? AND status = 'active'
                    """,
                    (current_version_id,),
                )
            updated = connection.execute(
                """
                UPDATE document_versions
                SET status = 'active', activated_at = COALESCE(activated_at, ?)
                WHERE version_id = ? AND document_id = ?
                """,
                (now, target_version_id, document_id),
            )
            if updated.rowcount != 1:
                raise RollbackUnavailableError(f"无法回退到版本：{target_version_id}")
            connection.execute(
                "UPDATE documents SET active_version_id = ?, updated_at = ? WHERE document_id = ?",
                (target_version_id, now, document_id),
            )
        return self.version(target_version_id)

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")
