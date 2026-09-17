from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .document_catalog import (
    DocumentCatalog,
    DocumentRecord,
    IngestionTicket,
    RollbackUnavailableError,
    VersionRecord,
)
from .models import ModelRuntime
from .types import SearchHit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class KnowledgeBase:
    def __init__(self, settings: Settings, runtime: ModelRuntime):
        self.settings = settings
        self.runtime = runtime
        self.catalog = DocumentCatalog(settings.metadata_db_path, settings.ingestion_lease_seconds)

    @property
    def db(self) -> Chroma:
        return Chroma(
            collection_name=self.settings.collection_name,
            persist_directory=str(self.settings.vector_db_dir),
            embedding_function=self.runtime.embedding,
        )

    def ingest(self, path: Path, replace: bool = False, source_name: str | None = None) -> int:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError("当前仅支持 PDF 文件")
        if replace:
            self.clear()

        source = Path(source_name or path.name).name
        file_hash = _sha256_file(path)
        ticket = self.catalog.begin_ingestion(source, str(path), file_hash)
        if ticket.already_active:
            return 0

        ids: list[str] = []
        try:
            # Clean up chunks left by a previously failed attempt of this version.
            stale = self.db.get(where={"version_id": ticket.version_id}).get("ids", [])
            if stale:
                self.db.delete(ids=stale)

            documents = PyPDFLoader(str(path)).load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            )
            chunks = splitter.split_documents(documents)
            if not chunks:
                raise ValueError("PDF 未提取到可索引文本，可能是扫描件")
            for index, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
                chunk_id = hashlib.sha256(
                    f"{ticket.version_id}:{index}:{chunk_hash}".encode("utf-8")
                ).hexdigest()
                chunk.metadata.update(
                    {
                        "document_id": ticket.document_id,
                        "version_id": ticket.version_id,
                        "chunk_id": chunk_id,
                        "source": source,
                        "source_path": str(path),
                        "file_hash": file_hash,
                        "chunk_hash": chunk_hash,
                        "chunk_index": index,
                    }
                )
                ids.append(chunk_id)
            self.db.add_documents(chunks, ids=ids)
            self.catalog.activate(ticket, len(chunks))
        except Exception as exc:
            if ids:
                try:
                    self.db.delete(ids=ids)
                except Exception:
                    pass
            self.catalog.mark_failed(ticket, f"{type(exc).__name__}: {exc}")
            raise

        # 激活后再回收历史向量，并保留最近 KEEP_HISTORY_VERSIONS 个已激活版本；
        # 这样一次失败的升级可以用 rollback() 回退，而查询只认 active version。
        self._prune_history(ticket)
        return len(chunks)

    def _prune_history(self, ticket: IngestionTicket) -> None:
        keep = max(1, self.settings.keep_history_versions)
        for record in self.catalog.activated_versions(ticket.document_id)[keep:]:
            try:
                ids = self.db.get(where={"version_id": record.version_id}).get("ids", [])
                if ids:
                    self.db.delete(ids=ids)
                self.catalog.mark_pruned(record.version_id)
            except Exception:
                # 回收失败不影响读路径：查询只过滤 active version。
                continue

    def search(self, query: str) -> list[SearchHit]:
        active_versions = self.catalog.active_version_ids()
        catalog_records = self.catalog.list_documents()
        if catalog_records and not active_versions:
            return []
        search_kwargs = {}
        if active_versions:
            search_kwargs["filter"] = (
                {"version_id": active_versions[0]}
                if len(active_versions) == 1
                else {"version_id": {"$in": active_versions}}
            )
        results = self.db.similarity_search_with_relevance_scores(
            query,
            k=self.settings.recall_k,
            **search_kwargs,
        )
        if not results:
            return []
        docs = [doc for doc, _ in results]
        vector_scores = [float(score) for _, score in results]

        if self.runtime.reranker is not None:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = [float(score) for score in self.runtime.reranker.predict(pairs)]
        else:
            scores = vector_scores

        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        hits: list[SearchHit] = []
        for doc, score in ranked[: self.settings.final_k]:
            metadata = dict(doc.metadata)
            hits.append(
                SearchHit(
                    content=doc.page_content,
                    source=Path(metadata.get("source", "未知文档")).name,
                    page=int(metadata.get("page", 0)) + 1,
                    score=score,
                    metadata=metadata,
                )
            )
        return hits

    def list_documents(self) -> list[str]:
        catalog_records = self.catalog.list_documents()
        if catalog_records:
            return [item.source for item in catalog_records if item.status == "active"]
        # Backward compatibility for an index created before the catalog existed.
        payload = self.db.get(include=["metadatas"])
        return sorted({item.get("source", "未知文档") for item in payload.get("metadatas", []) if item})

    def document_records(self) -> list[DocumentRecord]:
        return self.catalog.list_documents()

    def history_versions(self, source: str) -> list[VersionRecord]:
        return self.catalog.versions_for_source(source)

    def rollback(self, source: str, version_id: str | None = None) -> VersionRecord:
        """Restore a previously activated version after a failed upgrade."""
        return self.catalog.rollback(source, version_id)

    def health(self) -> dict[str, int | str]:
        return self.catalog.health()

    def clear(self) -> None:
        ids = self.db.get().get("ids", [])
        if ids:
            self.db.delete(ids=ids)
        self.catalog.clear()
