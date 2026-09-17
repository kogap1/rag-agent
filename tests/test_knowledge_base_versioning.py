from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import agentic_rag.knowledge_base as knowledge_base_module
from agentic_rag.document_catalog import RollbackUnavailableError
from agentic_rag.knowledge_base import KnowledgeBase


class FakeVectorStore:
    def __init__(self):
        self.rows = {}
        self.last_filter = None

    def get(self, where=None, include=None):
        rows = list(self.rows.items())
        if where:
            rows = [
                (identifier, document)
                for identifier, document in rows
                if all(document.metadata.get(key) == value for key, value in where.items())
            ]
        return {
            "ids": [identifier for identifier, _ in rows],
            "metadatas": [document.metadata for _, document in rows],
        }

    def add_documents(self, documents, ids):
        self.rows.update(dict(zip(ids, documents)))

    def delete(self, ids):
        for identifier in ids:
            self.rows.pop(identifier, None)

    def similarity_search_with_relevance_scores(self, query, k, **kwargs):
        self.last_filter = kwargs.get("filter")
        version_id = self.last_filter.get("version_id") if self.last_filter else None
        rows = [doc for doc in self.rows.values() if doc.metadata.get("version_id") == version_id]
        return [(doc, 0.9) for doc in rows[:k]]


class FakeKnowledgeBase(KnowledgeBase):
    def __init__(self, settings, runtime, vector_store):
        self._vector_store = vector_store
        super().__init__(settings, runtime)

    @property
    def db(self):
        return self._vector_store


def test_ingestion_activates_version_and_search_filters_it(tmp_path: Path, monkeypatch):
    source = tmp_path / "object.pdf"
    source.write_bytes(b"%PDF-test")

    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="这是可检索的规则文本。", metadata={"page": 0})]

    monkeypatch.setattr(knowledge_base_module, "PyPDFLoader", FakeLoader)
    settings = SimpleNamespace(
        metadata_db_path=tmp_path / "state.sqlite3",
        ingestion_lease_seconds=60,
        chunk_size=100,
        chunk_overlap=10,
        recall_k=5,
        final_k=2,
        keep_history_versions=2,
    )
    runtime = SimpleNamespace(reranker=None)
    store = FakeVectorStore()
    kb = FakeKnowledgeBase(settings, runtime, store)

    assert kb.ingest(source, source_name="规则.pdf") == 1
    assert kb.ingest(source, source_name="规则.pdf") == 0
    assert kb.list_documents() == ["规则.pdf"]

    hits = kb.search("规则")
    version_id = kb.catalog.active_version_ids()[0]
    assert store.last_filter == {"version_id": version_id}
    assert hits[0].source == "规则.pdf"
    assert hits[0].metadata["chunk_id"]
# RESUME_ALIGN_ACCEPTANCE


def _build_kb(tmp_path: Path, monkeypatch, store, keep_history_versions: int = 2) -> KnowledgeBase:
    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="这是可检索的规则文本。", metadata={"page": 0})]

    monkeypatch.setattr(knowledge_base_module, "PyPDFLoader", FakeLoader)
    settings = SimpleNamespace(
        metadata_db_path=tmp_path / "state-history.sqlite3",
        ingestion_lease_seconds=60,
        chunk_size=100,
        chunk_overlap=10,
        recall_k=5,
        final_k=2,
        keep_history_versions=keep_history_versions,
    )
    return FakeKnowledgeBase(settings, SimpleNamespace(reranker=None), store)


def _write_pdf(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(f"%PDF-{name}".encode())
    return path


def test_history_is_retained_and_rollback_restores_previous_version(tmp_path: Path, monkeypatch):
    store = FakeVectorStore()
    kb = _build_kb(tmp_path, monkeypatch, store, keep_history_versions=2)

    assert kb.ingest(_write_pdf(tmp_path, "v1.pdf"), source_name="规则.pdf") == 1
    first_version = kb.catalog.active_version_ids()[0]
    assert kb.ingest(_write_pdf(tmp_path, "v2.pdf"), source_name="规则.pdf") == 1
    second_version = kb.catalog.active_version_ids()[0]
    assert second_version != first_version

    record = kb.rollback("规则.pdf")

    assert record.version_id == first_version
    assert kb.catalog.active_version_ids() == [first_version]
    hits = kb.search("规则")
    assert hits and hits[0].metadata["version_id"] == first_version


def test_versions_beyond_history_window_are_pruned_and_not_restorable(tmp_path: Path, monkeypatch):
    store = FakeVectorStore()
    kb = _build_kb(tmp_path, monkeypatch, store, keep_history_versions=1)
    for index in (1, 2, 3):
        assert kb.ingest(_write_pdf(tmp_path, f"v{index}.pdf"), source_name="规则.pdf") == 1

    versions = kb.history_versions("规则.pdf")
    assert versions[0].status == "active"
    assert versions[-1].chunk_count == 0

    with pytest.raises(RollbackUnavailableError):
        kb.rollback("规则.pdf")


def test_failed_upgrade_cleans_partial_write_and_keeps_previous_version(tmp_path: Path, monkeypatch):
    class FailingVectorStore(FakeVectorStore):
        def __init__(self):
            super().__init__()
            self.fail_next_write = False

        def add_documents(self, documents, ids):
            super().add_documents(documents, ids)
            if self.fail_next_write:
                raise RuntimeError("vector write failed")

    store = FailingVectorStore()
    kb = _build_kb(tmp_path, monkeypatch, store, keep_history_versions=2)

    assert kb.ingest(_write_pdf(tmp_path, "v1.pdf"), source_name="规则.pdf") == 1
    active_before = kb.catalog.active_version_ids()

    store.fail_next_write = True
    with pytest.raises(RuntimeError):
        kb.ingest(_write_pdf(tmp_path, "v2.pdf"), source_name="规则.pdf")

    # 未完成的写入被清理，旧版本向量与 active 状态保持不变。
    assert kb.catalog.active_version_ids() == active_before
    assert store.rows
    assert {doc.metadata["version_id"] for doc in store.rows.values()} == set(active_before)
    assert kb.health()["failed_versions"] == 1
