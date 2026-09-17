from pathlib import Path

import pytest

from agentic_rag.document_catalog import (
    DocumentCatalog,
    IngestionConflictError,
    RollbackUnavailableError,
)


def test_catalog_activates_new_version_and_invalidates_previous(tmp_path: Path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    first = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    catalog.activate(first, 3)

    duplicate = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    assert duplicate.already_active is True

    second = catalog.begin_ingestion("规则.pdf", "objects/b.pdf", "b" * 64)
    assert second.previous_version_id == first.version_id
    catalog.activate(second, 5)

    assert catalog.active_version_ids() == [second.version_id]
    record = catalog.list_documents()[0]
    assert record.source == "规则.pdf"
    assert record.file_hash == "b" * 64
    assert record.chunk_count == 5
    assert record.status == "active"


def test_catalog_rejects_concurrent_build_and_tracks_failure(tmp_path: Path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    ticket = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    with pytest.raises(IngestionConflictError):
        catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)

    catalog.mark_failed(ticket, "parser failed")
    retry = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    catalog.activate(retry, 2)
    assert catalog.health()["active_versions"] == 1


def test_catalog_can_recover_an_expired_ingestion_lease(tmp_path: Path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3", lease_seconds=0)
    first = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    retry = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    assert retry.version_id == first.version_id
    catalog.activate(retry, 1)
    assert catalog.health()["active_versions"] == 1
# RESUME_ALIGN_ACCEPTANCE


def test_catalog_keeps_history_and_rolls_back_to_previous_version(tmp_path: Path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    first = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    catalog.activate(first, 3)
    second = catalog.begin_ingestion("规则.pdf", "objects/b.pdf", "b" * 64)
    catalog.activate(second, 5)

    versions = catalog.versions_for_source("规则.pdf")
    assert [item.version_id for item in versions] == [second.version_id, first.version_id]
    assert versions[1].status == "inactive"

    restored = catalog.rollback("规则.pdf")

    assert restored.version_id == first.version_id
    assert restored.status == "active"
    assert catalog.active_version_ids() == [first.version_id]
    assert catalog.version(second.version_id).status == "inactive"


def test_catalog_refuses_rollback_without_restorable_history(tmp_path: Path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    ticket = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    catalog.activate(ticket, 2)

    with pytest.raises(RollbackUnavailableError):
        catalog.rollback("规则.pdf")

    with pytest.raises(RollbackUnavailableError):
        catalog.rollback("不存在的.pdf")


def test_catalog_refuses_pruned_or_unknown_target_version(tmp_path: Path):
    catalog = DocumentCatalog(tmp_path / "catalog.sqlite3")
    first = catalog.begin_ingestion("规则.pdf", "objects/a.pdf", "a" * 64)
    catalog.activate(first, 3)
    second = catalog.begin_ingestion("规则.pdf", "objects/b.pdf", "b" * 64)
    catalog.activate(second, 5)
    catalog.mark_pruned(first.version_id)

    with pytest.raises(RollbackUnavailableError):
        catalog.rollback("规则.pdf")

    with pytest.raises(RollbackUnavailableError):
        catalog.rollback("规则.pdf", first.version_id)

    with pytest.raises(RollbackUnavailableError):
        catalog.rollback("规则.pdf", "f" * 32)
