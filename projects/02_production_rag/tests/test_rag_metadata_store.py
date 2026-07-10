import pytest

from rag_metadata_store import DocumentRecord, QueryLogRecord, RagMetadataStore


def make_doc(doc_id: str = "doc-1", status: str = "ingested") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        source_path="/docs/doc-1.md",
        title="Doc One",
        status=status,
        content_hash="abc123",
        chunk_count=3,
    )


class TestSaveAndGetDocument:
    def test_a_saved_document_can_be_retrieved(self):
        with RagMetadataStore(":memory:") as store:
            store.save_document(make_doc())
            record = store.get_document("doc-1")
            assert record is not None
            assert record.title == "Doc One"
            assert record.chunk_count == 3
            assert record.ingested_at is not None

    def test_missing_doc_id_returns_none(self):
        with RagMetadataStore(":memory:") as store:
            assert store.get_document("does-not-exist") is None

    def test_saving_the_same_doc_id_again_updates_in_place(self):
        with RagMetadataStore(":memory:") as store:
            store.save_document(make_doc(status="ingested"))
            store.save_document(DocumentRecord(
                doc_id="doc-1", source_path="/docs/doc-1.md", title="Doc One Updated",
                status="ingested", content_hash="def456", chunk_count=5,
            ))
            record = store.get_document("doc-1")
            assert record.title == "Doc One Updated"
            assert record.content_hash == "def456"
            assert record.chunk_count == 5

    def test_invalid_status_raises(self):
        with RagMetadataStore(":memory:") as store:
            with pytest.raises(ValueError):
                store.save_document(make_doc(status="not-a-real-status"))


class TestDeleteDocument:
    def test_deletes_an_existing_document_and_returns_true(self):
        with RagMetadataStore(":memory:") as store:
            store.save_document(make_doc())
            assert store.delete_document("doc-1") is True
            assert store.get_document("doc-1") is None

    def test_deleting_a_missing_document_returns_false(self):
        with RagMetadataStore(":memory:") as store:
            assert store.delete_document("does-not-exist") is False


class TestListDocuments:
    def test_lists_every_document(self):
        with RagMetadataStore(":memory:") as store:
            store.save_document(make_doc("doc-1"))
            store.save_document(make_doc("doc-2"))
            assert len(store.list_documents()) == 2

    def test_filters_by_status(self):
        with RagMetadataStore(":memory:") as store:
            store.save_document(make_doc("doc-1", status="ingested"))
            store.save_document(make_doc("doc-2", status="quarantined"))
            quarantined = store.list_documents(status="quarantined")
            assert len(quarantined) == 1
            assert quarantined[0].doc_id == "doc-2"


class TestQueryLog:
    def test_a_logged_query_is_recorded(self):
        with RagMetadataStore(":memory:") as store:
            store.log_query(
                QueryLogRecord(
                    query_id="q-1",
                    question="How do I reset my password?",
                    answer_text="Click forgot password.",
                    citation_count=1,
                    verified_citation_count=1,
                    latency_ms=12.5,
                )
            )
            rows = store._conn.execute("SELECT question FROM query_log").fetchall()
            assert rows == [("How do I reset my password?",)]


class TestPersistsAcrossCloseAndReopen:
    def test_a_document_survives_a_real_close_and_reopen_cycle(self, tmp_path):
        db_path = tmp_path / "rag_metadata.db"

        store = RagMetadataStore(db_path)
        store.save_document(make_doc())
        store.close()

        reopened = RagMetadataStore(db_path)
        record = reopened.get_document("doc-1")
        reopened.close()

        assert record is not None
        assert record.title == "Doc One"
