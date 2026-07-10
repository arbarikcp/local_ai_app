from extraction_storage import ExtractionRecord, ExtractionStore


def make_record(request_id: str = "req-1", *, needs_review: bool = False, confidence: str = "high") -> ExtractionRecord:
    return ExtractionRecord(
        request_id=request_id,
        trace_id="trace-1",
        schema_name="invoice_v1",
        raw_input="Invoice #A-1 for $10.",
        extracted_fields={"invoice_number": "A-1", "total_amount": 10.0},
        confidence=confidence,
        needs_review=needs_review,
        validation_error=None,
        used_repair_retry=False,
        latency_ms=42.0,
    )


class TestSaveAndGet:
    def test_a_saved_record_can_be_retrieved(self):
        with ExtractionStore(":memory:") as store:
            store.save(make_record())
            record = store.get("req-1")
            assert record is not None
            assert record.schema_name == "invoice_v1"
            assert record.extracted_fields == {"invoice_number": "A-1", "total_amount": 10.0}
            assert record.created_at is not None

    def test_missing_request_id_returns_none(self):
        with ExtractionStore(":memory:") as store:
            assert store.get("does-not-exist") is None


class TestListLowConfidence:
    def test_only_records_needing_review_are_returned(self):
        with ExtractionStore(":memory:") as store:
            store.save(make_record("req-1", needs_review=False, confidence="high"))
            store.save(make_record("req-2", needs_review=True, confidence="low"))
            low_confidence = store.list_low_confidence()
            assert [r.request_id for r in low_confidence] == ["req-2"]

    def test_keyed_on_needs_review_flag_not_confidence_string(self):
        # A record with confidence != "low" but needs_review=True (e.g. a
        # validation failure) must still show up - the query is not a raw
        # `confidence = 'low'` match.
        with ExtractionStore(":memory:") as store:
            store.save(make_record("req-1", needs_review=True, confidence="medium"))
            low_confidence = store.list_low_confidence()
            assert len(low_confidence) == 1
            assert low_confidence[0].confidence == "medium"

    def test_respects_the_limit(self):
        with ExtractionStore(":memory:") as store:
            for i in range(5):
                store.save(make_record(f"req-{i}", needs_review=True))
            assert len(store.list_low_confidence(limit=2)) == 2

    def test_empty_store_returns_empty_list(self):
        with ExtractionStore(":memory:") as store:
            assert store.list_low_confidence() == []


class TestListAll:
    def test_returns_every_record(self):
        with ExtractionStore(":memory:") as store:
            store.save(make_record("req-1"))
            store.save(make_record("req-2"))
            assert len(store.list_all()) == 2


class TestPersistsAcrossCloseAndReopen:
    def test_a_record_survives_a_real_close_and_reopen_cycle(self, tmp_path):
        db_path = tmp_path / "extraction.db"

        store = ExtractionStore(db_path)
        store.save(make_record())
        store.close()

        reopened = ExtractionStore(db_path)
        record = reopened.get("req-1")
        reopened.close()

        assert record is not None
        assert record.extracted_fields == {"invoice_number": "A-1", "total_amount": 10.0}
