import pytest

from local_ai_core.extraction.review_queue import ReviewQueue


class TestEnqueue:
    def test_enqueue_returns_an_item_with_a_generated_id(self):
        queue = ReviewQueue()
        item = queue.enqueue(
            extracted_fields={"name": "Maria", "age": None, "city": "Austin"},
            confidence="low",
            reason="missing required field: age",
            source_text="Maria lives in Austin.",
        )
        assert item.item_id
        assert item.confidence == "low"
        assert item.reason == "missing required field: age"

    def test_enqueue_increases_pending_count(self):
        queue = ReviewQueue()
        assert len(queue) == 0
        queue.enqueue({"a": 1}, "low", "reason", "text")
        assert len(queue) == 1

    def test_each_enqueued_item_gets_a_distinct_id(self):
        queue = ReviewQueue()
        item1 = queue.enqueue({"a": 1}, "low", "r", "t")
        item2 = queue.enqueue({"a": 2}, "low", "r", "t")
        assert item1.item_id != item2.item_id


class TestListPending:
    def test_empty_queue_returns_empty_list(self):
        assert ReviewQueue().list_pending() == []

    def test_lists_items_in_fifo_order(self):
        queue = ReviewQueue()
        first = queue.enqueue({"a": 1}, "low", "r1", "t1")
        second = queue.enqueue({"a": 2}, "low", "r2", "t2")
        pending = queue.list_pending()
        assert [item.item_id for item in pending] == [first.item_id, second.item_id]

    def test_resolved_items_are_not_listed_as_pending(self):
        queue = ReviewQueue()
        item = queue.enqueue({"a": 1}, "low", "r", "t")
        queue.resolve(item.item_id, approved=True)
        assert queue.list_pending() == []


class TestResolve:
    def test_resolve_removes_the_item_from_pending(self):
        queue = ReviewQueue()
        item = queue.enqueue({"a": 1}, "low", "r", "t")
        queue.resolve(item.item_id, approved=True)
        assert len(queue) == 0

    def test_resolve_records_approval_and_corrections(self):
        queue = ReviewQueue()
        item = queue.enqueue({"name": None}, "low", "missing name", "text")
        resolution = queue.resolve(item.item_id, approved=False, corrected_fields={"name": "Maria"})
        assert resolution.approved is False
        assert resolution.corrected_fields == {"name": "Maria"}

    def test_resolve_unknown_item_id_raises_key_error(self):
        queue = ReviewQueue()
        with pytest.raises(KeyError):
            queue.resolve("not-a-real-id", approved=True)

    def test_resolving_twice_raises_on_the_second_call(self):
        queue = ReviewQueue()
        item = queue.enqueue({"a": 1}, "low", "r", "t")
        queue.resolve(item.item_id, approved=True)
        with pytest.raises(KeyError):
            queue.resolve(item.item_id, approved=True)


class TestGetResolution:
    def test_returns_none_for_unresolved_item(self):
        queue = ReviewQueue()
        item = queue.enqueue({"a": 1}, "low", "r", "t")
        assert queue.get_resolution(item.item_id) is None

    def test_returns_the_resolution_after_resolving(self):
        queue = ReviewQueue()
        item = queue.enqueue({"a": 1}, "low", "r", "t")
        queue.resolve(item.item_id, approved=True)
        resolution = queue.get_resolution(item.item_id)
        assert resolution is not None
        assert resolution.approved is True
