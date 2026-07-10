from rag_eval_metrics import bytes_to_mb, current_process_rss_bytes


class TestCurrentProcessRssBytes:
    def test_returns_a_real_positive_number(self):
        rss = current_process_rss_bytes()
        assert rss > 0

    def test_allocating_memory_increases_rss(self):
        before = current_process_rss_bytes()
        # Real, sizable allocation (~50MB) - large enough that RSS growth
        # is reliably observable rather than lost in measurement noise.
        big_list = [0] * (50 * 1024 * 1024 // 8)
        after = current_process_rss_bytes()
        assert after > before
        assert len(big_list) > 0  # keep the allocation alive until measured


class TestBytesToMb:
    def test_converts_correctly(self):
        assert bytes_to_mb(1024 * 1024) == 1.0

    def test_zero_bytes_is_zero_mb(self):
        assert bytes_to_mb(0) == 0.0
