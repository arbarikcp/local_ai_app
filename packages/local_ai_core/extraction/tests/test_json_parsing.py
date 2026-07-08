from local_ai_core.extraction.json_parsing import strip_markdown_fence, try_parse_json


class TestStripMarkdownFence:
    def test_removes_json_fence(self):
        assert strip_markdown_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_removes_bare_fence(self):
        assert strip_markdown_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_noop_on_unfenced_text(self):
        assert strip_markdown_fence('{"a": 1}') == '{"a": 1}'


class TestTryParseJson:
    def test_parses_fenced_json(self):
        assert try_parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parses_unfenced_json(self):
        assert try_parse_json('{"a": 1}') == {"a": 1}

    def test_returns_none_on_garbage(self):
        assert try_parse_json("not json") is None

    def test_returns_none_on_empty_string(self):
        assert try_parse_json("") is None
