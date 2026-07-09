from local_ai_agents.tools.list_symbols import ListSymbolsArgs, list_symbols, make_list_symbols_tool

SAMPLE_SOURCE = '''"""A sample module."""


def top_level_function():
    pass


class SampleClass:
    def method_one(self):
        pass

    async def method_two(self):
        pass


async def top_level_async_function():
    pass
'''


def make_sample(tmp_path):
    (tmp_path / "sample.py").write_text(SAMPLE_SOURCE)
    return tmp_path


class TestListSymbols:
    def test_finds_top_level_functions(self, tmp_path):
        make_sample(tmp_path)
        symbols = list_symbols(tmp_path, "sample.py")
        names = {s.name for s in symbols}
        assert "top_level_function" in names

    def test_finds_classes(self, tmp_path):
        make_sample(tmp_path)
        symbols = list_symbols(tmp_path, "sample.py")
        class_symbols = [s for s in symbols if s.kind == "class"]
        assert class_symbols[0].name == "SampleClass"

    def test_finds_methods_inside_classes(self, tmp_path):
        make_sample(tmp_path)
        symbols = list_symbols(tmp_path, "sample.py")
        names = {s.name for s in symbols}
        assert "method_one" in names

    def test_distinguishes_async_functions(self, tmp_path):
        make_sample(tmp_path)
        symbols = list_symbols(tmp_path, "sample.py")
        async_names = {s.name for s in symbols if s.kind == "async_function"}
        assert "top_level_async_function" in async_names
        assert "method_two" in async_names

    def test_symbols_are_sorted_by_line_number(self, tmp_path):
        make_sample(tmp_path)
        symbols = list_symbols(tmp_path, "sample.py")
        lines = [s.line for s in symbols]
        assert lines == sorted(lines)

    def test_real_line_numbers_are_reported(self, tmp_path):
        make_sample(tmp_path)
        symbols = list_symbols(tmp_path, "sample.py")
        top_level = next(s for s in symbols if s.name == "top_level_function")
        assert top_level.line == 4


class TestMakeListSymbolsTool:
    async def test_handler_returns_serializable_dicts(self, tmp_path):
        make_sample(tmp_path)
        tool = make_list_symbols_tool(tmp_path)
        result = await tool.handler(ListSymbolsArgs(path="sample.py"))
        assert isinstance(result[0], dict)
        assert "name" in result[0]
