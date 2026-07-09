from local_ai_rag.loaders.markdown_loader import load_markdown_directory, load_markdown_file


def write_md(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadMarkdownFile:
    def test_uses_the_filename_stem_as_doc_id(self, tmp_path):
        path = write_md(tmp_path, "password_reset.md", "# Resetting Your Password\n\nBody text.")
        doc = load_markdown_file(path)
        assert doc.doc_id == "password_reset"

    def test_splits_the_leading_title_line_out_of_the_text(self, tmp_path):
        path = write_md(tmp_path, "doc.md", "# My Title\n\nFirst paragraph.\n\nSecond paragraph.")
        doc = load_markdown_file(path)
        assert doc.title == "My Title"
        assert "My Title" not in doc.text
        assert "First paragraph." in doc.text

    def test_handles_a_file_with_no_title_line(self, tmp_path):
        path = write_md(tmp_path, "doc.md", "Just body text, no heading.")
        doc = load_markdown_file(path)
        assert doc.title == ""
        assert doc.text == "Just body text, no heading."

    def test_source_path_is_recorded(self, tmp_path):
        path = write_md(tmp_path, "doc.md", "# T\n\nbody")
        doc = load_markdown_file(path)
        assert doc.source_path == str(path)


class TestLoadMarkdownDirectory:
    def test_loads_every_md_file_in_the_directory(self, tmp_path):
        write_md(tmp_path, "a.md", "# A\n\ntext a")
        write_md(tmp_path, "b.md", "# B\n\ntext b")
        write_md(tmp_path, "not_markdown.txt", "ignored")
        docs = load_markdown_directory(tmp_path)
        assert {d.doc_id for d in docs} == {"a", "b"}

    def test_returns_documents_sorted_by_doc_id(self, tmp_path):
        write_md(tmp_path, "zebra.md", "# Z\n\nz")
        write_md(tmp_path, "alpha.md", "# A\n\na")
        docs = load_markdown_directory(tmp_path)
        assert [d.doc_id for d in docs] == ["alpha", "zebra"]

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert load_markdown_directory(tmp_path) == []
