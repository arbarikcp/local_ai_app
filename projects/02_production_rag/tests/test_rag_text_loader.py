from rag_text_loader import load_text_directory, load_text_file, load_text_string


class TestLoadTextFile:
    def test_loads_a_real_txt_file(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("Some plain text notes.")
        document = load_text_file(path)
        assert document.doc_id == "notes"
        assert document.text == "Some plain text notes."
        assert document.source_path == str(path)
        assert document.title == "notes"


class TestLoadTextString:
    def test_builds_a_document_with_no_file_on_disk(self):
        document = load_text_string("inline-1", "Inline submitted text.")
        assert document.doc_id == "inline-1"
        assert document.text == "Inline submitted text."
        assert document.source_path == ""

    def test_accepts_an_explicit_title(self):
        document = load_text_string("inline-1", "text", title="My Title")
        assert document.title == "My Title"

    def test_defaults_title_to_doc_id(self):
        document = load_text_string("inline-1", "text")
        assert document.title == "inline-1"


class TestLoadTextDirectory:
    def test_loads_every_txt_file_sorted(self, tmp_path):
        (tmp_path / "b.txt").write_text("second")
        (tmp_path / "a.txt").write_text("first")
        (tmp_path / "ignored.md").write_text("not a txt file")

        documents = load_text_directory(tmp_path)

        assert [d.doc_id for d in documents] == ["a", "b"]

    def test_empty_directory_returns_no_documents(self, tmp_path):
        assert load_text_directory(tmp_path) == []
