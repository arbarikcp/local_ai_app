import pytest

from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_agents.tools.patch_tools import (
    PatchFormatError,
    apply_patch,
    propose_patch,
    validate_patch_format,
)

SAMPLE_FILE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def average(numbers):\n"
    "    return sum(numbers) / len(numbers)\n"
)

VALID_PATCH = (
    "--- calculator.py\n"
    "+++ calculator.py\n"
    "@@ -5,2 +5,4 @@\n"
    " def average(numbers):\n"
    "-    return sum(numbers) / len(numbers)\n"
    "+    if not numbers:\n"
    "+        return 0.0\n"
    "+    return sum(numbers) / len(numbers)\n"
)

# Same line number, but claims different (hallucinated) old content than the
# file actually has - a real stand-in for "the model described code that
# doesn't exist."
HALLUCINATED_PATCH = (
    "--- calculator.py\n"
    "+++ calculator.py\n"
    "@@ -5,2 +5,3 @@\n"
    " def average(numbers):\n"
    "-    return statistics.mean(numbers)\n"
    "+    if not numbers:\n"
    "+        return 0.0\n"
)


def make_sample_repo(tmp_path):
    (tmp_path / "calculator.py").write_text(SAMPLE_FILE)
    return tmp_path


class TestProposePatch:
    async def test_returns_the_runtimes_response_text(self):
        runtime = FakeRuntime(default_response=VALID_PATCH)
        patch_text = await propose_patch("fix average", {"calculator.py": SAMPLE_FILE}, runtime, model="fake-model")
        assert patch_text.strip() == VALID_PATCH.strip()

    async def test_includes_the_instruction_and_file_contents_in_the_prompt(self):
        runtime = FakeRuntime(default_response=VALID_PATCH)
        await propose_patch("fix average", {"calculator.py": SAMPLE_FILE}, runtime, model="fake-model")
        sent_prompt = runtime.requests_received[0].prompt
        assert "fix average" in sent_prompt
        assert "def average" in sent_prompt


class TestValidatePatchFormat:
    def test_parses_a_valid_patch(self):
        parsed = validate_patch_format(VALID_PATCH)
        assert parsed.file_path == "calculator.py"
        assert len(parsed.hunks) == 1
        assert parsed.hunks[0].old_start == 5

    def test_rejects_a_patch_missing_the_file_header(self):
        with pytest.raises(PatchFormatError):
            validate_patch_format("@@ -1,1 +1,1 @@\n-old\n+new\n")

    def test_rejects_a_patch_with_no_hunks(self):
        with pytest.raises(PatchFormatError):
            validate_patch_format("--- a.py\n+++ a.py\n")

    def test_rejects_a_malformed_hunk_line(self):
        malformed = "--- a.py\n+++ a.py\n@@ -1,1 +1,1 @@\n???not a valid diff line\n"
        with pytest.raises(PatchFormatError):
            validate_patch_format(malformed)


class TestApplyPatch:
    def test_applies_a_valid_patch_and_returns_the_relative_path(self, tmp_path):
        make_sample_repo(tmp_path)
        result_path = apply_patch(tmp_path, VALID_PATCH)
        assert result_path == "calculator.py"

    def test_the_file_content_actually_changes(self, tmp_path):
        make_sample_repo(tmp_path)
        apply_patch(tmp_path, VALID_PATCH)
        new_content = (tmp_path / "calculator.py").read_text()
        assert "if not numbers:" in new_content
        assert "return 0.0" in new_content

    def test_the_patched_function_is_actually_correct_python(self, tmp_path):
        make_sample_repo(tmp_path)
        apply_patch(tmp_path, VALID_PATCH)
        namespace: dict = {}
        exec((tmp_path / "calculator.py").read_text(), namespace)  # noqa: S102 - test-only, sandboxed tmp_path content
        assert namespace["average"]([]) == 0.0
        assert namespace["average"]([1, 2, 3]) == 2

    def test_a_hallucinated_patch_is_rejected_and_the_file_is_untouched(self, tmp_path):
        make_sample_repo(tmp_path)
        original_content = (tmp_path / "calculator.py").read_text()
        with pytest.raises(PatchFormatError):
            apply_patch(tmp_path, HALLUCINATED_PATCH)
        assert (tmp_path / "calculator.py").read_text() == original_content
