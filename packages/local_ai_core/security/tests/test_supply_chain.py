import hashlib

import pytest

from local_ai_core.security.supply_chain import (
    ChecksumMismatchError,
    ModelManifestEntry,
    compute_sha256,
    verify_against_manifest,
)


class TestComputeSha256:
    def test_matches_hashlib_computed_directly(self, tmp_path):
        content = b"model weights go here"
        path = tmp_path / "model.bin"
        path.write_bytes(content)

        assert compute_sha256(path) == hashlib.sha256(content).hexdigest()

    def test_different_content_produces_different_hashes(self, tmp_path):
        path_a = tmp_path / "a.bin"
        path_b = tmp_path / "b.bin"
        path_a.write_bytes(b"content a")
        path_b.write_bytes(b"content b")

        assert compute_sha256(path_a) != compute_sha256(path_b)


class TestVerifyAgainstManifest:
    def test_matching_checksum_does_not_raise(self, tmp_path):
        content = b"trusted model file"
        path = tmp_path / "model.bin"
        path.write_bytes(content)
        entry = ModelManifestEntry(
            name="test-model",
            source_url="https://example.com/model.bin",
            sha256=hashlib.sha256(content).hexdigest(),
            license="Apache-2.0",
        )

        verify_against_manifest(path, entry)  # should not raise

    def test_tampered_file_raises_checksum_mismatch(self, tmp_path):
        original_content = b"trusted model file"
        path = tmp_path / "model.bin"
        path.write_bytes(original_content)
        entry = ModelManifestEntry(
            name="test-model",
            source_url="https://example.com/model.bin",
            sha256=hashlib.sha256(original_content).hexdigest(),
            license="Apache-2.0",
        )

        path.write_bytes(b"tampered content")

        with pytest.raises(ChecksumMismatchError):
            verify_against_manifest(path, entry)

    def test_mismatch_error_reports_expected_and_actual(self, tmp_path):
        path = tmp_path / "model.bin"
        path.write_bytes(b"actual content")
        entry = ModelManifestEntry(
            name="test-model", source_url="https://example.com/model.bin", sha256="0" * 64, license="MIT"
        )

        with pytest.raises(ChecksumMismatchError) as exc_info:
            verify_against_manifest(path, entry)
        assert exc_info.value.expected == "0" * 64
        assert exc_info.value.name == "test-model"
