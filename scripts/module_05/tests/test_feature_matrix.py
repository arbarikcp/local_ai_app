from feature_matrix import (
    KNOWN_FEATURE_MATRIX,
    FeatureSupport,
    RuntimeFeatures,
    matrix_to_markdown,
    notes_appendix,
    unverified_entries,
)


def test_known_feature_matrix_covers_all_four_runtimes():
    assert set(KNOWN_FEATURE_MATRIX.keys()) == {
        "ollama",
        "llama_cpp_native",
        "llama_cpp_python_server",
        "mlx_lm",
    }


def test_every_entry_has_all_six_features_populated():
    required_attrs = [
        "structured_output",
        "grammar",
        "token_counting_endpoint",
        "streaming",
        "cancellation",
        "usage_reporting",
    ]
    for features in KNOWN_FEATURE_MATRIX.values():
        for attr in required_attrs:
            support = getattr(features, attr)
            assert isinstance(support, FeatureSupport)
            assert support.level in {"yes", "partial", "no", "n/a", "unknown"}


def test_matrix_to_markdown_includes_every_runtime_name():
    table = matrix_to_markdown(KNOWN_FEATURE_MATRIX)
    for features in KNOWN_FEATURE_MATRIX.values():
        assert features.runtime in table


def test_matrix_to_markdown_marks_unverified_entries_as_documented():
    table = matrix_to_markdown(KNOWN_FEATURE_MATRIX)
    assert "documented" in table
    assert "measured" not in table  # nothing has been measured yet on this machine


def test_matrix_to_markdown_flips_to_measured_once_verified():
    fake_matrix = {
        "x": RuntimeFeatures(
            runtime="x-runtime",
            structured_output=FeatureSupport("yes", True, "confirmed live"),
            grammar=FeatureSupport("no", False),
            token_counting_endpoint=FeatureSupport("no", False),
            streaming=FeatureSupport("no", False),
            cancellation=FeatureSupport("no", False),
            usage_reporting=FeatureSupport("no", False),
        )
    }
    table = matrix_to_markdown(fake_matrix)
    assert "yes (measured)" in table


def test_notes_appendix_includes_the_caveat():
    appendix = notes_appendix(KNOWN_FEATURE_MATRIX)
    assert "documented from public runtime documentation" in appendix
    assert "re-verify" in appendix


def test_notes_appendix_includes_nonempty_notes():
    appendix = notes_appendix(KNOWN_FEATURE_MATRIX)
    assert "GBNF" in appendix  # from llama.cpp's grammar notes


def test_unverified_entries_lists_every_feature_when_nothing_verified():
    pending = unverified_entries(KNOWN_FEATURE_MATRIX)
    assert len(pending) == len(KNOWN_FEATURE_MATRIX) * 6


def test_unverified_entries_excludes_verified_features():
    fake_matrix = {
        "x": RuntimeFeatures(
            runtime="x-runtime",
            structured_output=FeatureSupport("yes", True),
            grammar=FeatureSupport("no", False),
            token_counting_endpoint=FeatureSupport("no", False),
            streaming=FeatureSupport("no", False),
            cancellation=FeatureSupport("no", False),
            usage_reporting=FeatureSupport("no", False),
        )
    }
    pending = unverified_entries(fake_matrix)
    assert len(pending) == 5
    assert ("x-runtime", "Structured output") not in pending
