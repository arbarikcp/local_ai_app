import pytest

from scorers.json_validity import (
    has_required_keys,
    invalid_json_rate,
    is_valid_json,
    strip_markdown_fence,
    try_parse_json,
)


def test_strip_markdown_fence_removes_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert strip_markdown_fence(text) == '{"a": 1}'


def test_strip_markdown_fence_removes_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert strip_markdown_fence(text) == '{"a": 1}'


def test_strip_markdown_fence_is_noop_on_unfenced_text():
    text = '{"a": 1}'
    assert strip_markdown_fence(text) == text


def test_try_parse_json_handles_fenced_and_unfenced():
    assert try_parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert try_parse_json('{"a": 1}') == {"a": 1}


def test_try_parse_json_returns_none_on_garbage():
    assert try_parse_json("not json at all") is None
    assert try_parse_json("") is None


def test_is_valid_json_true_and_false_cases():
    assert is_valid_json('{"a": 1}') is True
    assert is_valid_json("nope") is False


def test_has_required_keys_all_present():
    text = '{"name": "Maria", "age": 29, "city": null}'
    assert has_required_keys(text, ["name", "age", "city"]) is True


def test_has_required_keys_missing_one():
    text = '{"name": "Maria", "age": 29}'
    assert has_required_keys(text, ["name", "age", "city"]) is False


def test_has_required_keys_false_when_not_an_object():
    assert has_required_keys("[1, 2, 3]", ["name"]) is False
    assert has_required_keys("not json", ["name"]) is False


def test_invalid_json_rate_computes_fraction_invalid():
    preds = ['{"a": 1}', "not json", '{"b": 2}', "```json\n{\"c\": 3}\n```"]
    assert invalid_json_rate(preds) == pytest.approx(0.25)


def test_invalid_json_rate_empty_list_is_zero():
    assert invalid_json_rate([]) == 0.0
