import pytest

from scorers.rubric_judge import (
    build_judge_prompt,
    judge,
    mean_score,
    parse_judge_score,
)


def test_build_judge_prompt_includes_task_prediction_and_reference():
    prompt = build_judge_prompt("Summarize the text", "a summary", reference="the gold summary")
    assert "Summarize the text" in prompt
    assert "a summary" in prompt
    assert "the gold summary" in prompt


def test_build_judge_prompt_handles_missing_reference():
    prompt = build_judge_prompt("Summarize the text", "a summary")
    assert "no reference provided" in prompt


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Score: 4\nGood coverage of key points.", 4),
        ("score:5", 5),
        ("SCORE: 1 - poor", 1),
        ("I think this deserves a Score: 3.", 3),
    ],
)
def test_parse_judge_score_extracts_score(response, expected):
    assert parse_judge_score(response) == expected


def test_parse_judge_score_returns_none_when_unparseable():
    assert parse_judge_score("This response is pretty good overall.") is None


def test_judge_calls_injected_fn_and_parses_result():
    def fake_judge_fn(prompt: str) -> str:
        assert "Summarize" in prompt
        return "Score: 4\nCovers the main points."

    result = judge(fake_judge_fn, "Summarize the article", "a decent summary")
    assert result.score == 4
    assert result.parse_succeeded is True
    assert "Covers the main points" in result.raw_response


def test_judge_marks_parse_failure_when_judge_response_has_no_score():
    def fake_judge_fn(prompt: str) -> str:
        return "This is fine I guess."

    result = judge(fake_judge_fn, "task", "prediction")
    assert result.score is None
    assert result.parse_succeeded is False


def test_mean_score_averages_parsed_scores_and_ignores_failures():
    def make_fn(text):
        return lambda prompt: text

    results = [
        judge(make_fn("Score: 4"), "t", "p"),
        judge(make_fn("Score: 2"), "t", "p"),
        judge(make_fn("no score here"), "t", "p"),
    ]
    assert mean_score(results) == pytest.approx(3.0)


def test_mean_score_returns_none_when_nothing_parsed():
    def make_fn(text):
        return lambda prompt: text

    results = [judge(make_fn("no score"), "t", "p")]
    assert mean_score(results) is None
