from mac_environment_check import (
    ToolCheck,
    _brew_formula,
    check_tool,
    detect_machine_profile,
    missing_required_tools,
    tools_to_markdown_table,
)


def test_check_tool_finds_a_tool_known_to_exist_in_this_repo_env():
    # `python3` must exist for pytest itself to be running.
    result = check_tool("python3", required=True)
    assert result.found is True
    assert result.path is not None


def test_check_tool_reports_missing_tool_cleanly():
    result = check_tool("definitely-not-a-real-binary-xyz", required=True)
    assert result.found is False
    assert result.path is None
    assert result.version is None


def test_missing_required_tools_only_flags_required_and_missing():
    checks = [
        ToolCheck(name="a", found=True, path="/bin/a", version="1.0", required=True),
        ToolCheck(name="b", found=False, path=None, version=None, required=True),
        ToolCheck(name="c", found=False, path=None, version=None, required=False),
    ]
    assert missing_required_tools(checks) == ["b"]


def test_brew_formula_maps_rg_to_ripgrep():
    assert _brew_formula("rg") == "ripgrep"


def test_brew_formula_defaults_to_same_name():
    assert _brew_formula("cmake") == "cmake"


def test_tools_to_markdown_table_renders_all_rows():
    checks = [
        ToolCheck(name="git", found=True, path="/usr/bin/git", version="git 2.40", required=True),
        ToolCheck(name="ollama", found=False, path=None, version=None, required=False),
    ]
    table = tools_to_markdown_table(checks)
    assert "git" in table
    assert "ollama" in table
    assert "n/a" in table


def test_detect_machine_profile_returns_populated_fields():
    profile = detect_machine_profile()
    assert profile.architecture
    assert isinstance(profile.is_apple_silicon, bool)
    assert profile.python_version
