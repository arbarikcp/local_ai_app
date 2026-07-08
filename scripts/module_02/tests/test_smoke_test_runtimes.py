import smoke_test_runtimes as sut


def test_build_report_includes_all_sections():
    report = sut.build_report()
    assert "# Module 2" in report
    assert "## Machine profile" in report
    assert "## Dev tool check" in report
    assert "## Model cache report" in report
    assert "## Lab 2.2 — Ollama" in report
    assert "## Lab 2.3 — llama-cpp-python server" in report
    assert "## Lab 2.4 — MLX" in report


def test_build_report_records_skip_exit_codes_on_this_machine():
    # On the machine this course is authored on, no runtime is installed, so
    # every lab section should show a nonzero exit code rather than silently
    # omitting the section.
    report = sut.build_report()
    assert report.count("Exit code: 1") >= 2
