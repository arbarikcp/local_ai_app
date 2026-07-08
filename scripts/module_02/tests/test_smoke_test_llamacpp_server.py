import smoke_test_llamacpp_server as sut


def test_check_openai_client_importable_is_a_bool():
    assert isinstance(sut.check_openai_client_importable(), bool)


def test_run_skips_cleanly_when_server_unreachable(capsys):
    # No llama-cpp-python server is expected to be running in the test environment.
    exit_code = sut.run(base_url="http://localhost:8080/v1", prompt="hi", model="test-model")
    assert exit_code == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "SKIPPED" in combined
