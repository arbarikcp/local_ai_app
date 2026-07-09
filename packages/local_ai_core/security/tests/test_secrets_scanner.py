from local_ai_core.security.secrets_scanner import scan_for_secrets


class TestAwsAccessKeyDetection:
    def test_detects_an_aws_access_key(self):
        result = scan_for_secrets("key: AKIAIOSFODNN7EXAMPLE")
        assert result.matches["aws_access_key"] == 1
        assert result.found_secrets is True


class TestPrivateKeyDetection:
    def test_detects_a_pem_private_key_header(self):
        result = scan_for_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIB...")
        assert result.matches["private_key_header"] == 1


class TestBearerTokenDetection:
    def test_detects_a_bearer_token(self):
        result = scan_for_secrets("Authorization: Bearer abcdef1234567890XYZ")
        assert result.matches["bearer_token"] == 1


class TestGenericApiKeyDetection:
    def test_detects_an_api_key_assignment(self):
        result = scan_for_secrets('api_key = "sk-abcdefghijklmnopqrstuvwx"')
        assert result.matches["generic_api_key"] == 1


class TestNoSecretsPresent:
    def test_clean_text_has_zero_matches(self):
        result = scan_for_secrets("The weather is nice today.")
        assert result.total_matches == 0
        assert result.found_secrets is False


class TestMultipleSecretsInOneText:
    def test_detects_every_category_present(self):
        text = "AKIAIOSFODNN7EXAMPLE and Bearer abcdef1234567890XYZ"
        result = scan_for_secrets(text)
        assert result.total_matches == 2
