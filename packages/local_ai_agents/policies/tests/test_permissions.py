from local_ai_agents.policies.permissions import PermissionPolicy


class TestIsAllowed:
    def test_unlisted_role_is_denied_everything(self):
        policy = PermissionPolicy()
        assert policy.is_allowed("guest", "calculator") is False

    def test_explicitly_allowed_tool_is_permitted(self):
        policy = PermissionPolicy()
        policy.allow("analyst", "calculator")
        assert policy.is_allowed("analyst", "calculator") is True

    def test_a_role_is_not_allowed_a_tool_it_was_never_granted(self):
        policy = PermissionPolicy()
        policy.allow("analyst", "calculator")
        assert policy.is_allowed("analyst", "write_file") is False

    def test_allow_all_grants_every_tool(self):
        policy = PermissionPolicy()
        policy.allow_all("admin")
        assert policy.is_allowed("admin", "calculator") is True
        assert policy.is_allowed("admin", "anything_not_yet_registered") is True

    def test_permissions_are_scoped_per_role(self):
        policy = PermissionPolicy()
        policy.allow("analyst", "calculator")
        assert policy.is_allowed("guest", "calculator") is False
