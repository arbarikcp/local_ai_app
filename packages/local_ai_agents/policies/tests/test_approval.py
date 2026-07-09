from local_ai_agents.policies.approval import AutoApprovalGate, CallbackApprovalGate, NullApprovalGate


class TestNullApprovalGate:
    async def test_always_denies(self):
        gate = NullApprovalGate()
        assert await gate.request_approval("write_file", {"path": "x"}) is False


class TestCallbackApprovalGate:
    async def test_delegates_to_the_injected_callback(self):
        async def approve_everything(tool_name: str, arguments: dict) -> bool:
            return True

        gate = CallbackApprovalGate(approve_everything)
        assert await gate.request_approval("write_file", {}) is True

    async def test_callback_can_deny(self):
        async def deny_everything(tool_name: str, arguments: dict) -> bool:
            return False

        gate = CallbackApprovalGate(deny_everything)
        assert await gate.request_approval("write_file", {}) is False

    async def test_callback_receives_the_tool_name_and_arguments(self):
        received = {}

        async def recording_callback(tool_name: str, arguments: dict) -> bool:
            received["tool_name"] = tool_name
            received["arguments"] = arguments
            return True

        gate = CallbackApprovalGate(recording_callback)
        await gate.request_approval("write_file", {"path": "notes.txt"})
        assert received == {"tool_name": "write_file", "arguments": {"path": "notes.txt"}}


class TestAutoApprovalGate:
    async def test_always_approves(self):
        gate = AutoApprovalGate()
        assert await gate.request_approval("write_file", {}) is True
