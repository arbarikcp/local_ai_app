from local_ai_agents.planners.memory import AgentMemory


class TestAdd:
    def test_entries_are_indexed_in_insertion_order(self):
        memory = AgentMemory()
        first = memory.add("reasoning", "thinking about it")
        second = memory.add("tool_call", "calculator(2+2)")
        assert first.step_index == 0
        assert second.step_index == 1

    def test_carries_optional_structured_data(self):
        memory = AgentMemory()
        entry = memory.add("tool_call", "calculator", data={"tool": "calculator", "arguments": {"expression": "2+2"}})
        assert entry.data["tool"] == "calculator"

    def test_len_reflects_entry_count(self):
        memory = AgentMemory()
        memory.add("reasoning", "a")
        memory.add("reasoning", "b")
        assert len(memory) == 2


class TestEntries:
    def test_returns_a_copy_not_the_internal_list(self):
        memory = AgentMemory()
        memory.add("reasoning", "a")
        entries = memory.entries()
        entries.append("mutated")
        assert len(memory) == 1


class TestTranscript:
    def test_renders_every_entry_with_its_kind(self):
        memory = AgentMemory()
        memory.add("reasoning", "I should check the ticket count")
        memory.add("observation", "3 open tickets found")
        transcript = memory.transcript()
        assert "[reasoning] I should check the ticket count" in transcript
        assert "[observation] 3 open tickets found" in transcript

    def test_empty_memory_produces_an_empty_transcript(self):
        assert AgentMemory().transcript() == ""
