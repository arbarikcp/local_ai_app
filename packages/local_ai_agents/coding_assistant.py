"""The local coding assistant (theory doc's architecture diagram) -
wires Module 15's `WorkflowGraph` engine around this module's real tools:
search -> read -> propose a patch -> validate -> (approval) apply ->
(approval) run tests -> report. Mirrors `local_ai_rag/pipeline.py`'s role
(Module 11) as the orchestration layer tying primitives from several
subpackages into one real pipeline.
"""

from __future__ import annotations

from pathlib import Path

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_agents.planners.memory import AgentMemory
from local_ai_agents.planners.workflow_graph import END, WorkflowGraph, WorkflowNode
from local_ai_agents.tools.patch_tools import PatchFormatError, apply_patch, propose_patch, validate_patch_format
from local_ai_agents.tools.read_file import read_file_lines
from local_ai_agents.tools.run_tests import run_tests
from local_ai_agents.tools.search_repo import search_repo


def build_coding_assistant_graph(repo_dir: Path, runtime: LLMRuntime, model: str) -> WorkflowGraph:
    async def search_node(state: dict, memory: AgentMemory) -> dict:
        matches = search_repo(repo_dir, state["query"])
        memory.add("observation", f"search_repo('{state['query']}') found {len(matches)} match(es)")
        if not matches:
            return {**state, "target_file": None, "failure_reason": "search_repo found no matches"}
        return {**state, "target_file": matches[0].path}

    async def read_context_node(state: dict, memory: AgentMemory) -> dict:
        content = read_file_lines(repo_dir, state["target_file"])
        memory.add("observation", f"read_file('{state['target_file']}')")
        return {**state, "file_contents": {state["target_file"]: content}}

    async def propose_node(state: dict, memory: AgentMemory) -> dict:
        patch_text = await propose_patch(state["instruction"], state["file_contents"], runtime, model)
        memory.add("reasoning", f"proposed a patch for '{state['instruction']}'")
        return {**state, "patch_text": patch_text}

    async def validate_node(state: dict, memory: AgentMemory) -> dict:
        try:
            parsed = validate_patch_format(state["patch_text"])
            memory.add("observation", f"patch format valid for '{parsed.file_path}'")
            return {**state, "patch_valid": True}
        except PatchFormatError as exc:
            memory.add("observation", f"patch format invalid: {exc}")
            return {**state, "patch_valid": False, "failure_reason": str(exc)}

    async def apply_node(state: dict, memory: AgentMemory) -> dict:
        applied_path = apply_patch(repo_dir, state["patch_text"])
        memory.add("observation", f"applied patch to '{applied_path}'")
        return {**state, "applied_path": applied_path}

    async def run_tests_node(state: dict, memory: AgentMemory) -> dict:
        result = await run_tests(repo_dir)
        memory.add("observation", f"tests {'passed' if result.passed else 'failed'} (exit {result.exit_code})")
        return {**state, "tests_passed": result.passed, "test_stdout": result.stdout}

    async def report_node(state: dict, memory: AgentMemory) -> dict:
        memory.add("reasoning", "reporting final result")
        return state

    graph = WorkflowGraph(start_node="search")
    graph.add_node(WorkflowNode(name="search", run=search_node))
    graph.add_node(WorkflowNode(name="read_context", run=read_context_node))
    graph.add_node(WorkflowNode(name="propose", run=propose_node))
    graph.add_node(WorkflowNode(name="validate", run=validate_node))
    graph.add_node(WorkflowNode(name="apply", run=apply_node, requires_approval_tool="apply_patch"))
    graph.add_node(WorkflowNode(name="run_tests", run=run_tests_node, requires_approval_tool="run_tests"))
    graph.add_node(WorkflowNode(name="report", run=report_node))

    graph.add_edge("search", "read_context", condition=lambda s: s.get("target_file") is not None)
    graph.add_edge("search", "report")  # no match found -> report directly, nothing to patch
    graph.add_edge("read_context", "propose")
    graph.add_edge("propose", "validate")
    graph.add_edge("validate", "apply", condition=lambda s: s.get("patch_valid") is True)
    graph.add_edge("validate", "report")  # invalid patch -> report, never attempt to apply it
    graph.add_edge("apply", "run_tests")
    graph.add_edge("run_tests", "report")
    graph.add_edge("report", END)
    return graph
