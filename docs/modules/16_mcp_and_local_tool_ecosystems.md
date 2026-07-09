# Module 16 — MCP and Local Tool Ecosystems

> Phase: Agents/tools · Bible reference: [curriculum.md §26](../../curriculum.md#26-module-16--mcp-and-local-tool-ecosystems)

## Goal

Understand MCP-style integration without letting protocol enthusiasm replace architecture.

> **Machine note:** this module is deterministic Python end to end - no LLM dependency at all.
> `McpLikeServer` dispatches real request/response objects in-process; "connecting tool results
> to a local LLM" (Lab 6) is a real prompt-assembly step, `FakeRuntime`-backed for the actual
> generation call, same pattern as every prior module.

## Repo structure note

Curriculum's literal deliverable path is `packages/local_ai_core/tools/mcp_like_server.py`;
this build uses `packages/local_ai_agents/tools/mcp_like_server.py` instead, for the same reason
Module 9's repo structure note gave: curriculum.md §8's own canonical structure has no
`local_ai_core/tools/` at all, and this module's entire point is exposing Module 14's
`Tool`/`ToolRegistry`/`ToolExecutor` over an MCP-shaped protocol layer - it belongs next to the
tools it wraps, not in a different package. `mcp_resources.py` and `mcp_prompts.py` join it in
the same directory for the two other MCP primitives (theory doc §2-3).

## What this module is not

**Not a spec-compliant MCP implementation.** Real MCP is JSON-RPC 2.0 over stdio/HTTP/SSE, with
capability negotiation, session lifecycle, and notifications - a real protocol spec, not a few
functions. `mcp_like_server.py` implements the *shape* MCP teaches (resources, prompts, tools,
method-based dispatch, discovery separate from authorization) as real, in-process
request/response dispatch - explicitly named "MCP-*like*" everywhere in this codebase, matching
curriculum's own filename, so nothing here is mistaken for spec compliance.

## Core topics

### 1. What MCP is

A standard for connecting model-facing applications to **tools** (actions with side effects),
**resources** (addressable read-only data), and **prompts** (reusable parameterized templates) -
curriculum's own three-primitive breakdown, each with a real registry in this module.

### 2. Resources

`mcp_resources.py`'s `ResourceRegistry` - addressable, read-only content behind Module 14's
`sandbox.py` path containment (a resource URI is a sandboxed relative path, never an
unconstrained filesystem read). `read()` returns real file content, not a placeholder.

### 3. Prompts

`mcp_prompts.py`'s `PromptRegistry` - named, parameterized templates (e.g. Module 11's minimal
RAG prompt, exposed as an exemplar) that a client can list and render with arguments - real
string formatting, real missing-argument errors, not a description of the concept.

### 4. Tools

Reused unchanged from Module 14: `Tool`, `ToolRegistry`, `ToolExecutor`. This module adds no new
tool abstraction - MCP's "tools" primitive maps directly onto what Module 14 already built.

### 5. Local MCP server

`mcp_like_server.py`'s `McpLikeServer` - a real, in-process method dispatcher:
`tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`.
Every `tools/call` request is routed through Module 14's `ToolExecutor`, not called directly -
theory doc §8's entire point made structurally true, not just asserted.

### 6. Filesystem tool

Module 14's `search_files` tool, exposed through the server's `tools/list`/`tools/call`, plus a
new `read_resource` MCP resource (a single sandboxed file's content) demonstrating the
tool/resource distinction: search returns *matches*, a resource returns *content*.

### 7. Database tool

Module 14's `sql_query` tool, exposed the same way - no new SQL logic, same two-layer defense
(query-text validation + real read-only SQLite connection) from Module 14.

### 8. Security boundary

**Real, structural proof, not a claim**: `McpLikeServer.dispatch()` for `tools/call` never
calls a tool's handler directly - it always goes through `ToolExecutor.execute()`, so
permissions, argument validation, approval gating, tool budgets, and audit logging (all of
Module 14's enforcement chain) apply to every MCP-shaped call exactly as they would to a direct
call. The protocol layer adds a dispatch shape; it does not add or remove any security guarantee.

### 9. Tool discovery

**Discovery is not authorization** (theory doc's own gotcha, made testable): `tools/list`
returns every registered tool's schema regardless of the caller's role - a role with zero
permitted tools still sees the full tool list. `tools/call` for that same role is denied by
`PermissionPolicy` exactly as it would be without MCP in the picture. §"Real proof" in the
deliverable notes demonstrates both facts about the same tool and role in one run.

### 10. Approval policies

A dangerous tool (`write_file`) called via `tools/call` still requires a real `ApprovalGate`
approval - `McpLikeServer` does not special-case or bypass Module 14's dangerous-tool gate for
protocol-shaped calls.

### 11. Logging

Every `dispatch()` call - not just `tools/call`, also `resources/read` and `prompts/get` - is
recorded to a real SQLite-backed audit log (`AuditLog`, reused from Module 14), so an MCP-shaped
integration has the same observability a direct integration would.

### 12. Compatibility with local models

Not new code: `tools/tool_call.py` (Module 14) already solves "can a small local model reliably
produce a tool-call JSON payload," with the same strict-parse-or-fail discipline Module 8's
structured-output ladder established. An MCP-shaped client asking a local model which tool to
call is the identical problem with an identical answer.

### 13. MCP vs A2A

Theory only, per curriculum's own note: MCP is the tool/resource/prompt integration layer;
A2A-style protocols are agent-to-agent coordination, a different layer this course's capstone
should not depend on unless the core local assistant is already reliable. Not implemented -
genuinely out of scope, not deferred.

## MCP teaching principle (the module's central claim, made structurally true)

> MCP does not remove the need for authorization, validation, sandboxing, human approval, audit
> logging, data minimization, or output verification.

Every one of these seven items is a real, already-built mechanism this module routes MCP-shaped
requests *through*, not around:

| Requirement | Enforced by |
|---|---|
| Authorization | Module 14's `PermissionPolicy`, checked inside `ToolExecutor` |
| Validation | Module 14's Pydantic `args_model.model_validate()`, inside `ToolExecutor` |
| Sandboxing | Module 14's `sandbox.py` path containment, inside both tools and `ResourceRegistry` |
| Human approval | Module 14's `ApprovalGate`, inside `ToolExecutor` |
| Audit logging | Module 14's `AuditLog`, now covering every `dispatch()` call, not just tool calls |
| Data minimization | `max_results`/`max_rows` schema limits (Module 14), unchanged |
| Output verification | Injection-pattern screening (§"Gotchas") applied to tool descriptions and resource content before they're exposed |

## Gotchas

- **Tool discovery is not authorization** - §9, demonstrated directly.
- **A protocol does not make a tool safe** - the entire enforcement chain above is what makes a
  tool safe; MCP-shaped dispatch is a routing convenience, nothing more.
- **Tool descriptions are prompt surface area and can be injection vectors** - `tools/list` and
  `resources/read` responses are screened with Module 13's
  `detect_prompt_injection_patterns()` before being returned, surfacing a `flagged_patterns`
  field rather than silently trusting arbitrary tool-author-supplied text. A screen, not a
  guarantee - same honesty standard as every heuristic in this course.
- **Local filesystem tools need path allowlists and approval rules** - already true via
  `sandbox.py` and `ApprovalGate`; this module doesn't re-solve it, just confirms it still
  applies once wrapped in MCP-shaped dispatch.
- **MCP/A2A enthusiasm should not replace deterministic policy enforcement** - the whole point
  of routing every `tools/call` through `ToolExecutor` rather than calling handlers directly.

## Hands-on labs

1. **Build a tiny local MCP-like tool server** — `mcp_like_server.py`,
   `scripts/module_16/build_server_demo.py`.
2. **Expose a file search tool** — Module 14's `search_files`, via `tools/list`/`tools/call`.
3. **Expose a read-only SQLite query tool** — Module 14's `sql_query`, same mechanism.
4. **Add tool metadata** — `Tool.json_schema()` (Module 14) already provides this; surfaced
   through `tools/list`.
5. **Add tool invocation logging** — `AuditLog`, every `dispatch()` call,
   `scripts/module_16/security_boundary_demo.py`.
6. **Connect tool results to local LLM** — same script, a real prompt built from a `tools/call`
   result, `FakeRuntime`-backed generation.

## Deliverable

```text
packages/local_ai_agents/tools/
  mcp_resources.py
  mcp_prompts.py
  mcp_like_server.py
  tests/
scripts/module_16/
  build_server_demo.py
  security_boundary_demo.py
reports/tool_ecosystem_security_notes.md
```
