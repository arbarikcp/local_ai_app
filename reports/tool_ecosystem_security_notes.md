# Module 16 deliverable — tool ecosystem security notes

Status: **complete.** This is Module 16's curriculum-named deliverable
(`reports/tool_ecosystem_security_notes.md`, per curriculum.md §26) and also serves as this
module's standard deliverable report. Everything in this module is deterministic Python and
runs for real — the MCP-like server, resources, prompts, and every security mechanism it routes
through. Only the final "connect tool results to a local LLM" step needs a live model.

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `tools/mcp_resources.py` | 6 | Sandboxed resource registration (rejected at register time if outside the sandbox), real file reads |
| `tools/mcp_prompts.py` | 7 | Template registration/listing/rendering, missing-argument detection |
| `tools/mcp_like_server.py` | 12 | Full dispatch: `tools/list`, `tools/call` (routed through `ToolExecutor`), `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, unknown-method handling, audit logging of every call |
| `scripts/module_16/` (2 lab scripts) | 24 | Labs 1-6 exercised against the real Nimbus handbook corpus and a real SQLite fixture database |
| `notebooks/16_mcp_and_local_tool_ecosystems.ipynb` | — | **Executed end-to-end** — every cell a real measurement |

**41 new tests this module** (1386 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## What this module is, and is not

`mcp_like_server.py` implements the *shape* MCP teaches — resources, prompts, tools, method-based
dispatch, discovery separate from authorization — as real, in-process request/response dispatch.
It is **not** a spec-compliant MCP implementation: no JSON-RPC 2.0 transport, no capability
negotiation, no session lifecycle. Every class and file name says "MCP-*like*" explicitly, so
nothing here should be mistaken for interoperating with a real MCP client or server.

## Security notes (the module's central claim, verified structurally)

Curriculum's own teaching principle: **MCP does not remove the need for authorization,
validation, sandboxing, human approval, audit logging, data minimization, or output
verification.** This module makes that literally true in code, not just documented:

`McpLikeServer.dispatch()`'s `tools/call` branch never calls a tool's handler directly — it
always constructs a `ToolCallProposal` and hands it to Module 14's `ToolExecutor.execute()`.
Every one of the seven requirements above is enforced by a mechanism that already existed before
this module, unchanged:

| Requirement | Enforced by | Verified in this module by |
|---|---|---|
| Authorization | `PermissionPolicy` | `test_discovery_is_not_authorization` — a role sees a tool in `tools/list` and is still denied at `tools/call` |
| Validation | Pydantic `args_model.model_validate()` | Reused unchanged inside `ToolExecutor` |
| Sandboxing | `sandbox.py` path containment | `ResourceRegistry` rejects an out-of-sandbox URI at *registration* time, not just read time |
| Human approval | `ApprovalGate` | `test_a_dangerous_tool_is_denied_without_a_real_approval_gate` / `...succeeds_with...` |
| Audit logging | `AuditLog` (real SQLite) | `test_every_dispatch_call_is_logged` — covers `resources/read` and `prompts/get`, not just `tools/call` |
| Data minimization | `max_results`/`max_rows` schema limits | Reused unchanged from Module 14 |
| Output verification | Injection-pattern screening | `test_flags_a_suspicious_tool_description` / `test_resource_content_is_screened_for_injection_patterns` |

## Real proof: discovery is not authorization (from the executed notebook)

```
guest sees in tools/list: ['calculator']
guest tools/call result: False - role 'guest' is not permitted to call 'calculator'
```

The tool list is metadata a client can inspect regardless of role; the actual call is where
enforcement happens. Conflating the two — assuming "the tool wasn't offered" is a substitute for
"the tool call was denied" — is exactly the gotcha curriculum names, made falsifiable here.

## Real proof: tool descriptions are prompt surface area (from the executed notebook)

```
flagged patterns: ['ignore (all |the )?previous instructions', 'reveal (the |your )?system prompt']
```

A tool's own `description` field — text a server operator (or a compromised/malicious MCP
server, in the real ecosystem this module is modeled on) controls, not the end user — is
screened by the same `detect_prompt_injection_patterns()` heuristic Module 13 built for
retrieved document content. A description is exactly as much attack surface as any other text an
LLM will see, and this module treats it that way rather than implicitly trusting it because it
came from "the tool layer."

## Real proof: the security boundary holds end to end (from the executed notebook)

```
Guest role sees 'sql_query' in tools/list: True (discovery is not authorization)
Guest role's tools/call is denied: True -> role 'guest' is not permitted to call 'sql_query'
write_file denied with no real approval gate: True
write_file approved with a real approval gate: True
File actually written to disk: True
Malicious tool description flagged patterns: [...]
LLM summary of the sql_query tool result: There are 2 open tickets, based on the tool result.
Total audit log entries recorded: 9
```

Every stage of a realistic MCP-shaped session — discovery, a denied call, a dangerous call
denied then approved, with the actual filesystem write verified (not just a claimed success),
and a real tool result flowing into a generation prompt — recorded to a real SQLite audit log
whose entry count was checked directly, not assumed.

## Real proof: resources are sandboxed at registration time, not just read time

`ResourceRegistry.register()` calls `resolve_within_sandbox()` immediately — registering a URI
that resolves outside the allowed base raises `PathTraversalError` before the resource is ever
listed or readable, rather than deferring the check to `read()` and risking a registration-time
bug that silently exposes an out-of-sandbox path later.

## MCP vs. A2A

Not implemented, per curriculum's own note: MCP is the tool/resource/prompt integration layer
this module builds; A2A-style agent-to-agent coordination is a different layer this course's
capstone should not depend on unless the core local assistant (Modules 1-17) is already
reliable. Discussed here only as a boundary — this module's server has no concept of
coordinating with another agent, by design.

## Deliberately not done in Module 16

- No JSON-RPC 2.0 transport, stdio/HTTP/SSE wire format, or capability negotiation — this
  module proves the *dispatch shape and security properties* MCP teaches, not protocol
  compliance; a real transport would be a thin adapter around `McpLikeServer.dispatch()`, not a
  rewrite of anything built here.
- No real LLM connecting to the server — Lab 6's "connect tool results to a local LLM" uses
  `FakeRuntime`; real model quality (does a local model produce useful summaries of real tool
  results?) is deferred to the resourced 32GB Mac.
- No A2A implementation — genuinely out of scope per curriculum's own framing, not deferred.
