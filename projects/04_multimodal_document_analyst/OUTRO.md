# Outro — Project 4: Multimodal Document Analyst

## What this achieved

A real, running FastAPI service over multi-page PDFs that don't uniformly have a usable text
layer — real per-page routing between a text LLM and a VLM, with the routing decision backed by
real image-token-cost math instead of just a text-length heuristic; real structured extraction on
the pages that have text; real page-cited Q&A with each citation independently verified against
which pages were actually analyzed; and an evaluation harness that proves both success and
failure are caught correctly, including one genuine, undoctored finding (see REPORT.md) about
where substring-correctness and citation-correctness diverge. It's also this course's clearest
example yet of closing a *found* gap rather than building net-new capability: Module 18 already
built every individual multimodal piece (PDF rendering, text-layer extraction, the VLM protocol,
the routing signal, image-token math) — this project's one piece of genuinely new logic
(`doc_routing.decide_route()`) is fewer than 30 lines of composition, wiring code that already
existed into a decision nothing in the repo had made before.

## What's still open (honest-skip, not forgotten)

- **Real VLM visual reasoning.** Every routing decision and every memory-cost number in
  REPORT.md is real; what a real VLM would actually see and describe on the scanned page 3 is
  not — `FakeVLM` returns the same scripted string regardless of image content, by design (a
  VLM's visual reasoning can't be meaningfully faked, only its call shape). That's the load-bearing
  number to get once this runs on the resourced 32GB Mac with `MlxVisionLanguageModel` injected
  via `build_doc_context(..., vlm=...)` — no other code change needed.
- **Single-page-per-document fixture diversity.** The one committed fixture exercises exactly one
  digital-native-vs-scanned split (2 text pages, 1 scanned). A second fixture with a different
  shape — a densely tabular page, a multi-column layout, a page that's mostly a photo with a small
  caption — would stress `extract_layout()`/`extract_tables()` (both real and tested at the
  Module 18 level, but never exercised end-to-end through this project's own ingestion pipeline)
  and give the routing threshold (`min_text_chars=40`) a more interesting edge to be tested
  against.
- **`min_text_chars=40` is still a single, repo-wide default.** `decide_route()` exposes it as a
  parameter, but nothing in this project's API surface lets an operator tune it per document type
  — a form with dense small print might have a real text layer well under 40 chars per page
  segment despite being perfectly text-LLM-answerable at the whole-page level, a genuinely
  different case from a truly blank scanned page.
- **No image input path**, only PDF. PROPOSAL.md documented this as an extension point rather
  than a functional gap: `render_page_to_image()` already returns a real `PIL.Image`, so a
  `POST /documents/image` endpoint accepting a raw image and routing it straight to the VLM path
  (skipping PDF loading and text-layer extraction entirely, since there's no text layer to check)
  is a small, well-scoped addition, not a redesign.

## What to explore next

- **A real VLM-vs-text-LLM head-to-head, once on the resourced Mac.** Run the exact same
  evaluation harness with a real `MlxVisionLanguageModel` against a fixture page engineered to
  have *both* a usable (if noisy) text layer and meaningful visual content (e.g., a page with a
  data table rendered as an image alongside body text) — forcing a genuine quality comparison
  between the routed-to path and the one `should_use_vlm()` would have skipped, the direct
  empirical version of PROPOSAL.md's functional requirement 7 that this project's routing proves
  structurally but can't yet prove qualitatively.
- **Layout- and table-aware extraction**, using the already-real, already-tested
  `extract_layout()`/`extract_tables()` (Module 18) as additional signal into
  `DocumentFieldExtraction` — right now extraction only sees the flattened text layer; a page
  whose key fields live in table cells (an invoice line-item table, a form's data grid) would
  benefit from passing structured layout, not just concatenated text, into the extraction prompt.
- **A real LLM-as-judge pass on VLM descriptions**, using Module 13's `LocalJudge` the same way
  Project 2's OUTRO.md proposes for RAG faithfulness — once real VLM output exists, judging
  whether a description is actually faithful to the rendered page (not just non-empty) is a
  natural, reusable next evaluation axis.
- **A confidence-aware review queue UI**, surfacing `needs_review=True` pages (every VLM-routed
  page, by this project's own design decision, plus any low-confidence extraction) the way
  Project 1's `ReviewQueue` already tracks them — currently queryable via `GET /documents/{id}`
  but with no dedicated "what needs a human" view across documents.
