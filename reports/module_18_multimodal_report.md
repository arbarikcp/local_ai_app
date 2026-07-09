# Module 18 deliverable — multimodal local applications report

Status: **complete.** Almost no honest-skip surface — `PyMuPDF`, `pdfplumber`, and `Pillow` are
real libraries, not LLM runtimes or model weights (same reasoning as Module 10's
`chromadb`/`lancedb`), so PDF rendering, text/layout/table extraction, and image preprocessing
all run for real. Only actual VLM inference stays honest-skip (`FakeVLM`-backed, DI pattern,
"Enabling this for real" instructions in `vlm.py`'s docstring).

## What's built and verified

| Artifact | Tests | What they verify |
|---|---:|---|
| `datasets/multimodal/sample_invoice.pdf` | — | A real, committed digital-native PDF with a real text layer and a real drawn table |
| `datasets/multimodal/scanned_receipt.pdf` | — | A real, committed PDF with a real embedded image and **genuinely no text layer** |
| `multimodal/pdf_extraction.py` | 9 | Real page rendering, real text-layer extraction, real word-level layout, real table extraction |
| `multimodal/image_preprocessing.py` | 9 | Real Pillow grayscale/contrast/resize/rotate operations |
| `multimodal/vlm.py` | 5 | DI pattern for a real VLM adapter, `FakeVLM` call-history tracking |
| `multimodal/memory_cost.py` | 8 | Real patch-based token estimation, real context-budget-fraction math |
| `multimodal/routing.py` | 6 | `should_use_vlm()`'s threshold logic on both branches |
| `local_ai_rag/loaders/pdf_loader.py` | 7 | Real per-page `Document` loading, page-encoded `doc_id` |
| `scripts/module_18/` (4 lab/fixture scripts) | 24 + 4 | Labs 1-6 exercised against the two real fixtures |
| `notebooks/18_multimodal_local_applications.ipynb` | — | **Executed end-to-end** — every cell a real measurement |

**70 new tests this module** (1514 total across the repo, 2 correctly-skipped, all passing),
`ruff check .` clean.

## Scope note: OCR (not honest-skipped — genuinely out of scope, documented as such)

Following Module 12's precedent for PDF/OCR tooling: real OCR needs either a system binary
(`tesseract`) or downloaded deep-learning weights (`easyocr`), both ruled out by this repo's
machine constraint. Rather than a placeholder, this module relies on **real PDF text-layer
extraction** wherever a document has one - genuinely real text, just not OCR output. The "OCR
gap" (a document with no text layer and no OCR available) is made honestly visible, not hidden:
`scanned_receipt.pdf` is a real fixture with a real image and zero extractable text, and
`should_use_vlm()` correctly and visibly escalates it rather than silently returning nothing.

## Real proof: the two fixtures behave exactly as their names claim

```
sample_invoice.pdf text layer: 197 chars
scanned_receipt.pdf text layer: 0 (genuinely empty - a real image-only page)
```

Verified directly by opening both PDFs with two independent libraries (`PyMuPDF` and
`pdfplumber`) during fixture construction, not assumed from how the fixtures were built.
`scanned_receipt.pdf` contains a real embedded PNG image (verified via
`page.get_images()` returning a non-empty list) and deliberately no `insert_text()` call at
all - the empty text layer is a structural fact about the file, not a quirk of one extraction
library.

## Real proof: table extraction works on a real drawn table

```
- ['Item', 'Qty', 'Unit Price']
- ['Personal Plan (monthly)', '1', '$6.99']
- ['Extra Storage (100GB)', '2', '$2.50']
- ['API Overage Fee', '1', '$4.00']
```

`pdfplumber.extract_tables()` correctly reconstructs the exact rows and columns from a real
grid of drawn rectangles and inserted text - not a hand-labeled expected output, the actual
return value of a real library call.

## Real proof: the memory-cost math produces a genuine overflow

```
224x224 image -> 256 tokens (3.2% of an 8000-token context window)
1024x1024 image -> 5476 tokens (68.5% of an 8000-token context window)
2048x2048 image -> 21609 tokens (270.1% of an 8000-token context window)
```

A 2048×2048 image genuinely costs more tokens than an entire 8000-token context window has room
for (270.1%) - a real, checkable consequence of the patch-based formula real ViT-style vision
encoders use, not an assumed "images are expensive" claim.

## Real proof: `should_use_vlm()` routes both real fixtures correctly

```
sample_invoice.pdf: routed to text_llm (text layer has 197 chars >= 40 threshold)
scanned_receipt.pdf: routed to vlm (text layer has only 0 chars < 40 threshold)
```

The full pipeline (`vlm_routing_demo.py`) applies this decision automatically per document and
only calls the (fake) VLM for the one that genuinely needs it - the "recommended pipeline
principle" diagram made structurally true, the same discipline Module 16 applied to the MCP
security boundary.

## A real bug found and fixed while building this module

`citation_packer.py`'s `extract_citations()` regex (`\[([A-Za-z0-9_.-]+::\d+)\]`) assumed a
chunk_id has exactly one `"::"` separator (`doc_id::chunk_index`). Module 18's PDF-page doc_ids
add a second one (`pdf_stem::pageN::chunk_index`), and the regex's character class didn't allow
colons in the segment *before* the final `::digits`, so a real citation like
`[sample_invoice::page1::0]` silently matched nothing - `extract_citations()` returned an empty
list even though the citation was present in the answer text verbatim. Caught by running
`multimodal_rag_demo.py` and noticing `chunk_level_citations` was empty despite the scripted
answer clearly containing a citation marker. Fixed by widening the character class to
`[A-Za-z0-9_.:-]+::\d+` (colons allowed throughout, not just as the separator). A second, related
issue in `summarize_source_citations()` was fixed at the same time: it used `split("::", 1)`
(splitting from the *left*), which would have collapsed `sample_invoice::page1::0` down to just
`sample_invoice`, losing the page number. Changed to `rsplit("::", 1)` (splitting from the
*right*, stripping only the trailing chunk index) so a page-qualified source citation correctly
stays `sample_invoice::page1`. Regression tests for both fixes were added to Module 11/12's
existing `test_citation_packer.py`, not just this module's own tests - the shared function
needed the fix, not a Module-18-local workaround.

## Deliberately not done in Module 18

- **Real OCR** — see the scope note above; genuinely out of scope on this machine, not
  honest-skipped.
- **Real VLM inference** — `vlm.py`'s `MlxVisionLanguageModel` is fully built with the same
  lazy-import/DI pattern as every other real-model adapter in this course; real model quality
  (does a local VLM correctly answer "what is the total on this receipt?") is deferred to the
  resourced 32GB Mac.
- **Automatic skew-angle detection** — `image_preprocessing.py`'s `rotate()` rotates by a
  *given* angle; estimating what angle a crooked scan needs is a real computer-vision problem
  (needing OpenCV or a trained model) intentionally left undone rather than approximated badly.
- **Diagram/chart interpretation, form extraction** (curriculum's example use cases) — not
  separately implemented; both would route through the same `should_use_vlm()` → `FakeVLM`
  path already proven for receipts and screenshots, exercising no new code.
