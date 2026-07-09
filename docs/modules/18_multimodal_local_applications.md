# Module 18 — Multimodal Local Applications

> Phase: Advanced · Bible reference: [curriculum.md §28](../../curriculum.md#28-module-18--multimodal-local-applications)

## Goal

Build local AI apps that handle images, screenshots, scanned PDFs, diagrams, and tables.

## Recommended pipeline principle (made real, not just diagrammed)

```text
Document/image
  -> OCR/layout extraction
  -> deterministic preprocessing
  -> text-based local LLM
  -> VLM only for visual reasoning gaps
```

**Do not use a VLM for everything.** `multimodal/routing.py`'s `should_use_vlm()` implements
this diagram as a real, testable decision function: a document with a usable text layer never
needs a VLM at all; only a document with no extractable text (a genuinely scanned/image-only
page) escalates.

> **Machine note:** PDF rendering, layout/table extraction, and image preprocessing are real
> libraries (`PyMuPDF`, `pdfplumber`, `Pillow`) — not LLM runtimes or model weights, so all three
> are installed and run for real, same reasoning as Module 10's `chromadb`/`lancedb`. VLM
> inference is a real model and stays honest-skip (`FakeVLM`-backed, DI pattern, "Enabling this
> for real" instructions in `vlm.py`'s docstring per this repo's heavy-dependency convention).
> Real OCR (`tesseract`, `easyocr`) needs either a system binary or downloaded model weights and
> is **not installed** - this module relies on real PDF text-layer extraction instead (see
> "Scope note" below), which needs neither.

## Scope note: OCR

Module 12 already established the precedent this module follows: real OCR needs either a system
binary (`tesseract`) or downloaded deep-learning weights (`easyocr`), and this repo's machine
constraint rules out both. Rather than a placeholder OCR function, this module uses **real PDF
text-layer extraction** (`pdfplumber`) wherever a document has one - genuinely real text, not
OCR output, but real. `scanned_receipt.pdf` (this module's second sample fixture) is a real PDF
with **no text layer at all** (an image inserted with no accompanying text), so
`should_use_vlm()` genuinely and correctly routes it away from the text path - the "OCR gap"
this module can't fill locally is made visible and honestly labeled, not hidden.

## Repo structure note

`packages/local_ai_core/multimodal/` (new) holds foundational, non-RAG-specific multimodal
primitives - the VLM protocol, image preprocessing, PDF extraction, memory-cost math, and the
VLM-routing decision - mirroring `local_ai_core/runtimes/`'s role for text models.
`packages/local_ai_rag/loaders/pdf_loader.py` joins Module 11's `markdown_loader.py` as a
sibling loader, reusing Module 11/12's existing chunker/pipeline/citation machinery unchanged
(topic 9, "Multimodal RAG" - a new *loader*, not a new RAG architecture).

## Core topics

### 1. OCR vs VLM

OCR extracts text from an image (what characters are present, where); a VLM reasons over an
entire image (layout, diagrams, charts, spatial relationships) without a separate text-extraction
step. `should_use_vlm()` operationalizes the choice: if a document already has machine-readable
text (a text layer, or OCR output if OCR were available), a text-based LLM is cheaper and more
reliable than a VLM for the same question - reserve the VLM for genuine visual reasoning gaps.

### 2. Vision-language models

`multimodal/vlm.py`'s `VisionLanguageModel` Protocol + `FakeVLM` - same DI pattern as Module 6's
`MLXRuntime`/Module 9's `SentenceTransformersEmbedder`: a real adapter's model-loading call is
lazy-imported inside a function body, tests inject a fake, real model honest-skip.

### 3. Image preprocessing

`multimodal/image_preprocessing.py` - real Pillow operations: grayscale conversion, contrast
enhancement, max-dimension resizing, and rotation by a *given* angle. Deliberately does not
implement automatic skew-*detection* (a real computer-vision problem needing OpenCV or a trained
model) - documented honestly as a gap, not silently approximated.

### 4. PDF rendering

`multimodal/pdf_extraction.py`'s `render_page_to_image()` - real `PyMuPDF` rasterization of a
PDF page to a real `PIL.Image`, the deterministic preprocessing step before any VLM call would
happen (if one is needed at all).

### 5. Layout extraction

`multimodal/pdf_extraction.py`'s `extract_layout()` - real `pdfplumber` word-level bounding
boxes (`x0, x1, top, bottom` per word), real page structure, not a guessed reading order.

### 6. Table extraction

`multimodal/pdf_extraction.py`'s `extract_tables()` - real `pdfplumber` table detection over
`sample_invoice.pdf`'s real drawn table, returning real rows/columns, not a placeholder.

### 7. Diagram understanding

Honest-skip - genuinely needs a VLM's visual reasoning. `FakeVLM`-backed in the labs;
mechanically real (the call shape, the prompt assembly), quality claim deferred to the resourced
Mac.

### 8. Screenshot question answering

Same honest-skip as §7, plus real deterministic preprocessing first: `load_image()` and region
cropping (`image_preprocessing.py`) prepare the image; only the actual visual-reasoning call is
`FakeVLM`-backed.

### 9. Multimodal RAG

`local_ai_rag/loaders/pdf_loader.py`'s `load_pdf_document()` - real per-page `Document` objects
(`doc_id = f"{pdf_stem}::page{n}"`, real extracted text) that flow through Module 11's existing
`chunk_documents()`/`NaiveRagPipeline` unchanged. Not a new RAG architecture - the exact same one
Module 11 already built, fed a new kind of real input.

### 10. Memory cost of images

`multimodal/memory_cost.py`'s `estimate_image_tokens()` - a real formula (patch-based token
estimate: `ceil(width/patch_size) * ceil(height/patch_size)`, the mechanism real VLM
architectures like ViT-based encoders use) extending Module 4's memory-math discipline to
images: a single high-resolution image can cost as many tokens as several pages of text, real
math a caller can use to budget a multimodal context window before running anything.

### 11. When not to use a VLM

`should_use_vlm()` again - the curriculum's own principle, made into one real, testable function
rather than a rule of thumb. A document with a long, coherent text layer should never reach a
VLM call; §"Real proof" in the deliverable report demonstrates both branches on real fixtures.

## Example use cases (curriculum's own list)

Receipt extraction, scanned invoice extraction, screenshot Q&A, architecture diagram
summarization, table explanation, chart interpretation, form extraction - this module's two
sample fixtures (`sample_invoice.pdf`, `scanned_receipt.pdf`) are directly modeled on the first
two.

## Hands-on labs

1. **Extract text from PDF** — `scripts/module_18/pdf_extraction_demo.py`, real text layer from
   `sample_invoice.pdf`.
2. **Extract table structure** — same script, real `extract_tables()`.
3. **Ask questions about a screenshot** — `scripts/module_18/vlm_routing_demo.py`, real image
   loading + `FakeVLM`-backed Q&A.
4. **Compare OCR+LLM vs VLM** — same script: the real text-layer path for `sample_invoice.pdf`
   vs. the real VLM-escalation path for `scanned_receipt.pdf` (which genuinely has no text
   layer to extract).
5. **Build multimodal extraction pipeline** — same script, `should_use_vlm()` routing both
   fixtures through the correct path automatically.
6. **Add page/region citations** — `scripts/module_18/multimodal_rag_demo.py`, real per-page
   citations flowing through Module 11's citation machinery unchanged.

## Deliverable

```text
datasets/multimodal/
  sample_invoice.pdf
  scanned_receipt.pdf
packages/local_ai_core/multimodal/
  vlm.py
  image_preprocessing.py
  pdf_extraction.py
  memory_cost.py
  routing.py
  tests/
packages/local_ai_rag/loaders/pdf_loader.py
scripts/module_18/
  pdf_extraction_demo.py
  vlm_routing_demo.py
  multimodal_rag_demo.py
reports/module_18_multimodal_report.md
```
