"""Builds this project's own real, committed multi-page PDF fixture
(ARCHITECTURE.md "The fixture") - run once to produce the committed file
under `datasets/multimodal/`. Same real construction Module 18's
`scripts/module_18/build_fixtures.py` established: `fitz`/PyMuPDF for real
embedded text, Pillow for a real embedded scanned image with deliberately
no text layer.

Both existing Module 18 fixtures (`sample_invoice.pdf`, `scanned_receipt.pdf`)
are single-page - too small to exercise page-level citations or a real
per-document OCR+LLM-vs-VLM comparison within one document. This one is
three pages: two digital-native text pages, one genuinely scanned
image-only page.

Lives in its own `datasets/multimodal/project_04/` subdirectory, not
alongside the Module 18 fixtures directly: `scripts/module_18/
multimodal_rag_demo.py` globs every `*.pdf` directly in `datasets/
multimodal/` (non-recursive) and hardcodes its own two-fixture assertions
- confirmed by a real test failure this fixture caused there before being
moved into its own subdirectory, which the non-recursive glob doesn't see.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

import fitz  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "datasets" / "multimodal" / "project_04" / "multi_page_form.pdf"


def build_multi_page_form(output_path: Path) -> None:
    doc = fitz.open()

    # Page 1: digital-native text - applicant details.
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Nimbus Cloud Storage - Account Closure Request", fontsize=16)
    page1.insert_text((72, 110), "Applicant Name: Jordan Rivera", fontsize=11)
    page1.insert_text((72, 130), "Account ID: NCS-88213", fontsize=11)
    page1.insert_text((72, 150), "Request Date: 2026-06-15", fontsize=11)

    # Page 2: digital-native text - reason and refund amount.
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Reason for Closure", fontsize=14)
    page2.insert_text(
        (72, 100),
        "The applicant no longer requires cloud storage services and has migrated",
        fontsize=11,
    )
    page2.insert_text((72, 118), "all data to a self-hosted solution.", fontsize=11)
    page2.insert_text((72, 150), "Refund Amount Owed: $42.50", fontsize=11)

    # Page 3: scanned, no text layer - a rendered signature/ID verification
    # image, deliberately no insert_text() call.
    image = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Signature and ID Verification", fill="black")
    draw.text((20, 60), "[signature image]", fill="black")
    draw.rectangle([10, 10, 390, 290], outline="black", width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    page3 = doc.new_page(width=400, height=300)
    page3.insert_image(fitz.Rect(0, 0, 400, 300), stream=image_bytes)

    doc.save(output_path)
    doc.close()


def main(argv: list[str] | None = None) -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_multi_page_form(OUTPUT_PATH)
    print(f"Wrote fixture to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
