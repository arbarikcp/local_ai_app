"""Builds Module 18's two real sample PDF fixtures - run once to produce
the committed files under `datasets/multimodal/`. Not part of the lab
flow itself, but real, deterministic, and re-runnable if the fixtures
ever need to be regenerated.

`sample_invoice.pdf` is digital-native: real embedded text plus a real
drawn table, extractable without OCR.

`scanned_receipt.pdf` has **no text layer at all** - a real receipt-shaped
image is rendered with Pillow and inserted as a picture, with no
accompanying `insert_text()` call, so `page.get_text()` genuinely returns
an empty string. This is what makes the "should_use_vlm() correctly
escalates a real OCR gap" demonstration honest rather than staged.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

import fitz  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "datasets" / "multimodal"


def build_sample_invoice(output_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 72), "Nimbus Cloud Storage - Invoice", fontsize=16)
    page.insert_text((72, 100), "Invoice #: INV-2026-0042", fontsize=10)
    page.insert_text((72, 116), "Date: 2026-07-09", fontsize=10)
    page.insert_text((72, 132), "Bill To: Acme Corp", fontsize=10)

    header = ["Item", "Qty", "Unit Price"]
    rows = [
        ["Personal Plan (monthly)", "1", "$6.99"],
        ["Extra Storage (100GB)", "2", "$2.50"],
        ["API Overage Fee", "1", "$4.00"],
    ]
    table = [header, *rows]

    x0, y0 = 72, 170
    col_widths = [220, 60, 90]
    row_h = 20
    for r, row in enumerate(table):
        x = x0
        for c, cell in enumerate(row):
            rect = fitz.Rect(x, y0 + r * row_h, x + col_widths[c], y0 + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((x + 4, y0 + r * row_h + 14), cell, fontsize=9)
            x += col_widths[c]

    doc.save(output_path)
    doc.close()


def build_scanned_receipt(output_path: Path) -> None:
    image = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Nimbus Cloud Storage", fill="black")
    draw.text((20, 40), "Receipt", fill="black")
    draw.text((20, 80), "Personal Plan .......... $6.99", fill="black")
    draw.text((20, 100), "Total .................. $6.99", fill="black")
    draw.rectangle([10, 10, 390, 290], outline="black", width=2)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_image(fitz.Rect(0, 0, 400, 300), stream=image_bytes)
    # Deliberately no insert_text() call - this page has no real text layer,
    # the same way a scanned/photographed document wouldn't.
    doc.save(output_path)
    doc.close()


def main(argv: list[str] | None = None) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_sample_invoice(OUTPUT_DIR / "sample_invoice.pdf")
    build_scanned_receipt(OUTPUT_DIR / "scanned_receipt.pdf")
    print(f"Wrote fixtures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
