"""Labs 1-2 - extract real text and a real table from a real PDF. Runs
for real, no live model needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.multimodal.pdf_extraction import (  # noqa: E402
    extract_layout,
    extract_tables,
    extract_text_layer,
    render_page_to_image,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_INVOICE = REPO_ROOT / "datasets" / "multimodal" / "sample_invoice.pdf"


def run_lab() -> dict:
    text = extract_text_layer(SAMPLE_INVOICE, 0)
    tables = extract_tables(SAMPLE_INVOICE, 0)
    words = extract_layout(SAMPLE_INVOICE, 0)
    image = render_page_to_image(SAMPLE_INVOICE, 0, dpi=150)

    return {
        "extracted_text": text,
        "table": tables[0] if tables else None,
        "word_count": len(words),
        "rendered_image_size": image.size,
    }


def result_to_markdown(result: dict) -> str:
    lines = ["# Labs 1-2 - real text and table extraction from a PDF\n"]
    lines.append(f"## Extracted text\n```\n{result['extracted_text']}\n```\n")
    lines.append("## Extracted table\n")
    for row in result["table"]:
        lines.append(f"- {row}")
    lines.append(f"\n- Word-level layout entries: {result['word_count']}")
    lines.append(f"- Rendered page image size: {result['rendered_image_size']}\n")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
