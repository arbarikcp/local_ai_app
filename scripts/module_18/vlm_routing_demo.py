"""Labs 3-5 - ask a question about a screenshot (FakeVLM-backed), compare
the real OCR+LLM-equivalent path against the real VLM-escalation path
across both sample fixtures, and build the full multimodal extraction
pipeline: `should_use_vlm()` routing each document to the correct path
automatically. Runs for real except the VLM call itself (`FakeVLM`).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.multimodal.pdf_extraction import extract_text_layer, render_page_to_image  # noqa: E402
from local_ai_core.multimodal.routing import MultimodalRoute, should_use_vlm  # noqa: E402
from local_ai_core.multimodal.vlm import FakeVLM  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_INVOICE = REPO_ROOT / "datasets" / "multimodal" / "sample_invoice.pdf"
SCANNED_RECEIPT = REPO_ROOT / "datasets" / "multimodal" / "scanned_receipt.pdf"


async def answer_via_text_llm(text: str, question: str, runtime) -> str:
    prompt = f"Context:\n{text}\n\nQuestion:\n{question}\n\nAnswer:"
    response = await runtime.generate(LLMRequest(model="fake-model", prompt=prompt))
    return response.text


async def answer_via_vlm(image, question: str, vlm) -> str:
    return await vlm.describe(image, question)


async def process_document(pdf_path: Path, question: str, runtime, vlm) -> dict:
    text = extract_text_layer(pdf_path, 0)
    decision = should_use_vlm(text)

    if decision.route == MultimodalRoute.TEXT_LLM:
        answer = await answer_via_text_llm(text, question, runtime)
        method = "text_llm"
    else:
        image = render_page_to_image(pdf_path, 0)
        answer = await answer_via_vlm(image, question, vlm)
        method = "vlm"

    return {
        "document": pdf_path.name,
        "route": decision.route.value,
        "reason": decision.reason,
        "method_used": method,
        "answer": answer,
    }


async def run_lab() -> dict:
    text_runtime = FakeRuntime(default_response="The invoice total is $6.99 for the Personal Plan.")
    vlm = FakeVLM(default_response="The receipt shows a Personal Plan charge of $6.99.")

    # Lab 3: a real screenshot/image question, VLM-backed.
    screenshot_image = render_page_to_image(SCANNED_RECEIPT, 0)
    screenshot_answer = await vlm.describe(screenshot_image, "What is the total on this receipt?")

    # Labs 4-5: the full pipeline, routing each real document automatically.
    invoice_result = await process_document(SAMPLE_INVOICE, "What is the invoice total?", text_runtime, vlm)
    receipt_result = await process_document(SCANNED_RECEIPT, "What is the total on this receipt?", text_runtime, vlm)

    return {
        "screenshot_answer": screenshot_answer,
        "invoice_result": invoice_result,
        "receipt_result": receipt_result,
    }


def result_to_markdown(result: dict) -> str:
    invoice = result["invoice_result"]
    receipt = result["receipt_result"]
    return (
        "# Labs 3-5 - screenshot Q&A, OCR+LLM vs VLM, full routing pipeline\n\n"
        f"## Lab 3: screenshot question answering (VLM-backed)\n{result['screenshot_answer']}\n\n"
        "## Labs 4-5: automatic routing per document\n"
        f"- {invoice['document']}: routed to **{invoice['route']}** ({invoice['reason']})\n"
        f"  -> {invoice['answer']}\n"
        f"- {receipt['document']}: routed to **{receipt['route']}** ({receipt['reason']})\n"
        f"  -> {receipt['answer']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
