"""extract_page_fields — a thin wrapper around Project 1's reused
`ExtractionPipeline`, applied to one page's text (ARCHITECTURE.md's
high-level diagram, `TEXT_LLM` route). Confirmed by PROPOSAL.md's survey:
`ExtractionPipeline` is fully generic with no dependency on Project 1's own
schemas, so this file adds no new extraction machinery, only the
`DocumentFieldExtraction` schema wiring.
"""

from __future__ import annotations

from local_ai_core.extraction.pipeline import ExtractionPipeline, ExtractionResult

from doc_schemas import DocumentFieldExtraction


async def extract_page_fields(
    text: str,
    *,
    pipeline: ExtractionPipeline[DocumentFieldExtraction],
    model: str,
) -> ExtractionResult[DocumentFieldExtraction]:
    return await pipeline.run(text, model)
