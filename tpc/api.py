"""
FastAPI Pre-Publication Screening API
======================================
REST endpoint for integrating TPC into manuscript submission systems
(OJS, Editorial Manager, ScholarOne, etc.).

Endpoints:
  POST /screen        — screen a manuscript PDF or plain text
  GET  /registry      — registry statistics
  GET  /health        — health check
  GET  /docs          — auto-generated OpenAPI docs (FastAPI default)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tortured Phrase Classifier API",
    description=(
        "Pre-publication screening API for detecting tortured phrases "
        "as terminological violations in scientific manuscripts. "
        "Grounded in warrant theory (Beghtol 1986) and Wüsterian terminology science."
    ),
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Lazy-initialized pipeline
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from tpc.pipeline import TorturedPhraseClassifier
        # Default: all three layers for pre-publication screening
        _pipeline = TorturedPhraseClassifier(layers=("exact", "embedding", "mlm"))
        logger.info("Pipeline initialized")
    return _pipeline


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/registry")
def registry_stats():
    """Return summary statistics of the signal registry."""
    from tpc.registry.loader import load_registry, registry_summary
    signals = load_registry(status_filter=("confirmed", "candidate", "deprecated"))
    return registry_summary(signals)


@app.post("/screen")
async def screen_manuscript(
    file:   Optional[UploadFile] = File(None),
    text:   Optional[str]        = Form(None),
    layers: str                  = Form("exact,embedding,mlm"),
    title:  Optional[str]        = Form(None),
    doi:    Optional[str]        = Form(None),
):
    """
    Screen a manuscript for tortured phrases.

    Accepts either:
      - A PDF file upload (multipart/form-data, field: 'file')
      - Plain text (multipart/form-data, field: 'text')

    Returns a structured evidence report with risk score, all hits,
    and per-layer breakdowns.
    """
    if file is None and text is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'file' (PDF) or 'text' (plain text).",
        )

    # Extract text
    if file is not None:
        content = await file.read()
        manuscript_text = _extract_pdf_text(content)
        if not manuscript_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. "
                       "Ensure the PDF contains selectable text (not scanned).",
            )
    else:
        manuscript_text = text or ""

    if len(manuscript_text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Text too short for meaningful screening (minimum 50 characters).",
        )

    # Determine which layers to activate
    active_layers = tuple(l.strip() for l in layers.split(",") if l.strip())
    valid_layers  = {"exact", "embedding", "mlm"}
    invalid       = set(active_layers) - valid_layers
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid layers: {invalid}. Valid: {valid_layers}",
        )

    try:
        pipeline = _get_pipeline()
        result   = pipeline.classify(manuscript_text)
    except Exception as e:
        logger.exception("Classification error")
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    from tpc.report import build_report
    report = build_report(
        result=result,
        text=manuscript_text,
        metadata={"title": title, "doi": doi},
    )

    return report


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    try:
        import fitz  # pymupdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc   = fitz.open(tmp_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        Path(tmp_path).unlink(missing_ok=True)
        return "\n".join(pages)
    except Exception as e:
        logger.error("PDF extraction failed: %s", e)
        return ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
