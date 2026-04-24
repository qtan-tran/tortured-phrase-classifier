"""
Evidence Report Generator
==========================
Generates structured JSON and human-readable evidence reports.
Reports are designed to support editors and integrity officers in
making defensible decisions — not to auto-reject.

Per the reviewer's guidance: the primary output is warranted, explainable
evidence, not a black-box score. Each finding links back to the specific
warrant type that supports it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tpc.pipeline import ClassificationResult


def build_report(
    result:     ClassificationResult,
    text:       str,
    metadata:   Optional[dict] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Build a structured evidence report from a ClassificationResult.

    Args:
        result:      Output from TorturedPhraseClassifier.classify()
        text:        The original manuscript text
        metadata:    Optional paper metadata (title, authors, doi, journal)
        output_path: If provided, write JSON to this path

    Returns:
        Complete evidence report as a dict (JSON-serializable).
    """
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tpc_version":  "0.1.0",
        "metadata":     metadata or {},
        "risk_assessment": {
            "risk_score":  result.risk_score,
            "risk_level":  result.risk_level,
            "summary":     result.summary,
            "layers_used": result.layers_used,
            "text_length": result.text_length,
            "guidance": _risk_guidance(result.risk_level),
        },
        "findings": {
            "confirmed_phrases":   _format_confirmed(result),
            "variant_phrases":     _format_variants(result),
            "novel_candidates":    _format_novel(result),
            "corroborated_hits":   _format_corroborated(result),
        },
        "layer_summary": {
            "exact_match": {
                "description": (
                    "Layer 1: Exact match against the warrant-based registry. "
                    "Warrant type: literary + terminological. "
                    "Precision: near-perfect (CI-gated at ≥0.95)."
                ),
                "hit_count": result.layer_counts.get("exact_match", 0),
                "hits": result.layer_hits.get("exact_match", []),
            },
            "embedding_similarity": {
                "description": (
                    "Layer 2: SPECTER embedding similarity with context coherence. "
                    "Warrant type: terminological (conceptual proximity). "
                    "Detects variants not in registry."
                ),
                "hit_count": result.layer_counts.get("embedding_similarity", 0),
                "hits": result.layer_hits.get("embedding_similarity", []),
            },
            "mlm_perplexity": {
                "description": (
                    "Layer 3: SciBERT masked language model perplexity. "
                    "Warrant type: statistical/procedural. "
                    "Detects novel phrases with no prior knowledge. "
                    "Candidates require human expert review before registry submission."
                ),
                "hit_count": result.layer_counts.get("mlm_perplexity", 0),
                "hits": result.layer_hits.get("mlm_perplexity", []),
            },
        },
        "registry_submission_candidates": _format_submission_candidates(result),
        "disclaimer": (
            "This report is produced by an automated system and should be reviewed "
            "by a qualified editor or integrity officer. The system surfaces evidence; "
            "final decisions rest with human reviewers. False positives are possible, "
            "particularly for non-native English authors using unusual but legitimate "
            "terminology. Novel candidates (Layer 3) are probabilistic and unconfirmed."
        ),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2))

    return report


def _risk_guidance(risk_level: str) -> str:
    guidance = {
        "low": (
            "No significant indicators detected. Proceed with standard review."
        ),
        "medium": (
            "Some suspicious spans detected, possibly from non-native English "
            "or unusual terminology. Manual review of flagged spans recommended "
            "before proceeding."
        ),
        "high": (
            "Multiple indicators of potential paper mill origin detected. "
            "Recommend thorough review of flagged phrases, author credentials, "
            "and citation validity before acceptance."
        ),
        "critical": (
            "Strong evidence of tortured phrases consistent with paper mill "
            "production. Recommend rejection pending investigation. Cross-check "
            "author network and image provenance independently."
        ),
    }
    return guidance.get(risk_level, "See findings for details.")


def _format_confirmed(result: ClassificationResult) -> list[dict]:
    """Exact-match hits: highest confidence, fully warranted."""
    return [
        {
            "tortured":    h["tortured"],
            "canonical":   h["canonical"],
            "domain":      h["domain"],
            "signal_id":   h["signal_id"],
            "confidence":  h["confidence"],
            "context":     h.get("context", ""),
            "explanation": h.get("explanation", ""),
            "warrant":     "confirmed (literary + terminological + statistical)",
        }
        for h in result.layer_hits.get("exact_match", [])
    ]


def _format_variants(result: ClassificationResult) -> list[dict]:
    """Embedding-similarity hits: candidate variants."""
    return [
        {
            "tortured":          h["tortured"],
            "likely_canonical":  h["canonical"],
            "domain":            h["domain"],
            "confidence":        h["confidence"],
            "sim_to_canonical":  h.get("sim_to_canonical"),
            "context_coherence": h.get("context_coherence"),
            "context":           h.get("context", ""),
            "explanation":       h.get("explanation", ""),
            "warrant":           "candidate (terminological proximity; not yet registry-confirmed)",
        }
        for h in result.layer_hits.get("embedding_similarity", [])
    ]


def _format_novel(result: ClassificationResult) -> list[dict]:
    """MLM perplexity hits: novel candidates needing human review."""
    return [
        {
            "span":            h["tortured"],
            "log_perplexity":  h.get("log_perplexity"),
            "confidence":      h["confidence"],
            "context":         h.get("context", ""),
            "explanation":     h.get("explanation", ""),
            "canonical":       None,
            "warrant":         "statistical/procedural only; human expert review required",
            "action":          "Review and if confirmed, submit to registry via GitHub PR",
        }
        for h in result.novel_spans
    ]


def _format_corroborated(result: ClassificationResult) -> list[dict]:
    """Hits flagged by multiple layers (highest confidence)."""
    return [
        h for h in result.hits
        if h.get("corroborated_by")
    ]


def _format_submission_candidates(result: ClassificationResult) -> list[dict]:
    """Novel spans suitable for registry submission (high perplexity, novel)."""
    candidates = []
    for h in result.novel_spans:
        if h.get("confidence", 0) > 0.5:
            candidates.append({
                "span":           h["tortured"],
                "log_perplexity": h.get("log_perplexity"),
                "suggested_action": (
                    "1. Verify this phrase appears in ≥3 other papers (literary warrant). "
                    "2. Identify the canonical scientific term (terminological warrant). "
                    "3. Submit a PR to signals/phrases/<domain>/<slug>.yaml with status='candidate'."
                ),
            })
    return candidates
