"""
Layer 1: Exact Match Detector
==============================
Aho-Corasick multi-pattern string matching against the warrant-based registry.

KO rationale: This layer represents controlled vocabulary in its strictest form.
Only phrases that have passed all three warrant gates (literary, terminological,
statistical) appear here. This embodies ISO 25964's principle that a controlled
vocabulary term must be authorized before deployment.

Warrant type: Literary + Terminological (both required for registry inclusion)
Precision: near-perfect by design (CI gate ensures precision ≥ 0.95)
Recall: bounded by registry completeness (addressed by Layers 2 & 3)
"""

from __future__ import annotations

import logging
from typing import Optional

import ahocorasick

from tpc.registry.loader import Signal, load_registry

logger = logging.getLogger(__name__)


class ExactMatchDetector:
    """
    Layer 1: Exact-match detection using Aho-Corasick automaton.
    Scans the full text against all registered tortured phrases in O(n) time.
    """

    def __init__(
        self,
        signals:        Optional[list[Signal]] = None,
        status_filter:  tuple[str, ...] = ("confirmed",),
        domain_filter:  Optional[list[str]] = None,
    ):
        """
        Args:
            signals:       Pre-loaded signals (if None, loads from registry)
            status_filter: Only use signals with these statuses.
                           Default: confirmed only — highest precision.
            domain_filter: Restrict to specific domains (None = all)
        """
        self.signals = signals or load_registry(
            status_filter=status_filter,
            domain_filter=domain_filter,
        )
        self._automaton = self._build_automaton()
        logger.info(
            "Layer 1 (ExactMatch): loaded %d signals covering %d term strings",
            len(self.signals),
            sum(len(s.all_terms) for s in self.signals),
        )

    def _build_automaton(self) -> ahocorasick.Automaton:
        """Build Aho-Corasick automaton from all signal terms."""
        A = ahocorasick.Automaton()
        for sig in self.signals:
            for term in sig.all_terms:
                term_lower = term.lower()
                # Store full signal metadata at each pattern
                A.add_word(term_lower, {
                    "signal_id":  sig.id,
                    "tortured":   term,
                    "canonical":  sig.canonical,
                    "domain":     sig.domain,
                    "status":     sig.status.value,
                    "precision":  sig.warrant.precision_on_clean,
                    "recall":     sig.warrant.recall_on_retracted,
                    "sightings":  sig.prevalence_retracted,
                    "tool":       sig.paraphrase_tool_origin,
                })
        A.make_automaton()
        return A

    def detect(self, text: str) -> list[dict]:
        """
        Scan text for all registered tortured phrases.

        Args:
            text: Full manuscript text (or section)

        Returns:
            List of hit dicts with signal metadata, position, and context snippet.
        """
        text_lower = text.lower()
        hits: list[dict] = []
        seen_terms: set[str] = set()

        for end_idx, payload in self._automaton.iter(text_lower):
            term    = payload["tortured"].lower()
            if term in seen_terms:
                continue
            seen_terms.add(term)

            start_idx = end_idx - len(term) + 1
            context   = self._extract_context(text, start_idx, end_idx)

            hits.append({
                **payload,
                "layer":      "exact_match",
                "confidence": 1.0,
                "position":   {"start": start_idx, "end": end_idx + 1},
                "context":    context,
                "explanation": (
                    f"'{payload['tortured']}' is a known tortured phrase "
                    f"(canonical: '{payload['canonical']}'). "
                    f"Attested in {payload['sightings']} retracted papers. "
                    f"Warrant: confirmed (precision={payload['precision']:.3f}, "
                    f"recall={payload['recall']:.3f})."
                ),
            })

        return sorted(hits, key=lambda h: h["position"]["start"])

    @staticmethod
    def _extract_context(text: str, start: int, end: int,
                          window: int = 80) -> str:
        """Extract surrounding context for evidence reporting."""
        ctx_start = max(0, start - window)
        ctx_end   = min(len(text), end + window + 1)
        snippet   = text[ctx_start:ctx_end].replace("\n", " ").strip()
        if ctx_start > 0:
            snippet = "…" + snippet
        if ctx_end < len(text):
            snippet = snippet + "…"
        return snippet

    def reload(self) -> None:
        """Reload registry from disk (for long-running services)."""
        self.signals = load_registry()
        self._automaton = self._build_automaton()
        logger.info("Layer 1: registry reloaded")
