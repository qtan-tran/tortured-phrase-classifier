"""
Pipeline Orchestrator
=====================
Combines the three detection layers into a unified classification pipeline.
Implements deduplication, confidence aggregation, and structured evidence reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from tpc.layers.exact_match import ExactMatchDetector
from tpc.layers.embedding import EmbeddingDetector
from tpc.layers.mlm_perplexity import PerplexityDetector

logger = logging.getLogger(__name__)

# Per-layer weights for risk score aggregation
# Exact match has highest weight (fully warranted); MLM lowest (probabilistic)
LAYER_WEIGHTS: dict[str, float] = {
    "exact_match":          1.0,
    "embedding_similarity": 0.7,
    "mlm_perplexity":       0.5,
}

# Normalization denominator (5 weighted hits → risk score 1.0)
SCORE_NORMALIZATION: float = 5.0


@dataclass
class ClassificationResult:
    """Structured result from the full three-layer pipeline."""

    # Core outputs
    risk_score:     float                  # 0.0 – 1.0
    risk_level:     str                    # low | medium | high | critical
    summary:        str
    hits:           list[dict] = field(default_factory=list)
    novel_spans:    list[dict] = field(default_factory=list)

    # Per-layer breakdowns
    layer_hits:     dict[str, list[dict]] = field(default_factory=dict)
    layer_counts:   dict[str, int]        = field(default_factory=dict)

    # Metadata
    text_length:    int = 0
    layers_used:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "risk_score":   self.risk_score,
            "risk_level":   self.risk_level,
            "summary":      self.summary,
            "hits":         self.hits,
            "novel_spans":  self.novel_spans,
            "layer_hits":   self.layer_hits,
            "layer_counts": self.layer_counts,
            "text_length":  self.text_length,
            "layers_used":  self.layers_used,
        }


class TorturedPhraseClassifier:
    """
    Three-layer tortured phrase classification pipeline.

    Layer 1 (ExactMatch):   fast, zero-FP on known phrases, O(n)
    Layer 2 (Embedding):    catches variants and paraphrases of known phrases
    Layer 3 (Perplexity):   catches entirely novel phrases (no prior knowledge)

    Each layer is independently configurable and the results are merged
    with deduplication prioritizing higher-precision layers.
    """

    def __init__(
        self,
        layers:          tuple[str, ...] = ("exact", "embedding", "mlm"),
        exact_kwargs:    Optional[dict]  = None,
        embedding_kwargs: Optional[dict] = None,
        mlm_kwargs:      Optional[dict]  = None,
    ):
        """
        Args:
            layers: Which layers to activate. Subset of ("exact", "embedding", "mlm").
                    For fast/lightweight usage, use ("exact",) only.
                    For pre-publication screening, use all three.
            exact_kwargs:     Kwargs passed to ExactMatchDetector
            embedding_kwargs: Kwargs passed to EmbeddingDetector
            mlm_kwargs:       Kwargs passed to PerplexityDetector
        """
        self.layers_used: list[str] = []
        self._detectors: dict[str, object] = {}

        if "exact" in layers:
            logger.info("Initializing Layer 1 (Exact Match)")
            self._detectors["exact"] = ExactMatchDetector(**(exact_kwargs or {}))
            self.layers_used.append("exact_match")

        if "embedding" in layers:
            logger.info("Initializing Layer 2 (Embedding Similarity)")
            self._detectors["embedding"] = EmbeddingDetector(**(embedding_kwargs or {}))
            self.layers_used.append("embedding_similarity")

        if "mlm" in layers:
            logger.info("Initializing Layer 3 (MLM Perplexity)")
            self._detectors["mlm"] = PerplexityDetector(**(mlm_kwargs or {}))
            self.layers_used.append("mlm_perplexity")

        logger.info("Pipeline ready: layers = %s", self.layers_used)

    def classify(self, text: str) -> ClassificationResult:
        """
        Run all active layers on the input text.

        Args:
            text: Manuscript text (extracted from PDF or provided directly)

        Returns:
            ClassificationResult with risk score, all hits, and per-layer breakdown.
        """
        # Collect per-layer hits
        layer_hits: dict[str, list[dict]] = {}

        if "exact" in self._detectors:
            layer_hits["exact_match"] = self._detectors["exact"].detect(text)

        if "embedding" in self._detectors:
            layer_hits["embedding_similarity"] = self._detectors["embedding"].detect(text)

        if "mlm" in self._detectors:
            layer_hits["mlm_perplexity"] = self._detectors["mlm"].detect(text)

        # Deduplicate across layers (exact_match takes precedence)
        merged_hits = self._deduplicate(layer_hits)

        # Novel spans: MLM hits with no canonical identified
        novel_spans = [
            h for h in layer_hits.get("mlm_perplexity", [])
            if h.get("canonical") is None
        ]

        risk_score = self._aggregate_score(merged_hits)
        risk_level = self._risk_level(risk_score)
        summary    = self._summarize(merged_hits, novel_spans, layer_hits)

        return ClassificationResult(
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
            hits=merged_hits,
            novel_spans=novel_spans,
            layer_hits=layer_hits,
            layer_counts={k: len(v) for k, v in layer_hits.items()},
            text_length=len(text),
            layers_used=self.layers_used,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(layer_hits: dict[str, list[dict]]) -> list[dict]:
        """
        Merge hits across layers, deduplicating by tortured phrase string.
        Priority: exact_match > embedding_similarity > mlm_perplexity.
        If a phrase is caught by multiple layers, keep the higher-precision hit
        and annotate it as corroborated.
        """
        seen:   set[str] = set()
        merged: list[dict] = []

        for layer in ("exact_match", "embedding_similarity", "mlm_perplexity"):
            for hit in layer_hits.get(layer, []):
                key = (hit.get("tortured") or hit.get("span", "")).lower().strip()
                if not key:
                    continue
                if key not in seen:
                    seen.add(key)
                    merged.append(hit)
                else:
                    # Annotate existing hit as corroborated by multiple layers
                    for existing in merged:
                        existing_key = (
                            existing.get("tortured") or existing.get("span", "")
                        ).lower().strip()
                        if existing_key == key:
                            existing["corroborated_by"] = (
                                existing.get("corroborated_by", []) + [layer]
                            )
                            break

        return merged

    @staticmethod
    def _aggregate_score(hits: list[dict]) -> float:
        """
        Weighted risk score: sum of (layer_weight × confidence) / normalization.
        Capped at 1.0.
        """
        if not hits:
            return 0.0
        raw = sum(
            LAYER_WEIGHTS.get(h["layer"], 0.5) * h.get("confidence", 0.5)
            for h in hits
        )
        return round(min(1.0, raw / SCORE_NORMALIZATION), 3)

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 0.2:  return "low"
        if score < 0.5:  return "medium"
        if score < 0.8:  return "high"
        return "critical"

    @staticmethod
    def _summarize(
        hits: list[dict],
        novel_spans: list[dict],
        layer_hits: dict[str, list[dict]],
    ) -> str:
        n_total   = len(hits)
        n_exact   = len(layer_hits.get("exact_match", []))
        n_emb     = len(layer_hits.get("embedding_similarity", []))
        n_mlm     = len(layer_hits.get("mlm_perplexity", []))
        n_novel   = len(novel_spans)

        parts = [f"{n_total} suspicious span(s) detected"]
        if n_exact:   parts.append(f"{n_exact} exact-match (confirmed registry)")
        if n_emb:     parts.append(f"{n_emb} embedding-similarity (variant)")
        if n_mlm:     parts.append(f"{n_mlm} perplexity-anomaly (potentially novel)")
        if n_novel:   parts.append(
            f"{n_novel} novel candidate(s) flagged for registry review"
        )

        return "; ".join(parts) + "."
