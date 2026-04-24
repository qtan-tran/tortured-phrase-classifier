"""
Layer 2: Embedding Similarity Detector
=======================================
SPECTER-based detection of variant and paraphrase tortured phrases.

KO rationale: This layer encodes the conceptual network within which terms
acquire meaning — directly aligned with Wüster's principle that terms live
in concept systems, not in isolation (ISO 704 §5.1). A phrase that is
semantically proximate to a known canonical term but incoherent with its
local textual context violates the domain-coherence requirement of
terminological warrant.

The dual signal (similarity to canonical × incoherence with context) is the
key insight: a legitimate domain term is coherent with scientific prose;
a tortured replacement is semantically proximate to the real term but
jarring in its textual environment.

Model choice: SPECTER (Lo et al. 2020) — trained on scientific paper
title/abstract pairs and citation graphs. Superior to generic BERT for
scientific term proximity because it encodes domain-specific conceptual
relationships.

Warrant type: Terminological (conceptual proximity + domain coherence)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tpc.registry.loader import Signal, load_registry

logger = logging.getLogger(__name__)

# Lazy import to avoid slow startup when only Layer 1 is needed
_SENTENCE_TRANSFORMER = None


def _get_model(model_name: str):
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _SENTENCE_TRANSFORMER = SentenceTransformer(model_name)
    return _SENTENCE_TRANSFORMER


class EmbeddingDetector:
    """
    Layer 2: Semantic similarity detection using SPECTER embeddings.

    Detects tortured phrases that are paraphrases or variants of known
    canonical terms but that do not appear verbatim in the registry.
    """

    DEFAULT_MODEL = "allenai-specter"

    def __init__(
        self,
        signals:          Optional[list[Signal]] = None,
        model_name:       str = DEFAULT_MODEL,
        sim_threshold:    float = 0.82,   # min similarity to a known canonical
        suspicion_threshold: float = 0.30, # min suspicion score to report
        ngram_sizes:      tuple[int, ...] = (2, 3, 4, 5),
        context_window:   int = 40,        # words each side for coherence
    ):
        self.signals           = signals or load_registry()
        self.model_name        = model_name
        self.sim_threshold     = sim_threshold
        self.suspicion_threshold = suspicion_threshold
        self.ngram_sizes       = ngram_sizes
        self.context_window    = context_window

        # Pre-compute canonical embeddings
        self.canonicals  = [s.canonical for s in self.signals]
        self.domains     = [s.domain    for s in self.signals]
        self.signal_ids  = [s.id        for s in self.signals]

        model = _get_model(model_name)
        logger.info("Layer 2 (Embedding): encoding %d canonical terms", len(self.canonicals))
        self.canonical_embeddings = model.encode(
            self.canonicals,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )

    def detect(self, text: str) -> list[dict]:
        """
        Detect variant tortured phrases via embedding similarity.

        Args:
            text: Full manuscript text

        Returns:
            List of suspicious span dicts with similarity scores.
        """
        model = _get_model(self.model_name)
        words = text.lower().split()

        if len(words) < min(self.ngram_sizes):
            return []

        ngrams, positions = self._extract_ngrams_with_positions(words)
        if not ngrams:
            return []

        # Encode all ngrams in one batch
        ngram_embeddings = model.encode(
            ngrams,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
        )

        # Similarity matrix: (n_ngrams, n_canonicals)
        sim_matrix = cosine_similarity(ngram_embeddings, self.canonical_embeddings)

        hits = []
        seen = set()

        for i, (ngram, (start, end)) in enumerate(zip(ngrams, positions)):
            # Skip if already caught by exact match (avoid redundancy)
            if ngram in seen:
                continue

            max_sim_idx = int(sim_matrix[i].argmax())
            max_sim     = float(sim_matrix[i][max_sim_idx])

            if max_sim < self.sim_threshold:
                continue

            # Context coherence: extract surrounding words and compare
            ctx_start = max(0, start - self.context_window)
            ctx_end   = min(len(words), end + self.context_window)
            context   = " ".join(words[ctx_start:ctx_end])

            if not context.strip():
                continue

            ctx_embedding = model.encode(
                [context],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            context_coherence = float(
                cosine_similarity([ngram_embeddings[i]], [ctx_embedding])[0][0]
            )

            # Suspicion = high similarity to canonical × low coherence with context
            suspicion = max_sim * (1.0 - context_coherence)

            if suspicion < self.suspicion_threshold:
                continue

            seen.add(ngram)
            canonical  = self.canonicals[max_sim_idx]
            domain     = self.domains[max_sim_idx]
            signal_id  = self.signal_ids[max_sim_idx]

            hits.append({
                "signal_id":         signal_id,
                "tortured":          ngram,
                "canonical":         canonical,
                "domain":            domain,
                "layer":             "embedding_similarity",
                "confidence":        round(suspicion, 3),
                "sim_to_canonical":  round(max_sim, 3),
                "context_coherence": round(context_coherence, 3),
                "position":          {"word_start": start, "word_end": end},
                "context":           context,
                "explanation": (
                    f"'{ngram}' is semantically similar to canonical term "
                    f"'{canonical}' (sim={max_sim:.3f}) but incoherent with "
                    f"surrounding text (coherence={context_coherence:.3f}). "
                    f"Suspicion score: {suspicion:.3f}. "
                    f"Domain: {domain}."
                ),
            })

        # Sort by suspicion descending
        return sorted(hits, key=lambda h: h["confidence"], reverse=True)

    def _extract_ngrams_with_positions(
        self, words: list[str]
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Extract all n-grams with their word-level positions."""
        ngrams, positions = [], []
        for n in self.ngram_sizes:
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.append(ngram)
                positions.append((i, i + n))
        return ngrams, positions
