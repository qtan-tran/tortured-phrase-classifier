"""
Layer 3: MLM Perplexity Detector
==================================
SciBERT masked language model for detecting novel tortured phrases
with no prior knowledge of specific phrase pairs.

KO rationale: SciBERT trained on 1.14M scientific papers implicitly
encodes the literary warrant of legitimate science — it learns what the
scientific community's discourse sanctions. A span with anomalously high
perplexity is one that the community's warranted language would not produce.

This operationalizes statistical/procedural warrant as an extension of
Beghtol's (1986) warrant theory into computational settings: the model's
training corpus represents the cumulative literary warrant of a domain,
and deviation from it signals a possible terminological violation.

Critically, this layer requires NO prior knowledge of any specific tortured
phrase. It detects novelty by deviation from corpus norms — enabling the
system to catch emerging paper mill vocabulary before it is hand-curated.

Novel phrase candidates detected here feed the community review queue,
where they can be assessed for literary, terminological, and statistical
warrant before entering the registry.

Model: allenai/scibert_scivocab_uncased
  - Trained on 1.14M scientific papers (Beltagy et al. 2019)
  - Domain-appropriate vocabulary (scientific tokenization)
  - Superior to general-domain BERT for scientific prose perplexity

Warrant type: Statistical/procedural (novel extension of warrant theory)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_TOKENIZER = None
_MODEL     = None


def _load_model(model_name: str):
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        logger.info("Loading MLM model: %s", model_name)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL     = AutoModelForMaskedLM.from_pretrained(model_name)
        _MODEL.eval()
        logger.info("MLM model loaded")
    return _TOKENIZER, _MODEL


class PerplexityDetector:
    """
    Layer 3: Novel tortured phrase detection via masked language model perplexity.

    Slides a window over the manuscript text and flags spans where the
    average token log-probability under SciBERT is anomalously high —
    indicating text that legitimate scientific discourse would not produce.
    """

    DEFAULT_MODEL = "allenai/scibert_scivocab_uncased"

    def __init__(
        self,
        model_name:          str   = DEFAULT_MODEL,
        window_tokens:       int   = 6,     # tokens per span
        stride_tokens:       int   = 3,     # sliding step
        perplexity_threshold: float = 4.5,  # log-perplexity threshold
        max_text_tokens:     int   = 512,   # truncate long texts
    ):
        """
        Args:
            model_name:           HuggingFace model identifier
            window_tokens:        Number of tokens per sliding window span
            stride_tokens:        Stride of sliding window
            perplexity_threshold: Spans above this log-perplexity are flagged.
                                  Calibrated on clean PubMed corpus (see notebooks/).
            max_text_tokens:      Maximum tokens to process (performance)
        """
        self.model_name           = model_name
        self.window_tokens        = window_tokens
        self.stride_tokens        = stride_tokens
        self.perplexity_threshold = perplexity_threshold
        self.max_text_tokens      = max_text_tokens

    def detect(self, text: str) -> list[dict]:
        """
        Detect anomalously high-perplexity spans in text.

        Args:
            text: Manuscript text to scan

        Returns:
            List of suspicious span dicts. 'canonical' is None at this
            layer — these are candidates for human expert review.
        """
        import torch
        tokenizer, model = _load_model(self.model_name)

        tokens = tokenizer.tokenize(text.lower())
        if len(tokens) > self.max_text_tokens:
            logger.debug(
                "Layer 3: text truncated from %d to %d tokens",
                len(tokens), self.max_text_tokens
            )
            tokens = tokens[:self.max_text_tokens]

        if len(tokens) < self.window_tokens:
            return []

        suspicious_spans = []
        i = 0

        while i <= len(tokens) - self.window_tokens:
            span_tokens = tokens[i:i + self.window_tokens]
            span_text   = tokenizer.convert_tokens_to_string(span_tokens)
            lp          = self._span_log_perplexity(
                span_tokens, tokenizer, model, torch
            )

            if lp > self.perplexity_threshold:
                # Confidence: normalized excess above threshold
                confidence = min(1.0, (lp - self.perplexity_threshold) / 5.0)

                suspicious_spans.append({
                    "signal_id":     None,      # unknown — needs registry lookup
                    "tortured":      span_text,
                    "canonical":     None,       # unknown — needs human review
                    "domain":        None,
                    "layer":         "mlm_perplexity",
                    "log_perplexity": round(lp, 3),
                    "threshold":     self.perplexity_threshold,
                    "confidence":    round(confidence, 3),
                    "position":      {"token_start": i, "token_end": i + self.window_tokens},
                    "context":       span_text,
                    "canonical":     None,
                    "explanation": (
                        f"Span '{span_text}' has log-perplexity={lp:.3f} "
                        f"(threshold={self.perplexity_threshold}). "
                        f"SciBERT assigns this span anomalously low probability "
                        f"under the literary warrant of legitimate scientific prose. "
                        f"Candidate for human expert review and registry submission."
                    ),
                })

            i += self.stride_tokens

        # Merge overlapping spans (keep highest perplexity per region)
        return self._merge_overlapping(suspicious_spans)

    @staticmethod
    def _span_log_perplexity(
        span_tokens: list[str],
        tokenizer,
        model,
        torch,
    ) -> float:
        """
        Compute average token log-perplexity for a span by masking each
        token in turn and measuring how surprising it is to the MLM.

        This implements the per-token pseudo-log-likelihood approximation
        (Salazar et al. 2020) adapted for span-level anomaly detection.
        """
        scores = []

        for mask_pos, target_token in enumerate(span_tokens):
            masked = list(span_tokens)
            masked[mask_pos] = tokenizer.mask_token

            input_ids = tokenizer.convert_tokens_to_ids(masked)
            input_tensor = torch.tensor([[
                tokenizer.cls_token_id,
                *input_ids,
                tokenizer.sep_token_id,
            ]])

            with torch.no_grad():
                logits = model(input_tensor).logits

            # Position offset by 1 for [CLS]
            token_logits = logits[0, mask_pos + 1, :]
            probs        = torch.softmax(token_logits, dim=-1)

            target_id = tokenizer.convert_tokens_to_ids([target_token])[0]
            prob      = float(probs[target_id].item())
            scores.append(-np.log(prob + 1e-9))

        return float(np.mean(scores))

    @staticmethod
    def _merge_overlapping(spans: list[dict]) -> list[dict]:
        """
        Merge overlapping token spans, keeping the highest-perplexity hit
        per overlapping region. Reduces redundant reporting.
        """
        if not spans:
            return []

        # Sort by token start
        spans = sorted(spans, key=lambda s: s["position"]["token_start"])
        merged = [spans[0]]

        for span in spans[1:]:
            last = merged[-1]
            # Check overlap
            if span["position"]["token_start"] < last["position"]["token_end"]:
                # Keep higher perplexity
                if span["log_perplexity"] > last["log_perplexity"]:
                    merged[-1] = span
            else:
                merged.append(span)

        return merged

    def calibrate_threshold(
        self,
        clean_texts: list[str],
        percentile: float = 95.0,
    ) -> float:
        """
        Calibrate the perplexity threshold on a clean corpus.
        Sets threshold at the given percentile of clean-corpus span perplexities,
        so that (100 - percentile)% of clean spans are flagged (false positive rate).

        Usage: run this on your clean PubMed corpus to set domain-appropriate threshold.
        """
        import torch
        tokenizer, model = _load_model(self.model_name)
        all_perplexities = []

        for text in clean_texts:
            tokens = tokenizer.tokenize(text.lower())[:self.max_text_tokens]
            for i in range(0, len(tokens) - self.window_tokens, self.stride_tokens):
                span = tokens[i:i + self.window_tokens]
                lp = self._span_log_perplexity(span, tokenizer, model, torch)
                all_perplexities.append(lp)

        if not all_perplexities:
            return self.perplexity_threshold

        threshold = float(np.percentile(all_perplexities, percentile))
        logger.info(
            "Calibrated threshold (p%.0f on %d clean texts): %.3f",
            percentile, len(clean_texts), threshold
        )
        return threshold
