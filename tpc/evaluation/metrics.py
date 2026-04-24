"""
Evaluation: Metrics and Benchmark
===================================
Computes per-layer and combined precision, recall, F1, and AUROC.
Implements the ablation study design for the Knowledge Organization paper.

Evaluation design (Section 5.4 of paper):
  - Positive corpus:  PubMed retracted papers
  - Negative corpus:  PubMed clean papers (domain-matched)
  - Synthetic corpus: WordNet + back-translation
  - Novel corpus:     2024-25 papers not in any existing list

Metrics:
  - Precision, Recall, F1 (standard binary classification)
  - AUROC (threshold-independent performance)
  - Novel phrase detection rate (Layer 3 specific)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LayerMetrics:
    """Evaluation metrics for a single detection layer."""
    layer:          str
    corpus:         str
    true_positives:  int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives:  int = 0
    scores:          list[float] = field(default_factory=list)
    labels:          list[int]   = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0

    @property
    def auroc(self) -> float:
        if len(set(self.labels)) < 2:
            return 0.5
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(self.labels, self.scores))

    def to_dict(self) -> dict:
        return {
            "layer":      self.layer,
            "corpus":     self.corpus,
            "precision":  round(self.precision, 4),
            "recall":     round(self.recall, 4),
            "f1":         round(self.f1, 4),
            "auroc":      round(self.auroc, 4),
            "tp":         self.true_positives,
            "fp":         self.false_positives,
            "fn":         self.false_negatives,
            "tn":         self.true_negatives,
        }


def evaluate_layer_on_corpus(
    detector,
    papers:       list[dict],
    layer_name:   str,
    corpus_name:  str,
    text_field:   str = "abstract",
    label_field:  str = "label",
    positive_label: str = "retracted",
) -> LayerMetrics:
    """
    Evaluate a single detection layer on a corpus of labeled papers.

    Args:
        detector:       A detector with a .detect(text) -> list[dict] method
        papers:         List of paper dicts with text and label fields
        layer_name:     Name for reporting
        corpus_name:    Corpus identifier for reporting
        text_field:     Field containing the text to classify
        label_field:    Field containing the ground-truth label
        positive_label: Label value for positive class

    Returns:
        LayerMetrics with precision, recall, F1, AUROC.
    """
    metrics = LayerMetrics(layer=layer_name, corpus=corpus_name)

    for paper in papers:
        text     = paper.get(text_field, "")
        is_pos   = paper.get(label_field) == positive_label
        gt_label = 1 if is_pos else 0

        if not text.strip():
            continue

        hits = detector.detect(text)

        # Prediction: positive if any hit with confidence > 0
        predicted_pos = len(hits) > 0
        max_score     = max((h.get("confidence", 0.5) for h in hits), default=0.0)

        metrics.scores.append(max_score)
        metrics.labels.append(gt_label)

        if predicted_pos and is_pos:
            metrics.true_positives += 1
        elif predicted_pos and not is_pos:
            metrics.false_positives += 1
        elif not predicted_pos and is_pos:
            metrics.false_negatives += 1
        else:
            metrics.true_negatives += 1

    return metrics


def run_ablation_study(
    papers:       list[dict],
    corpus_name:  str,
    layers:       tuple[str, ...] = ("exact", "embedding", "mlm"),
    **pipeline_kwargs,
) -> pd.DataFrame:
    """
    Run the full ablation study across all 7 layer combinations.
    Returns a DataFrame with one row per (combination, corpus).

    Layer combinations:
      L1, L2, L3, L1+L2, L1+L3, L2+L3, L1+L2+L3

    This is the core experimental design for the paper (Section 5.4).
    """
    from tpc.pipeline import TorturedPhraseClassifier

    # All non-empty subsets of active layers
    active = [l for l in ("exact", "embedding", "mlm") if l in layers]
    combinations = []
    for n in range(1, len(active) + 1):
        from itertools import combinations as comb
        combinations.extend(list(comb(active, n)))

    results = []

    for combo in combinations:
        combo_name = "+".join({
            "exact": "L1", "embedding": "L2", "mlm": "L3"
        }[l] for l in combo)

        logger.info("Ablation: evaluating combination %s on %s", combo_name, corpus_name)

        clf = TorturedPhraseClassifier(layers=combo, **pipeline_kwargs)

        # Evaluate combined pipeline
        tp = fp = fn = tn = 0
        scores, labels = [], []

        for paper in papers:
            text   = paper.get("abstract", "")
            is_pos = paper.get("label") == "retracted"

            if not text.strip():
                continue

            result   = clf.classify(text)
            pred_pos = result.risk_score > 0.1

            scores.append(result.risk_score)
            labels.append(1 if is_pos else 0)

            if pred_pos and is_pos:      tp += 1
            elif pred_pos and not is_pos: fp += 1
            elif not pred_pos and is_pos: fn += 1
            else:                         tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        auroc = 0.5
        if len(set(labels)) == 2:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(labels, scores))

        results.append({
            "combination": combo_name,
            "corpus":      corpus_name,
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "f1":          round(f1, 4),
            "auroc":       round(auroc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(results)


def compute_novel_detection_rate(
    novel_papers: list[dict],
    perplexity_detector,
    expert_assessments: Optional[dict[str, bool]] = None,
) -> dict:
    """
    Compute Layer 3's novel phrase detection rate on papers not in any
    existing tortured phrase list. Addresses RQ2 (novel detection capability).

    Args:
        novel_papers: Papers from 2024-25 not covered by existing lists
        perplexity_detector: Initialized PerplexityDetector
        expert_assessments: Optional dict of {span: is_truly_tortured}
                            from human expert review

    Returns:
        Dict with detection statistics and candidate phrase list.
    """
    all_candidates = []
    papers_with_flags = 0

    for paper in novel_papers:
        text = paper.get("abstract", "")
        if not text.strip():
            continue

        hits = perplexity_detector.detect(text)
        if hits:
            papers_with_flags += 1
            for h in hits:
                all_candidates.append({
                    "span":         h["tortured"],
                    "log_perplexity": h["log_perplexity"],
                    "paper_pmid":   paper.get("pmid", ""),
                    "paper_label":  paper.get("label", ""),
                    "confidence":   h["confidence"],
                })

    # Sort by perplexity descending (highest suspicion first)
    all_candidates = sorted(
        all_candidates, key=lambda x: x["log_perplexity"], reverse=True
    )

    # If expert assessments provided, compute precision
    expert_precision = None
    if expert_assessments:
        assessed = [c for c in all_candidates if c["span"] in expert_assessments]
        if assessed:
            expert_precision = sum(
                1 for c in assessed if expert_assessments[c["span"]]
            ) / len(assessed)

    return {
        "total_papers":         len(novel_papers),
        "papers_with_flags":    papers_with_flags,
        "flag_rate":            papers_with_flags / max(len(novel_papers), 1),
        "total_candidates":     len(all_candidates),
        "unique_spans":         len({c["span"] for c in all_candidates}),
        "top_candidates":       all_candidates[:50],  # Appendix D
        "expert_precision":     expert_precision,
    }
