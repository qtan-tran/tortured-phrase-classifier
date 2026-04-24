"""
ISO 704 Quality Validator
=========================
Operationalizes ISO 704 (Terminology work — Principles and methods)
term quality criteria as computable pre-screening checks.

This module provides automated first-pass assessment of whether a
proposed canonical form satisfies ISO 704 criteria. Domain expert
review remains required for terminological warrant confirmation.

ISO 704 criteria implemented:
  precision       — term designates exactly one concept
  economy         — shortest form that maintains precision
  appropriateness — fits domain's linguistic conventions
  consistency     — coherent with related terms in same concept system
  transparency    — meaning inferable from term's morphological form
  derivability    — enables formation of related terms
  linguistic_correctness — grammatically well-formed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


# ---------------------------------------------------------------------------
# Domain glossaries — simplified; extend with real terminology databases
# ---------------------------------------------------------------------------

DOMAIN_GLOSSARIES: dict[str, set[str]] = {
    "biochemistry": {
        "amino acid", "nucleic acid", "protein", "enzyme", "lipid",
        "carbohydrate", "glucose", "lactose", "dna", "rna", "atp",
        "mitochondria", "ribosome", "cell membrane", "cytoplasm",
        "gene expression", "transcription", "translation", "metabolism",
    },
    "computing": {
        "neural network", "deep learning", "machine learning",
        "artificial intelligence", "natural language processing",
        "computer vision", "random forest", "support vector machine",
        "convolutional neural network", "recurrent neural network",
        "transformer", "algorithm", "data structure", "database",
        "signal to noise", "precision", "recall", "f1 score",
    },
    "medicine": {
        "breast cancer", "lung cancer", "diabetes mellitus",
        "hypertension", "myocardial infarction", "stroke",
        "kidney failure", "liver disease", "blood pressure",
        "immune system", "clinical trial", "placebo", "randomized",
    },
    "statistics": {
        "random sampling", "standard deviation", "confidence interval",
        "hypothesis testing", "p-value", "regression analysis",
        "correlation coefficient", "analysis of variance",
        "monte carlo", "bayesian inference", "likelihood ratio",
    },
    "general": {
        "signal to noise", "mean temperature", "standard error",
        "control group", "sample size", "effect size", "bias",
    },
}

# Common lay synonyms that should NOT appear in technical terms
LAY_MARKERS: set[str] = {
    "thing", "stuff", "problem", "issue", "effect", "nice", "bad",
    "good", "big", "small", "hard", "easy", "fast", "slow",
    "peril", "danger", "woe", "harsh", "corrosive", "fake",
    "counterfeit", "sham", "profound", "commotion", "clamor",
}

# Scientific morphemes that indicate domain-appropriate terms
SCIENTIFIC_SUFFIXES: tuple[str, ...] = (
    "-ase", "-ine", "-ose", "-ol", "-ide", "-ation", "-itis",
    "-oma", "-logy", "-graphy", "-metry", "-scopy", "-tomy",
    "-plasty", "-ectomy", "-emia", "-uria", "-penia",
)


@dataclass
class ISO704Assessment:
    """
    Full ISO 704 quality assessment for a candidate canonical term.
    All criteria must pass for terminological warrant to be satisfied.
    """
    precision:              bool    # designates exactly one concept
    economy:                bool    # shortest unambiguous form
    appropriateness:        bool    # fits domain linguistic conventions
    consistency:            bool    # coherent with related terms
    transparency:           bool    # meaning inferable from form
    derivability:           bool    # enables related term formation
    linguistic_correctness: bool    # grammatically well-formed

    notes: Optional[str] = None

    REQUIRED_CRITERIA = (
        "precision", "economy", "appropriateness",
        "consistency", "transparency"
    )

    @property
    def is_valid_canonical(self) -> bool:
        """All required criteria must be satisfied."""
        return all(getattr(self, c) for c in self.REQUIRED_CRITERIA)

    @property
    def failed_criteria(self) -> list[str]:
        all_criteria = (
            "precision", "economy", "appropriateness",
            "consistency", "transparency", "derivability",
            "linguistic_correctness"
        )
        return [c for c in all_criteria if not getattr(self, c)]

    def to_dict(self) -> dict:
        return {
            "precision":              self.precision,
            "economy":                self.economy,
            "appropriateness":        self.appropriateness,
            "consistency":            self.consistency,
            "transparency":           self.transparency,
            "derivability":           self.derivability,
            "linguistic_correctness": self.linguistic_correctness,
            "is_valid_canonical":     self.is_valid_canonical,
            "failed_criteria":        self.failed_criteria,
            "notes":                  self.notes,
        }


def assess_canonical(
    canonical: str,
    domain: str,
    tortured: Optional[str] = None,
) -> ISO704Assessment:
    """
    Automated pre-screening of ISO 704 criteria for a proposed canonical term.

    This provides a first-pass automated assessment. A domain expert must
    still confirm terminological warrant before a signal reaches 'confirmed'.

    Args:
        canonical: The legitimate scientific term (e.g., "amino acid")
        domain:    Scientific domain (biochemistry, computing, medicine, etc.)
        tortured:  The proposed tortured variant (used for contrast checks)

    Returns:
        ISO704Assessment with pass/fail per criterion and overall validity.
    """
    domain_terms = DOMAIN_GLOSSARIES.get(domain, set())
    words        = canonical.lower().split()
    notes_parts  = []

    # --- Precision ---
    # Term appears in domain glossary with a single well-defined concept
    precision = canonical.lower() in domain_terms
    if not precision:
        notes_parts.append(
            f"'{canonical}' not found in {domain} glossary — "
            "manual verification required"
        )

    # --- Economy ---
    # Not unnecessarily long; ≤4 words is a reasonable heuristic
    economy = len(words) <= 4
    if not economy:
        notes_parts.append(
            f"Term has {len(words)} words — consider whether a shorter "
            "form exists without loss of precision"
        )

    # --- Appropriateness ---
    # No lay synonyms where technical terms exist; domain register respected
    has_lay = any(w in LAY_MARKERS for w in words)
    appropriateness = not has_lay
    if has_lay:
        bad = [w for w in words if w in LAY_MARKERS]
        notes_parts.append(
            f"Lay markers found: {bad} — these suggest the proposed canonical "
            "may itself be a non-standard form"
        )

    # --- Consistency ---
    # Related terms in domain use the same root/pattern
    consistency = _check_family_consistency(canonical, domain_terms)
    if not consistency and domain_terms:
        notes_parts.append(
            "No morphologically related terms found in domain glossary — "
            "verify term family coherence"
        )

    # --- Transparency ---
    # Meaning somewhat inferable from morphological form
    transparency = _has_recognizable_morphology(canonical)
    if not transparency:
        notes_parts.append(
            "Term transparency is low — meaning not obviously inferable "
            "from form alone (acceptable if established by convention)"
        )

    # --- Derivability ---
    # Can generate adjective/verb/noun variants (e.g., "protein" → "proteinaceous")
    derivability = _has_derivable_forms(canonical)

    # --- Linguistic correctness ---
    linguistic_correctness = _is_grammatical(canonical)
    if not linguistic_correctness:
        notes_parts.append(
            "Term fails basic grammaticality check — verify formatting"
        )

    return ISO704Assessment(
        precision=precision,
        economy=economy,
        appropriateness=appropriateness,
        consistency=consistency,
        transparency=transparency,
        derivability=derivability,
        linguistic_correctness=linguistic_correctness,
        notes="; ".join(notes_parts) if notes_parts else None,
    )


def assess_tortured_incoherence(tortured: str, canonical: str,
                                 domain: str) -> dict:
    """
    Assess how severely a tortured phrase violates ISO 704 criteria
    relative to its canonical form. Used for Layer 2 signal strength
    calibration and for the paper's qualitative examples table.
    """
    domain_terms = DOMAIN_GLOSSARIES.get(domain, set())
    tortured_words = tortured.lower().split()

    violations = []

    # Does the tortured phrase appear in domain glossary? (It should not.)
    if tortured.lower() in domain_terms:
        violations.append("false_positive_risk: tortured phrase is a legitimate term")

    # Lay marker contamination
    lay_contamination = [w for w in tortured_words if w in LAY_MARKERS]
    if lay_contamination:
        violations.append(f"lay_marker_substitution: {lay_contamination}")

    # Morphological incoherence (scientific suffix lost)
    canonical_has_suffix = any(
        canonical.lower().endswith(s.lstrip("-"))
        for s in SCIENTIFIC_SUFFIXES
    )
    tortured_has_suffix = any(
        tortured.lower().endswith(s.lstrip("-"))
        for s in SCIENTIFIC_SUFFIXES
    )
    if canonical_has_suffix and not tortured_has_suffix:
        violations.append("morphological_degradation: scientific suffix lost in tortured form")

    # Word count change
    c_words = len(canonical.split())
    t_words = len(tortured.split())
    if abs(c_words - t_words) > 1:
        violations.append(f"economy_violation: word count changed {c_words}→{t_words}")

    return {
        "tortured":    tortured,
        "canonical":   canonical,
        "domain":      domain,
        "violations":  violations,
        "severity":    "high" if len(violations) >= 2 else
                       "medium" if violations else "low",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_family_consistency(canonical: str, domain_terms: set[str]) -> bool:
    """Check that morphological relatives exist in the domain glossary."""
    if not domain_terms:
        return True  # Can't assess without glossary
    root = canonical.split()[0].lower()
    return any(t.startswith(root) for t in domain_terms if t != canonical.lower())


def _has_recognizable_morphology(term: str) -> bool:
    """Check if term has recognizable scientific morphological markers."""
    term_lower = term.lower()
    if any(term_lower.endswith(s.lstrip("-")) for s in SCIENTIFIC_SUFFIXES):
        return True
    # Multi-word terms are generally transparent by composition
    if len(term.split()) > 1:
        return True
    # Single word >= 6 chars with a root pattern
    return len(term) >= 6 and bool(re.search(r"[aeiou]{1,2}[a-z]{2,}", term_lower))


def _has_derivable_forms(term: str) -> bool:
    """Rough check: can adjective/verb forms be derived?"""
    # Most multi-word scientific terms are derivable
    if len(term.split()) > 1:
        return True
    # Single words with derivation suffixes
    DERIVATION_STEMS = ("-ize", "-ic", "-ous", "-al", "-ase", "-ation")
    term_lower = term.lower()
    return any(term_lower.endswith(s.lstrip("-")) for s in DERIVATION_STEMS)


def _is_grammatical(term: str) -> bool:
    """Basic grammaticality: lowercase start, no trailing punctuation."""
    if not term:
        return False
    return (term[0].islower() and
            term == term.strip() and
            not term.endswith((".", ",", ";", ":")))
