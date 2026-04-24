"""
Warrant Governance Module
=========================
Operationalizes warrant theory (Beghtol 1986) as computable checks.
Maps literary, terminological, and statistical warrant to explicit
design decisions in the TPC registry.

Theoretical basis:
- Hulme (1911): literary warrant — terms justified by corpus occurrence
- Beghtol (1986): warrant types in classification systems
- Wüster (1979) / ISO 704: terminological warrant criteria
- This paper: statistical/procedural warrant as extension for computational systems
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Warrant gate thresholds — explicitly documented design choices
# ---------------------------------------------------------------------------

LITERARY_MIN_SIGHTINGS:    int   = 3      # minimum independent corpus sightings
STATISTICAL_MIN_PRECISION: float = 0.95  # on clean (non-retracted) corpus
STATISTICAL_MIN_RECALL:    float = 0.70  # on retracted/flagged corpus


class WarrantGateResult(str, Enum):
    PASS       = "pass"
    FAIL       = "fail"
    INCOMPLETE = "incomplete"   # data not yet collected


@dataclass
class LiteraryWarrantAssessment:
    """
    Literary warrant: a term is justified when it is attested in
    sufficient independent documents in the target corpus.
    Adapted from Hulme (1911) via Beghtol (1986).
    """
    independent_sightings:  int
    evidence_dois:          list[str]
    min_required:           int = LITERARY_MIN_SIGHTINGS

    @property
    def result(self) -> WarrantGateResult:
        if self.independent_sightings >= self.min_required:
            return WarrantGateResult.PASS
        elif self.independent_sightings > 0:
            return WarrantGateResult.INCOMPLETE
        return WarrantGateResult.FAIL

    @property
    def deficit(self) -> int:
        """How many more sightings needed to pass."""
        return max(0, self.min_required - self.independent_sightings)

    def report(self) -> dict:
        return {
            "type": "literary",
            "result": self.result.value,
            "sightings": self.independent_sightings,
            "required": self.min_required,
            "deficit": self.deficit,
            "evidence_dois": self.evidence_dois,
            "rationale": (
                "A tortured phrase must be independently observed in ≥3 "
                "retracted or flagged papers before entering the registry. "
                "This operationalizes Hulme's (1911) literary warrant: terms "
                "are justified by their occurrence in the literature."
            ),
        }


@dataclass
class TerminologicalWarrantAssessment:
    """
    Terminological warrant: the canonical form must satisfy ISO 704
    term quality criteria as assessed by a domain expert.
    Grounds the registry in Wüster's General Theory of Terminology.
    """
    iso704_criteria:      dict
    domain_expert_orcid:  Optional[str]
    review_date:          Optional[str]

    ISO704_REQUIRED = ("precision", "economy", "appropriateness",
                       "consistency", "transparency")

    @property
    def result(self) -> WarrantGateResult:
        if self.domain_expert_orcid is None:
            return WarrantGateResult.INCOMPLETE
        failed = [k for k in self.ISO704_REQUIRED
                  if not self.iso704_criteria.get(k)]
        return WarrantGateResult.PASS if not failed else WarrantGateResult.FAIL

    @property
    def failed_criteria(self) -> list[str]:
        return [k for k in self.ISO704_REQUIRED
                if not self.iso704_criteria.get(k)]

    def report(self) -> dict:
        return {
            "type": "terminological",
            "result": self.result.value,
            "iso704_criteria": self.iso704_criteria,
            "failed_criteria": self.failed_criteria,
            "domain_expert_orcid": self.domain_expert_orcid,
            "review_date": self.review_date,
            "rationale": (
                "The canonical form must satisfy ISO 704 quality criteria: "
                "precision, economy, appropriateness, consistency, transparency. "
                "Assessment requires a domain expert reviewer identified by ORCID."
            ),
        }


@dataclass
class StatisticalWarrantAssessment:
    """
    Statistical/procedural warrant: a novel extension of warrant theory
    to computational detection systems.

    A signal is statistically warranted when it achieves acceptable
    precision on the clean corpus (avoiding false positives against
    legitimate papers) and acceptable recall on the retracted corpus
    (detecting real paper mill content).
    """
    precision_on_clean:   Optional[float]
    recall_on_retracted:  Optional[float]
    eval_corpus_size:     Optional[int]
    eval_date:            Optional[str]
    min_precision:        float = STATISTICAL_MIN_PRECISION
    min_recall:           float = STATISTICAL_MIN_RECALL

    @property
    def result(self) -> WarrantGateResult:
        if self.precision_on_clean is None or self.recall_on_retracted is None:
            return WarrantGateResult.INCOMPLETE
        if (self.precision_on_clean >= self.min_precision and
                self.recall_on_retracted >= self.min_recall):
            return WarrantGateResult.PASS
        return WarrantGateResult.FAIL

    @property
    def precision_deficit(self) -> float:
        if self.precision_on_clean is None:
            return self.min_precision
        return max(0.0, self.min_precision - self.precision_on_clean)

    @property
    def recall_deficit(self) -> float:
        if self.recall_on_retracted is None:
            return self.min_recall
        return max(0.0, self.min_recall - self.recall_on_retracted)

    def report(self) -> dict:
        return {
            "type": "statistical",
            "result": self.result.value,
            "precision_on_clean": self.precision_on_clean,
            "recall_on_retracted": self.recall_on_retracted,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "precision_deficit": self.precision_deficit,
            "recall_deficit": self.recall_deficit,
            "eval_corpus_size": self.eval_corpus_size,
            "rationale": (
                "A signal must achieve precision ≥ 0.95 on the clean corpus "
                "(protecting legitimate authors from false positives) and recall "
                "≥ 0.70 on the retracted corpus (detecting real tortured phrases). "
                "This is statistical/procedural warrant: the community's prior "
                "practice (encoded in SciBERT/SPECTER) legitimizes detection."
            ),
        }


@dataclass
class FullWarrantAssessment:
    """Complete warrant assessment for a signal across all three types."""
    signal_id:       str
    literary:        LiteraryWarrantAssessment
    terminological:  TerminologicalWarrantAssessment
    statistical:     StatisticalWarrantAssessment

    @property
    def recommended_status(self) -> str:
        if all(w == WarrantGateResult.PASS for w in [
            self.literary.result,
            self.terminological.result,
            self.statistical.result,
        ]):
            return "confirmed"
        elif self.literary.result == WarrantGateResult.PASS:
            return "candidate"
        return "rejected"

    def full_report(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "recommended_status": self.recommended_status,
            "warrant_assessments": {
                "literary":       self.literary.report(),
                "terminological": self.terminological.report(),
                "statistical":    self.statistical.report(),
            },
            "all_passed": self.recommended_status == "confirmed",
        }


def assess_signal_file(yaml_path: Path) -> FullWarrantAssessment:
    """Run warrant assessment on a signal YAML file."""
    raw = yaml.safe_load(yaml_path.read_text())
    w   = raw.get("warrant", {})
    lit = w.get("literary", {})
    ter = w.get("terminological", {})
    sta = w.get("statistical", {})

    return FullWarrantAssessment(
        signal_id=raw["id"],
        literary=LiteraryWarrantAssessment(
            independent_sightings=lit.get("independent_sightings", 0),
            evidence_dois=lit.get("evidence_dois", []),
        ),
        terminological=TerminologicalWarrantAssessment(
            iso704_criteria=ter.get("iso704_criteria", {}),
            domain_expert_orcid=ter.get("domain_expert_orcid"),
            review_date=ter.get("review_date"),
        ),
        statistical=StatisticalWarrantAssessment(
            precision_on_clean=sta.get("precision_on_clean"),
            recall_on_retracted=sta.get("recall_on_retracted"),
            eval_corpus_size=sta.get("eval_corpus_size"),
            eval_date=sta.get("eval_date"),
        ),
    )
