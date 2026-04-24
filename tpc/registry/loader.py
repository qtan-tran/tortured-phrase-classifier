"""
Registry Loader
===============
Loads and validates the warrant-based signal registry from YAML files.
Implements literary + terminological + statistical warrant typing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
import jsonschema

logger = logging.getLogger(__name__)

SCHEMA_PATH  = Path(__file__).parent.parent.parent / "signals" / "schema.json"
SIGNALS_DIR  = Path(__file__).parent.parent.parent / "signals" / "phrases"


class SignalStatus(str, Enum):
    CONFIRMED  = "confirmed"
    CANDIDATE  = "candidate"
    DEPRECATED = "deprecated"


class WarrantType(str, Enum):
    """
    Three warrant types following Beghtol (1986) and ISO 704.
    Each maps to a distinct layer of the detection architecture.
    """
    LITERARY       = "literary"        # ≥3 independent corpus sightings
    TERMINOLOGICAL = "terminological"  # ISO 704 canonical quality
    STATISTICAL    = "statistical"     # CI precision/recall gates


@dataclass
class WarrantRecord:
    literary_satisfied:         bool
    literary_sightings:         int
    literary_dois:              list[str]
    terminological_satisfied:   bool
    iso704_criteria:            dict
    domain_expert_orcid:        Optional[str]
    review_date:                Optional[str]
    statistical_satisfied:      bool
    precision_on_clean:         Optional[float]
    recall_on_retracted:        Optional[float]

    @property
    def all_satisfied(self) -> bool:
        return (self.literary_satisfied and
                self.terminological_satisfied and
                self.statistical_satisfied)

    def unsatisfied_types(self) -> list[WarrantType]:
        result = []
        if not self.literary_satisfied:       result.append(WarrantType.LITERARY)
        if not self.terminological_satisfied: result.append(WarrantType.TERMINOLOGICAL)
        if not self.statistical_satisfied:    result.append(WarrantType.STATISTICAL)
        return result


@dataclass
class Signal:
    id:                     str
    tortured:               str
    canonical:              str
    domain:                 str
    status:                 SignalStatus
    warrant:                WarrantRecord
    known_variants:         list[str]    = field(default_factory=list)
    prevalence_retracted:   int          = 0
    prevalence_legitimate:  int          = 0
    paraphrase_tool_origin: Optional[str] = None
    discovery_date:         Optional[str] = None
    discovered_by:          Optional[str] = None
    notes:                  Optional[str] = None

    @property
    def all_terms(self) -> list[str]:
        """All term strings this signal covers (tortured + variants)."""
        return [self.tortured] + self.known_variants

    @property
    def suspicion_ratio(self) -> float:
        """Ratio of occurrences in retracted vs legitimate papers."""
        if self.prevalence_legitimate == 0:
            return float("inf") if self.prevalence_retracted > 0 else 0.0
        return self.prevalence_retracted / self.prevalence_legitimate


def _parse_warrant(raw_warrant: dict) -> WarrantRecord:
    lit = raw_warrant.get("literary", {})
    ter = raw_warrant.get("terminological", {})
    sta = raw_warrant.get("statistical", {})
    return WarrantRecord(
        literary_satisfied=lit.get("satisfied", False),
        literary_sightings=lit.get("independent_sightings", 0),
        literary_dois=lit.get("evidence_dois", []),
        terminological_satisfied=ter.get("satisfied", False),
        iso704_criteria=ter.get("iso704_criteria", {}),
        domain_expert_orcid=ter.get("domain_expert_orcid"),
        review_date=ter.get("review_date"),
        statistical_satisfied=sta.get("satisfied", False),
        precision_on_clean=sta.get("precision_on_clean"),
        recall_on_retracted=sta.get("recall_on_retracted"),
    )


def _parse_signal(raw: dict) -> Signal:
    prev = raw.get("prevalence", {})
    return Signal(
        id=raw["id"],
        tortured=raw["tortured"],
        canonical=raw["canonical"],
        domain=raw["domain"],
        status=SignalStatus(raw["status"]),
        warrant=_parse_warrant(raw.get("warrant", {})),
        known_variants=raw.get("known_variants", []),
        prevalence_retracted=prev.get("retracted_papers", 0),
        prevalence_legitimate=prev.get("legitimate_papers", 0),
        paraphrase_tool_origin=raw.get("paraphrase_tool_origin"),
        discovery_date=raw.get("discovery_date"),
        discovered_by=raw.get("discovered_by"),
        notes=raw.get("notes"),
    )


def load_registry(
    status_filter: tuple[str, ...] = ("confirmed",),
    domain_filter: Optional[list[str]] = None,
    signals_dir:   Optional[Path] = None,
) -> list[Signal]:
    """
    Load and validate all signals from the YAML registry.

    Args:
        status_filter: Only return signals with these statuses.
                       Default: ('confirmed',) — only fully warranted signals.
        domain_filter: If given, restrict to these domains.
        signals_dir:   Override default signals directory (for testing).

    Returns:
        List of Signal objects passing the filters.
    """
    schema    = json.loads(SCHEMA_PATH.read_text())
    sig_dir   = signals_dir or SIGNALS_DIR
    signals   = []
    errors    = []

    for yaml_path in sorted(sig_dir.rglob("*.yaml")):
        try:
            raw = yaml.safe_load(yaml_path.read_text())
            jsonschema.validate(raw, schema)
            sig = _parse_signal(raw)
            if sig.status.value in status_filter:
                if domain_filter is None or sig.domain in domain_filter:
                    signals.append(sig)
        except jsonschema.ValidationError as e:
            errors.append(f"{yaml_path.name}: schema error — {e.message}")
        except Exception as e:
            errors.append(f"{yaml_path.name}: parse error — {e}")

    if errors:
        for err in errors:
            logger.warning("Registry load error: %s", err)

    logger.info(
        "Registry loaded: %d signals (%s) from %s",
        len(signals), ", ".join(status_filter), sig_dir
    )
    return signals


def registry_summary(signals: Optional[list[Signal]] = None) -> dict:
    """Return summary statistics of the registry — for paper Table/RQ1."""
    if signals is None:
        signals = load_registry(status_filter=("confirmed", "candidate", "deprecated"))

    by_status = {}
    for s in signals:
        by_status.setdefault(s.status.value, []).append(s)

    by_domain = {}
    for s in signals:
        by_domain.setdefault(s.domain, []).append(s)

    confirmed = by_status.get("confirmed", [])
    return {
        "total":               len(signals),
        "by_status":           {k: len(v) for k, v in by_status.items()},
        "by_domain":           {k: len(v) for k, v in by_domain.items()},
        "mean_precision":      (
            sum(s.warrant.precision_on_clean or 0 for s in confirmed) / len(confirmed)
            if confirmed else 0.0
        ),
        "mean_recall":         (
            sum(s.warrant.recall_on_retracted or 0 for s in confirmed) / len(confirmed)
            if confirmed else 0.0
        ),
        "total_sightings":     sum(s.prevalence_retracted for s in signals),
        "warrant_pass_rates":  {
            "literary":       sum(1 for s in signals if s.warrant.literary_satisfied) / max(len(signals), 1),
            "terminological": sum(1 for s in signals if s.warrant.terminological_satisfied) / max(len(signals), 1),
            "statistical":    sum(1 for s in signals if s.warrant.statistical_satisfied) / max(len(signals), 1),
        },
    }
