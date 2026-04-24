"""
Registry Validator
==================
CLI-callable validator for the signal registry.
Used by CI workflows to enforce warrant gates on every PR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import jsonschema

from tpc.registry.warrant import (
    assess_signal_file,
    WarrantGateResult,
    LITERARY_MIN_SIGHTINGS,
    STATISTICAL_MIN_PRECISION,
    STATISTICAL_MIN_RECALL,
)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "signals" / "schema.json"


def validate_registry(signals_dir: Path, strict: bool = False) -> tuple[bool, list[str]]:
    """
    Validate all signals in the registry.

    Args:
        signals_dir: Path to signals/phrases directory
        strict:      If True, fail on any candidate (non-confirmed) signals

    Returns:
        (all_valid: bool, errors: list[str])
    """
    schema = json.loads(SCHEMA_PATH.read_text())
    errors = []
    warnings = []

    yaml_files = sorted(signals_dir.rglob("*.yaml"))
    if not yaml_files:
        errors.append(f"No YAML files found in {signals_dir}")
        return False, errors

    seen_ids: set[str] = set()

    for yaml_path in yaml_files:
        rel = yaml_path.relative_to(signals_dir.parent)

        # 1. Parse YAML
        try:
            raw = yaml.safe_load(yaml_path.read_text())
        except yaml.YAMLError as e:
            errors.append(f"{rel}: YAML parse error — {e}")
            continue

        # 2. Schema validation
        try:
            jsonschema.validate(raw, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"{rel}: schema error — {e.message}")
            continue

        # 3. Duplicate ID check
        sig_id = raw.get("id", "")
        if sig_id in seen_ids:
            errors.append(f"{rel}: duplicate signal ID '{sig_id}'")
        seen_ids.add(sig_id)

        # 4. Warrant consistency check
        assessment = assess_signal_file(yaml_path)
        status = raw.get("status")

        if status == "confirmed":
            # All three warrant types must pass
            for warrant_type in ("literary", "terminological", "statistical"):
                result = getattr(assessment, warrant_type).result
                if result != WarrantGateResult.PASS:
                    errors.append(
                        f"{rel}: status='confirmed' but {warrant_type} warrant "
                        f"is '{result.value}' — must pass all three warrant types"
                    )

            # Statistical thresholds
            prec = assessment.statistical.precision_on_clean
            rec  = assessment.statistical.recall_on_retracted
            if prec is not None and prec < STATISTICAL_MIN_PRECISION:
                errors.append(
                    f"{rel}: precision_on_clean={prec:.3f} < "
                    f"required {STATISTICAL_MIN_PRECISION} for 'confirmed' status"
                )
            if rec is not None and rec < STATISTICAL_MIN_RECALL:
                errors.append(
                    f"{rel}: recall_on_retracted={rec:.3f} < "
                    f"required {STATISTICAL_MIN_RECALL} for 'confirmed' status"
                )

        elif status == "candidate":
            # Must have literary warrant
            if assessment.literary.result != WarrantGateResult.PASS:
                sightings = assessment.literary.independent_sightings
                errors.append(
                    f"{rel}: status='candidate' but literary warrant not satisfied "
                    f"(sightings={sightings}, required={LITERARY_MIN_SIGHTINGS})"
                )
            if strict:
                warnings.append(
                    f"{rel}: signal is 'candidate' — terminological and statistical "
                    "warrant review pending"
                )

        # 5. Evidence DOI format check
        dois = raw.get("warrant", {}).get("literary", {}).get("evidence_dois", [])
        for doi in dois:
            if not doi.startswith("10."):
                errors.append(f"{rel}: invalid DOI format '{doi}' — must start with '10.'")

    all_valid = len(errors) == 0
    return all_valid, errors + (warnings if strict else [])


def main():
    """Entry point for CI and CLI validation."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate TPC signal registry")
    parser.add_argument("signals_dir", type=Path, nargs="?",
                        default=Path(__file__).parent.parent.parent / "signals" / "phrases")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on candidate (unconfirmed) signals")
    args = parser.parse_args()

    print(f"Validating registry at: {args.signals_dir}")
    valid, messages = validate_registry(args.signals_dir, strict=args.strict)

    if messages:
        for msg in messages:
            prefix = "ERROR" if not valid else "WARN"
            print(f"  [{prefix}] {msg}")

    if valid:
        print(f"\n✓ Registry valid ({len(list(args.signals_dir.rglob('*.yaml')))} signals)")
        sys.exit(0)
    else:
        print(f"\n✗ Registry invalid — {len(messages)} error(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
