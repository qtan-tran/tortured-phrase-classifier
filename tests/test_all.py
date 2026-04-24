"""
Test Suite for Tortured Phrase Classifier
==========================================
Covers registry loading, warrant validation, all three detection layers,
and the pipeline orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signals_dir(tmp_path):
    """Create a minimal signal registry for testing."""
    schema_src = Path(__file__).parent.parent / "signals" / "schema.json"
    phrases_dir = tmp_path / "phrases" / "biochemistry"
    phrases_dir.mkdir(parents=True)

    # Copy schema
    schema_dst = tmp_path / "schema.json"
    schema_dst.write_text(schema_src.read_text())

    # Write a confirmed signal
    confirmed = {
        "id": "TP-0001",
        "tortured": "amino corrosive",
        "canonical": "amino acid",
        "domain": "biochemistry",
        "status": "confirmed",
        "warrant": {
            "literary": {
                "satisfied": True,
                "independent_sightings": 847,
                "evidence_dois": ["10.1016/test.001", "10.1016/test.002", "10.1016/test.003"],
            },
            "terminological": {
                "satisfied": True,
                "iso704_criteria": {
                    "precision": True, "economy": True,
                    "appropriateness": True, "consistency": True,
                    "transparency": True,
                },
                "domain_expert_orcid": "0000-0002-1234-5678",
                "review_date": "2024-03-15",
            },
            "statistical": {
                "satisfied": True,
                "precision_on_clean": 0.997,
                "recall_on_retracted": 0.831,
                "eval_corpus_size": 10000,
                "eval_date": "2024-04-01",
            },
        },
        "known_variants": ["amino corrosives"],
        "prevalence": {"retracted_papers": 847, "legitimate_papers": 0},
        "paraphrase_tool_origin": "CigoSpinner",
        "discovery_date": "2021-03-15",
        "discovered_by": "cabanac_et_al_2021",
        "notes": "Test signal.",
    }
    (phrases_dir / "amino_corrosive.yaml").write_text(yaml.dump(confirmed))

    # Write a candidate signal
    candidate = {
        "id": "TP-0099",
        "tortured": "profound learning",
        "canonical": "deep learning",
        "domain": "computing",
        "status": "candidate",
        "warrant": {
            "literary": {
                "satisfied": True,
                "independent_sightings": 5,
                "evidence_dois": ["10.1016/t.001", "10.1016/t.002", "10.1016/t.003"],
            },
            "terminological": {
                "satisfied": False,
                "iso704_criteria": {},
                "domain_expert_orcid": None,
                "review_date": None,
            },
            "statistical": {
                "satisfied": False,
                "precision_on_clean": None,
                "recall_on_retracted": None,
                "eval_corpus_size": None,
                "eval_date": None,
            },
        },
        "known_variants": [],
        "prevalence": {"retracted_papers": 5, "legitimate_papers": 0},
        "paraphrase_tool_origin": None,
        "discovery_date": "2024-01-01",
        "discovered_by": "test",
        "notes": None,
    }
    computing_dir = tmp_path / "phrases" / "computing"
    computing_dir.mkdir(parents=True)
    (computing_dir / "profound_learning.yaml").write_text(yaml.dump(candidate))

    return tmp_path


@pytest.fixture
def tortured_text():
    return (
        "In this study we examined the amino corrosive sequence and found "
        "significant variation in the nucleic harsh binding affinity. "
        "The profound learning model achieved high accuracy."
    )


@pytest.fixture
def clean_text():
    return (
        "In this study we examined the amino acid sequence and found "
        "significant variation in the nucleic acid binding affinity. "
        "The deep learning model achieved high accuracy on the benchmark dataset."
    )


# ---------------------------------------------------------------------------
# Registry loader tests
# ---------------------------------------------------------------------------

class TestRegistryLoader:

    def test_load_confirmed_signals(self, sample_signals_dir):
        from tpc.registry.loader import load_registry
        signals = load_registry(
            status_filter=("confirmed",),
            signals_dir=sample_signals_dir / "phrases",
        )
        assert len(signals) == 1
        assert signals[0].id == "TP-0001"
        assert signals[0].tortured == "amino corrosive"
        assert signals[0].canonical == "amino acid"

    def test_load_all_statuses(self, sample_signals_dir):
        from tpc.registry.loader import load_registry
        signals = load_registry(
            status_filter=("confirmed", "candidate"),
            signals_dir=sample_signals_dir / "phrases",
        )
        assert len(signals) == 2

    def test_signal_all_terms(self, sample_signals_dir):
        from tpc.registry.loader import load_registry
        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        sig = signals[0]
        assert "amino corrosive" in sig.all_terms
        assert "amino corrosives" in sig.all_terms

    def test_registry_summary(self, sample_signals_dir):
        from tpc.registry.loader import load_registry, registry_summary
        signals = load_registry(
            status_filter=("confirmed", "candidate"),
            signals_dir=sample_signals_dir / "phrases",
        )
        summary = registry_summary(signals)
        assert summary["total"] == 2
        assert "confirmed" in summary["by_status"]
        assert summary["by_domain"]["biochemistry"] == 1


# ---------------------------------------------------------------------------
# Warrant module tests
# ---------------------------------------------------------------------------

class TestWarrant:

    def test_warrant_record_status(self, sample_signals_dir):
        from tpc.registry.warrant import assess_signal_file
        yaml_path = sample_signals_dir / "phrases" / "biochemistry" / "amino_corrosive.yaml"
        assessment = assess_signal_file(yaml_path)
        assert assessment.recommended_status == "confirmed"

    def test_candidate_warrant(self, sample_signals_dir):
        from tpc.registry.warrant import assess_signal_file
        yaml_path = sample_signals_dir / "phrases" / "computing" / "profound_learning.yaml"
        assessment = assess_signal_file(yaml_path)
        assert assessment.recommended_status == "candidate"

    def test_literary_warrant_deficit(self, sample_signals_dir):
        from tpc.registry.warrant import LiteraryWarrantAssessment, WarrantGateResult
        assessment = LiteraryWarrantAssessment(
            independent_sightings=1,
            evidence_dois=["10.1016/test.001"],
        )
        assert assessment.result == WarrantGateResult.INCOMPLETE
        assert assessment.deficit == 2

    def test_statistical_warrant_fail(self):
        from tpc.registry.warrant import StatisticalWarrantAssessment, WarrantGateResult
        assessment = StatisticalWarrantAssessment(
            precision_on_clean=0.85,    # below 0.95 threshold
            recall_on_retracted=0.80,
            eval_corpus_size=1000,
            eval_date="2024-01-01",
        )
        assert assessment.result == WarrantGateResult.FAIL
        assert assessment.precision_deficit > 0


# ---------------------------------------------------------------------------
# ISO 704 validator tests
# ---------------------------------------------------------------------------

class TestISO704Validator:

    def test_valid_canonical(self):
        from tpc.registry.iso704_validator import assess_canonical
        result = assess_canonical("amino acid", "biochemistry")
        # Economy and linguistic correctness should pass
        assert result.economy is True
        assert result.linguistic_correctness is True

    def test_lay_marker_detected(self):
        from tpc.registry.iso704_validator import assess_canonical
        result = assess_canonical("breast peril", "medicine")
        # 'peril' is a lay marker — appropriateness should fail
        assert result.appropriateness is False

    def test_tortured_incoherence_assessment(self):
        from tpc.registry.iso704_validator import assess_tortured_incoherence
        result = assess_tortured_incoherence(
            tortured="amino corrosive",
            canonical="amino acid",
            domain="biochemistry",
        )
        assert result["severity"] in ("medium", "high")
        assert len(result["violations"]) > 0


# ---------------------------------------------------------------------------
# Registry validator tests
# ---------------------------------------------------------------------------

class TestRegistryValidator:

    def test_valid_registry_passes(self, sample_signals_dir):
        from tpc.registry.validator import validate_registry
        valid, errors = validate_registry(sample_signals_dir / "phrases")
        # Errors may exist for candidate signals with incomplete warrant
        # but schema should be valid
        schema_errors = [e for e in errors if "schema error" in e]
        assert len(schema_errors) == 0

    def test_invalid_doi_flagged(self, tmp_path):
        from tpc.registry.validator import validate_registry

        schema_src = Path(__file__).parent.parent / "signals" / "schema.json"
        (tmp_path / "schema.json").write_text(schema_src.read_text())

        d = tmp_path / "phrases" / "general"
        d.mkdir(parents=True)
        bad_signal = {
            "id": "TP-9999",
            "tortured": "test phrase",
            "canonical": "test term",
            "domain": "general",
            "status": "candidate",
            "warrant": {
                "literary": {
                    "satisfied": True,
                    "independent_sightings": 3,
                    "evidence_dois": ["not-a-doi"],  # invalid
                },
                "terminological": {"satisfied": False},
                "statistical": {"satisfied": False},
            },
            "known_variants": [],
            "prevalence": {"retracted_papers": 3, "legitimate_papers": 0},
            "discovery_date": "2024-01-01",
            "discovered_by": "test",
        }
        (d / "test_phrase.yaml").write_text(yaml.dump(bad_signal))

        valid, errors = validate_registry(tmp_path / "phrases")
        doi_errors = [e for e in errors if "DOI" in e]
        assert len(doi_errors) > 0
        assert not valid


# ---------------------------------------------------------------------------
# Layer 1: Exact Match tests
# ---------------------------------------------------------------------------

class TestExactMatchDetector:

    def test_detects_known_phrase(self, sample_signals_dir, tortured_text):
        from tpc.layers.exact_match import ExactMatchDetector
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        detector = ExactMatchDetector(signals=signals)
        hits = detector.detect(tortured_text)

        hit_tortured = [h["tortured"].lower() for h in hits]
        assert "amino corrosive" in hit_tortured

    def test_no_false_positive_on_clean(self, sample_signals_dir, clean_text):
        from tpc.layers.exact_match import ExactMatchDetector
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        detector = ExactMatchDetector(signals=signals)
        hits = detector.detect(clean_text)

        tortured_hits = [h for h in hits if h["tortured"] == "amino corrosive"]
        assert len(tortured_hits) == 0

    def test_hit_has_required_fields(self, sample_signals_dir, tortured_text):
        from tpc.layers.exact_match import ExactMatchDetector
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        detector = ExactMatchDetector(signals=signals)
        hits = detector.detect(tortured_text)

        if hits:
            h = hits[0]
            for field in ("signal_id", "tortured", "canonical", "domain",
                          "layer", "confidence", "position", "explanation"):
                assert field in h, f"Missing field: {field}"

    def test_confidence_is_one(self, sample_signals_dir, tortured_text):
        from tpc.layers.exact_match import ExactMatchDetector
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        detector = ExactMatchDetector(signals=signals)
        hits = detector.detect(tortured_text)

        for h in hits:
            assert h["confidence"] == 1.0

    def test_variant_detected(self, sample_signals_dir):
        from tpc.layers.exact_match import ExactMatchDetector
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        detector = ExactMatchDetector(signals=signals)
        text = "The amino corrosives were analyzed."
        hits = detector.detect(text)
        assert any("amino corrosive" in h["tortured"] for h in hits)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_exact_only_pipeline(self, sample_signals_dir, tortured_text):
        from tpc.pipeline import TorturedPhraseClassifier
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        clf = TorturedPhraseClassifier(
            layers=("exact",),
            exact_kwargs={"signals": signals},
        )
        result = clf.classify(tortured_text)

        assert result.risk_score > 0
        assert result.risk_level in ("low", "medium", "high", "critical")
        assert len(result.hits) > 0

    def test_clean_text_low_risk(self, sample_signals_dir, clean_text):
        from tpc.pipeline import TorturedPhraseClassifier
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        clf = TorturedPhraseClassifier(
            layers=("exact",),
            exact_kwargs={"signals": signals},
        )
        result = clf.classify(clean_text)
        assert result.risk_score < 0.3

    def test_result_to_dict(self, sample_signals_dir, tortured_text):
        from tpc.pipeline import TorturedPhraseClassifier
        from tpc.registry.loader import load_registry

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        clf = TorturedPhraseClassifier(
            layers=("exact",),
            exact_kwargs={"signals": signals},
        )
        result = clf.classify(tortured_text)
        d = result.to_dict()

        for key in ("risk_score", "risk_level", "summary", "hits",
                    "layer_hits", "layers_used"):
            assert key in d


# ---------------------------------------------------------------------------
# Report generator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:

    def test_report_structure(self, sample_signals_dir, tortured_text):
        from tpc.pipeline import TorturedPhraseClassifier
        from tpc.registry.loader import load_registry
        from tpc.report import build_report

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        clf = TorturedPhraseClassifier(
            layers=("exact",),
            exact_kwargs={"signals": signals},
        )
        result = clf.classify(tortured_text)
        report = build_report(result=result, text=tortured_text)

        for key in ("generated_at", "risk_assessment", "findings",
                    "layer_summary", "disclaimer"):
            assert key in report

    def test_report_written_to_file(self, sample_signals_dir, tortured_text, tmp_path):
        from tpc.pipeline import TorturedPhraseClassifier
        from tpc.registry.loader import load_registry
        from tpc.report import build_report

        signals = load_registry(signals_dir=sample_signals_dir / "phrases")
        clf = TorturedPhraseClassifier(
            layers=("exact",),
            exact_kwargs={"signals": signals},
        )
        result  = clf.classify(tortured_text)
        outpath = tmp_path / "report.json"
        build_report(result=result, text=tortured_text, output_path=str(outpath))

        assert outpath.exists()
        data = json.loads(outpath.read_text())
        assert "risk_assessment" in data
