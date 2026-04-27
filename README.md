# Tortured Phrase Classifier (TPC)

> A warrant-based, three-layer classification system for detecting tortured phrases as terminological violations in scientific literature.

[![CI](https://github.com/your-org/tortured-phrase-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/tortured-phrase-classifier/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Paper mills corrupt the terminological infrastructure of science by substituting established domain terms with semantically incoherent paraphrases — "tortured phrases." From a knowledge organization perspective, this constitutes **terminological drift**: the systematic severing of the concept-term bond that controlled vocabularies depend upon.

This project provides:

1. **A three-layer detection pipeline** (lexical → semantic → probabilistic)
2. **A community-extensible, CI-gated signal registry** governed by explicit warrant criteria
3. **A pre-publication screening API** integrable with OJS and other manuscript systems
4. **Reproducible evaluation code** for the associated *Knowledge Organization* paper

## Theoretical Grounding

The system is grounded in three warrant types from KO/terminology theory:

| Warrant Type | Theoretical Basis | Design Role |
|---|---|---|
| Literary | Hulme (1911), Beghtol (1986) | Registry scope: ≥3 independent sightings |
| Terminological | Wüster (1979), ISO 704 | Registry quality: ISO 704 criteria |
| Statistical/Procedural | Floridi (2011), this paper | Layer 3 detection threshold |

## Architecture

```
manuscript PDF
      │
      ▼
  text extraction (pymupdf)
      │
      ├──► Layer 1: Exact Match (Aho-Corasick on warrant registry)
      │         literary + terminological warrant
      │
      ├──► Layer 2: Embedding Similarity (SPECTER + context coherence)
      │         terminological warrant / conceptual proximity
      │
      └──► Layer 3: MLM Perplexity (SciBERT masked LM)
                    statistical/procedural warrant
                          │
                          ▼
               novel phrase candidates
                          │
                          ▼
               human expert review → PR to registry
                          │
                          ▼
               CI validation (warrant gates) → merged
```

## Quick Start

```bash
pip install -e ".[dev]"

# Screen a PDF manuscript
tpc screen paper.pdf

# Screen with full evidence report
tpc screen paper.pdf --report --output report.json

# Run benchmark evaluation
tpc evaluate --corpus data/ground_truth/

# Validate the signal registry
tpc validate-registry
```

## Signal Registry

Every tortured phrase in the registry is a YAML file with explicit warrant documentation:

```yaml
id: TP-0042
tortured: "amino corrosive"
canonical: "amino acid"
domain: biochemistry
status: confirmed

warrant:
  literary:
    satisfied: true
    independent_sightings: 847
  terminological:
    satisfied: true
    iso704_criteria:
      precision: true
      economy: true
      appropriateness: true
  statistical:
    satisfied: true
    precision_on_clean: 0.997
    recall_on_retracted: 0.831
```

### Adding a New Signal

1. Open an issue with the candidate phrase and evidence DOIs
2. Fork the repo and create `signals/phrases/<domain>/<phrase_slug>.yaml`
3. Run `tpc validate-registry` locally — must pass all warrant gates
4. Open a pull request — CI enforces precision ≥ 0.95 and recall ≥ 0.70
5. Domain expert review required for terminological warrant (`confirmed` status)

## Repository Structure

```
tortured-phrase-classifier/
├── signals/                    # The warrant-based registry
│   ├── schema.json             # JSON Schema for signal validation
│   └── phrases/                # One YAML per tortured phrase
│       ├── biochemistry/
│       ├── computing/
│       ├── medicine/
│       ├── statistics/
│       └── general/
├── tpc/                        # Python package
│   ├── layers/                 # Three detection layers
│   ├── registry/               # Registry loader + warrant + ISO 704 validator
│   ├── acquisition/            # PubMed, OpenAlex, synthetic data
│   ├── evaluation/             # Metrics, benchmark, registry growth
│   ├── pipeline.py             # Orchestrator
│   ├── report.py               # Evidence report generator
│   ├── api.py                  # FastAPI pre-publication endpoint
│   └── cli.py                  # Typer CLI
├── data/
│   ├── ground_truth/           # Evaluation corpora
│   └── appendix/               # Paper appendix data
├── notebooks/                  # Reproducible analysis notebooks
├── tests/                      # pytest suite
└── .github/workflows/          # CI: validation + weekly benchmark
```

## Evaluation Results

*(populated after running `tpc evaluate`)*

| Layer | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| L1 exact match | — | — | — | — |
| L2 embedding | — | — | — | — |
| L3 perplexity | — | — | — | — |
| Combined | — | — | — | — |

## Citation

```bibtex
@software{Tran_Terminological_Drift_2026,
  author = {Tran, Quoc-Tan},
  title = {{Terminological Drift and Adversarial Paraphrasing: A Warrant-Based Classification System for Detecting Tortured Phrases (Tortured Phrase Classifier)}},
  month = {4},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/qtan-tran/tortured-phrase-classifier}},
  license = {Apache-2.0}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
