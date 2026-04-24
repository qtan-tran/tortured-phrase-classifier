# Contributing to the Tortured Phrase Classifier

Thank you for helping defend scientific vocabulary integrity. This guide explains how to contribute new signals to the registry, improve detection layers, or extend the codebase.

---

## Adding a New Signal to the Registry

A signal is a confirmed tortured phrase with documented warrant. The process mirrors a controlled vocabulary editorial workflow.

### Step 1 — Open an Issue

Open a GitHub issue titled `[Signal Proposal] <tortured phrase>` and include:
- The tortured phrase (exact string as it appears in papers)
- The canonical form (the legitimate scientific term it replaces)
- The domain (biochemistry, computing, medicine, statistics, general)
- At least 3 DOIs of papers where you observed the phrase
- The likely paraphrasing tool (if known)

### Step 2 — Create the YAML file

Fork the repository and create:
`signals/phrases/<domain>/<phrase_slug>.yaml`

Use this template:

```yaml
id: TP-XXXX                    # request a number in your issue
tortured: "your phrase here"
canonical: "the real term"
domain: biochemistry            # biochemistry|computing|medicine|statistics|general
status: candidate               # always start as candidate

warrant:
  literary:
    satisfied: true             # you must have ≥3 independent sightings
    independent_sightings: 3    # count
    evidence_dois:
      - "10.xxxx/xxxxxx"
      - "10.xxxx/xxxxxx"
      - "10.xxxx/xxxxxx"
  terminological:
    satisfied: false            # leave false; domain expert will assess
    iso704_criteria:
      precision: null
      economy: null
      appropriateness: null
      consistency: null
      transparency: null
    domain_expert_orcid: null
    review_date: null
  statistical:
    satisfied: false            # CI gates will populate this
    precision_on_clean: null
    recall_on_retracted: null

known_variants: []
prevalence:
  retracted_papers: 3           # your sightings count
  legitimate_papers: 0
paraphrase_tool_origin: "unknown"
discovery_date: "2024-01-01"
discovered_by: "your-github-handle"
notes: >
  Brief description of context and paraphrase tool if known.
```

### Step 3 — Run local validation

```bash
tpc validate-registry
```

This checks schema compliance. CI will run the full precision/recall gates.

### Step 4 — Open a Pull Request

Title: `[Signal] Add TP-XXXX: "<tortured>" → "<canonical>"`

A maintainer will:
1. Assign a domain expert for terminological warrant review
2. Monitor CI results for statistical warrant gates
3. Merge once all three warrant types are satisfied (status → `confirmed`)

---

## Warrant Gate Requirements

| Gate | Criterion | Who checks |
|---|---|---|
| Literary | ≥3 independent corpus sightings | Proposer + CI |
| Terminological | ISO 704 criteria satisfied | Domain expert reviewer |
| Statistical | precision ≥ 0.95 on clean corpus; recall ≥ 0.70 on retracted corpus | CI automation |

A signal **cannot** reach `confirmed` status without all three gates passing.

---

## Contributing Code

- Open an issue describing the change before writing code
- Follow existing code style (ruff + black, type hints throughout)
- All new modules require unit tests in `tests/`
- Run `pytest` before opening a PR

## Code of Conduct

This project is committed to open, respectful collaboration in service of scientific integrity.
