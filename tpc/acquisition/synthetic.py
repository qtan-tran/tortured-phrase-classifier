"""
Synthetic Tortured Phrase Generator
=====================================
Generates synthetic training data by simulating paper mill paraphrasing.
Provides procedural warrant for training data: we know ground truth because
we constructed it.

Methods:
  1. WordNet synonym substitution (simulates early paper mill tools)
  2. Back-translation EN→ZH→EN (simulates Chinese-language paraphrasers)
  3. Controlled term substitution using registry signal pairs
"""

from __future__ import annotations

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)


def generate_from_registry(
    clean_sentences: list[str],
    n_samples:       int = 10000,
    seed:            int = 42,
) -> list[dict]:
    """
    Generate synthetic tortured sentences by substituting known canonical
    terms with their registered tortured variants.

    This is the highest-quality synthetic generation method because it
    uses ground-truth signal pairs from the warrant-based registry.

    Args:
        clean_sentences: Legitimate scientific sentences
        n_samples: Number of synthetic examples to generate
        seed: Random seed for reproducibility

    Returns:
        List of dicts with original, tortured, method, signal_id, label fields.
    """
    from tpc.registry.loader import load_registry
    signals = load_registry(status_filter=("confirmed",))

    # Build lookup: canonical → list of (tortured, signal_id) pairs
    canonical_map: dict[str, list[tuple[str, str]]] = {}
    for sig in signals:
        for variant in [sig.tortured] + sig.known_variants:
            canonical_map.setdefault(sig.canonical.lower(), []).append(
                (variant, sig.id)
            )

    random.seed(seed)
    results = []

    for sent in clean_sentences[:n_samples]:
        sent_lower = sent.lower()
        matched_canonical = None

        for canonical, variants in canonical_map.items():
            if canonical in sent_lower:
                matched_canonical = (canonical, variants)
                break

        if matched_canonical is None:
            continue

        canonical_str, variants = matched_canonical
        tortured_form, sig_id   = random.choice(variants)
        tortured_sent = sent_lower.replace(canonical_str, tortured_form)

        results.append({
            "original":     sent,
            "tortured":     tortured_sent,
            "signal_id":    sig_id,
            "canonical":    canonical_str,
            "tortured_phrase": tortured_form,
            "method":       "registry_substitution",
            "label":        "synthetic_tortured",
        })

    logger.info("Generated %d synthetic examples (registry substitution)", len(results))
    return results


def generate_wordnet(
    clean_sentences: list[str],
    substitution_rate: float = 0.15,
    technical_only:    bool  = True,
    seed:              int   = 42,
) -> list[dict]:
    """
    Generate synthetic tortured phrases via WordNet synonym substitution.
    Simulates early paper mill tools that used simple thesaurus replacement.

    Args:
        clean_sentences: Input legitimate scientific sentences
        substitution_rate: Proportion of eligible words to substitute
        technical_only: Only substitute content words (nouns, verbs, adjectives)
        seed: Random seed

    Returns:
        List of (original, tortured) sentence pairs.
    """
    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag

        # NLTK resource names vary across versions (e.g., punkt_tab,
        # averaged_perceptron_tagger_eng). Download a compatibility set.
        for resource in (
            "wordnet",
            "omw-1.4",
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                # Keep going; unavailable aliases are expected on some versions.
                pass
    except ImportError:
        logger.error("nltk not installed. Run: pip install nltk")
        return []

    TARGET_POS = {"NN", "NNS", "VB", "VBD", "VBG", "JJ"}  # content words
    random.seed(seed)
    results = []

    for sent in clean_sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        output = list(tokens)
        tortured_any = False

        for i, (word, pos) in enumerate(tagged):
            if technical_only and pos not in TARGET_POS:
                continue
            if random.random() > substitution_rate:
                continue

            synsets = wordnet.synsets(word)
            if not synsets:
                continue

            # Collect all lemma names across synsets, excluding the original
            synonyms = [
                lemma.name().replace("_", " ")
                for ss in synsets
                for lemma in ss.lemmas()
                if lemma.name().lower() != word.lower()
            ]
            if not synonyms:
                continue

            output[i] = random.choice(synonyms)
            tortured_any = True

        if tortured_any:
            results.append({
                "original": sent,
                "tortured": " ".join(output),
                "method":   "wordnet_substitution",
                "label":    "synthetic_tortured",
                "signal_id": None,
            })

    logger.info("Generated %d synthetic examples (WordNet)", len(results))
    return results


def generate_backtranslation(
    clean_sentences: list[str],
    src_lang:        str = "en",
    pivot_lang:      str = "zh",
    batch_size:      int = 32,
) -> list[dict]:
    """
    Generate synthetic tortured phrases via back-translation (EN→ZH→EN).
    Simulates Chinese-language paraphrasers — the dominant source of real
    tortured phrases in the literature.

    Requires: transformers with Helsinki-NLP/opus-mt models.

    Args:
        clean_sentences: Input legitimate scientific sentences
        src_lang: Source language code
        pivot_lang: Pivot language code (ZH for Chinese paraphrasers)
        batch_size: Sentences per translation batch

    Returns:
        List of (original, backtranslated) pairs.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.error("transformers not installed.")
        return []

    logger.info("Loading translation models (EN→%s→EN)", pivot_lang.upper())

    fwd_model = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
    bwd_model = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"

    translator_fwd = hf_pipeline("translation", model=fwd_model, max_length=512)
    translator_bwd = hf_pipeline("translation", model=bwd_model, max_length=512)

    results = []

    for i in range(0, len(clean_sentences), batch_size):
        batch = clean_sentences[i:i + batch_size]

        try:
            # Forward: EN → pivot
            pivot_texts = [
                r["translation_text"]
                for r in translator_fwd(batch, batch_size=batch_size)
            ]
            # Backward: pivot → EN
            back_texts = [
                r["translation_text"]
                for r in translator_bwd(pivot_texts, batch_size=batch_size)
            ]

            for original, backtranslated in zip(batch, back_texts):
                if backtranslated.lower() != original.lower():
                    results.append({
                        "original":  original,
                        "tortured":  backtranslated,
                        "method":    f"backtranslation_{src_lang}_{pivot_lang}_{src_lang}",
                        "label":     "synthetic_tortured",
                        "signal_id": None,
                    })
        except Exception as e:
            logger.warning("Translation batch failed: %s", e)
            continue

        logger.debug("Back-translated batch %d/%d", i + batch_size, len(clean_sentences))

    logger.info("Generated %d synthetic examples (back-translation)", len(results))
    return results
