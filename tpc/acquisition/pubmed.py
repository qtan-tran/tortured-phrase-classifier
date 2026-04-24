"""
PubMed Data Acquisition
========================
Fetches retracted paper abstracts from PubMed for corpus construction.
Provides the positive (tortured) training corpus for evaluation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_RATE_LIMIT = 0.4  # seconds between requests (NCBI: max 3/sec without key)


def fetch_retracted_abstracts(
    max_results:     int = 5000,
    reasons:         Optional[list[str]] = None,
    api_key:         Optional[str] = None,
    batch_size:      int = 200,
) -> list[dict]:
    """
    Fetch abstracts of retracted papers from PubMed.
    These form the positive (tortured phrase) training corpus.

    Args:
        max_results: Maximum number of papers to retrieve
        reasons:     Filter by retraction reason keywords (None = all retractions)
        api_key:     NCBI API key (increases rate limit to 10 req/sec)
        batch_size:  Papers per API request

    Returns:
        List of paper dicts with pmid, title, abstract, label fields.
    """
    search_term = "retracted publication[pt]"
    if reasons:
        reason_clause = " OR ".join(f'"{r}"[ti]' for r in reasons)
        search_term = f"({search_term}) AND ({reason_clause})"

    params: dict = {
        "db": "pubmed",
        "term": search_term,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    logger.info("Searching PubMed: %s (max_results=%d)", search_term, max_results)
    resp = requests.get(PUBMED_ESEARCH, params=params, timeout=30)
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]
    logger.info("Found %d PMIDs", len(ids))

    papers = []
    for batch_start in range(0, len(ids), batch_size):
        batch = ids[batch_start:batch_start + batch_size]
        papers.extend(_fetch_batch(batch, api_key=api_key))
        logger.debug("Fetched %d/%d papers", len(papers), len(ids))
        time.sleep(PUBMED_RATE_LIMIT)

    logger.info("Retrieved %d abstracts", len(papers))
    return papers


def fetch_clean_abstracts(
    domains:         Optional[list[str]] = None,
    max_results:     int = 5000,
    api_key:         Optional[str] = None,
    exclude_retracted: bool = True,
) -> list[dict]:
    """
    Fetch clean (non-retracted) paper abstracts for the negative corpus.
    Domain-matched to the positive corpus to avoid domain-specific false positives.

    Args:
        domains: MeSH domain terms to filter by (None = all biomedical)
        max_results: Maximum papers
        api_key: NCBI API key
        exclude_retracted: Explicitly exclude retracted papers

    Returns:
        List of paper dicts with label='clean'.
    """
    search_parts = ["hasabstract"]
    if exclude_retracted:
        search_parts.append("NOT retracted publication[pt]")
    if domains:
        domain_clause = " OR ".join(f'"{d}"[MeSH]' for d in domains)
        search_parts.append(f"({domain_clause})")

    search_term = " AND ".join(search_parts)
    params: dict = {
        "db": "pubmed",
        "term": search_term,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(PUBMED_ESEARCH, params=params, timeout=30)
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]

    papers = []
    for batch_start in range(0, len(ids), 200):
        batch = ids[batch_start:batch_start + 200]
        raw = _fetch_batch(batch, api_key=api_key)
        for p in raw:
            p["label"] = "clean"
        papers.extend(raw)
        time.sleep(PUBMED_RATE_LIMIT)

    return papers


def _fetch_batch(
    pmids: list[str],
    api_key: Optional[str] = None,
) -> list[dict]:
    """Fetch a batch of PMIDs and parse XML response."""
    import xml.etree.ElementTree as ET

    params: dict = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(PUBMED_EFETCH, params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el     = article.find(".//PMID")
        title_el    = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        journal_el  = article.find(".//Journal/Title")
        year_el     = article.find(".//PubDate/Year")

        if abstract_el is None or not (abstract_el.text or "").strip():
            continue

        papers.append({
            "pmid":     pmid_el.text if pmid_el is not None else "",
            "title":    (title_el.text or "") if title_el is not None else "",
            "abstract": abstract_el.text or "",
            "journal":  (journal_el.text or "") if journal_el is not None else "",
            "year":     (year_el.text or "") if year_el is not None else "",
            "label":    "retracted",
            "source":   "pubmed",
        })

    return papers
