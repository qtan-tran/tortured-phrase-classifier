"""
Tortured Phrase Classifier (TPC)
=================================
A warrant-based, three-layer classification system for detecting
tortured phrases as terminological violations in scientific literature.

Theoretical grounding:
- Wüster (1979): General Theory of Terminology — concept-term bonds
- Beghtol (1986): Warrant in classification systems
- Floridi (2011): Semantic information and misinformation
- ISO 704: Terminology work — principles and methods
"""

from typing import Any

__version__ = "0.1.0"
__author__ = "TPC Contributors"
__license__ = "Apache-2.0"

__all__ = ["TorturedPhraseClassifier"]


def __getattr__(name: str) -> Any:
    """Lazily expose heavy imports at package level.

    This keeps lightweight CLI commands (e.g. `tpc validate-registry`) usable
    even when optional runtime dependencies for model inference are not
    installed.
    """
    if name == "TorturedPhraseClassifier":
        from tpc.pipeline import TorturedPhraseClassifier

        return TorturedPhraseClassifier
    raise AttributeError(f"module 'tpc' has no attribute {name!r}")
