"""
TPC Command-Line Interface
===========================
Typer-based CLI for screening manuscripts, validating the registry,
and running benchmark evaluation.

Usage:
  tpc screen paper.pdf
  tpc screen paper.pdf --layers exact --report --output report.json
  tpc validate-registry
  tpc registry-stats
  tpc evaluate --corpus data/ground_truth/
  tpc serve                    # start FastAPI server
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app     = typer.Typer(help="Tortured Phrase Classifier — pre-publication screening tool")
console = Console()

logging.basicConfig(level=logging.WARNING)


@app.command()
def screen(
    input_path: Path = typer.Argument(..., help="PDF file or text file to screen"),
    layers:     str  = typer.Option("exact,embedding,mlm",
                                    help="Comma-separated layers: exact,embedding,mlm"),
    report:     bool = typer.Option(False, "--report", help="Generate full evidence report"),
    output:     Optional[Path] = typer.Option(None, "--output", "-o",
                                              help="Save report JSON to this path"),
    verbose:    bool = typer.Option(False, "--verbose", "-v"),
):
    """Screen a manuscript (PDF or TXT) for tortured phrases."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)

    # Extract text
    if input_path.suffix.lower() == ".pdf":
        text = _extract_pdf(input_path)
    else:
        text = input_path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        console.print("[red]No text could be extracted.[/red]")
        raise typer.Exit(1)

    # Run pipeline
    active_layers = tuple(l.strip() for l in layers.split(","))
    from tpc.pipeline import TorturedPhraseClassifier
    clf    = TorturedPhraseClassifier(layers=active_layers)
    result = clf.classify(text)

    # Display results
    _display_result(result)

    if report or output:
        from tpc.report import build_report
        full_report = build_report(
            result=result,
            text=text,
            metadata={"source": str(input_path)},
            output_path=str(output) if output else None,
        )
        if output:
            console.print(f"\n[green]Report saved:[/green] {output}")
        elif report:
            rprint(full_report)


@app.command("validate-registry")
def validate_registry(
    signals_dir: Optional[Path] = typer.Argument(None),
    strict:      bool = typer.Option(False, "--strict",
                                     help="Fail on unconfirmed (candidate) signals"),
):
    """Validate all signals in the YAML registry against warrant gates."""
    from tpc.registry.validator import validate_registry as _validate

    sig_dir = signals_dir or (Path(__file__).parent.parent / "signals" / "phrases")
    console.print(f"Validating registry at [cyan]{sig_dir}[/cyan]")

    valid, messages = _validate(sig_dir, strict=strict)

    n_signals = len(list(sig_dir.rglob("*.yaml")))

    if messages:
        for msg in messages:
            colour = "red" if not valid else "yellow"
            console.print(f"  [{colour}]{msg}[/{colour}]")

    if valid:
        console.print(f"\n[green]Registry valid[/green] ({n_signals} signals)")
        raise typer.Exit(0)
    else:
        console.print(f"\n[red]Registry invalid - {len(messages)} error(s)[/red]")
        raise typer.Exit(1)


@app.command("registry-stats")
def registry_stats(
    include_candidates: bool = typer.Option(False, "--candidates"),
):
    """Print registry summary statistics (answers RQ1)."""
    from tpc.registry.loader import load_registry, registry_summary

    status = ("confirmed", "candidate") if include_candidates else ("confirmed",)
    signals = load_registry(status_filter=status)
    summary = registry_summary(signals)

    table = Table(title="Signal Registry Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total signals",      str(summary["total"]))
    table.add_row("By status",          str(summary["by_status"]))
    table.add_row("By domain",          str(summary["by_domain"]))
    table.add_row("Mean precision",     f"{summary['mean_precision']:.4f}")
    table.add_row("Mean recall",        f"{summary['mean_recall']:.4f}")
    table.add_row("Total sightings",    str(summary["total_sightings"]))
    table.add_row("Literary warrant %", f"{summary['warrant_pass_rates']['literary']:.1%}")
    table.add_row("Terminological %",   f"{summary['warrant_pass_rates']['terminological']:.1%}")
    table.add_row("Statistical %",      f"{summary['warrant_pass_rates']['statistical']:.1%}")

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
):
    """Start the FastAPI pre-publication screening server."""
    import uvicorn
    console.print(f"Starting TPC API server at [cyan]http://{host}:{port}[/cyan]")
    console.print("API docs: [cyan]http://localhost:8000/docs[/cyan]")
    uvicorn.run("tpc.api:app", host=host, port=port, reload=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        console.print("[yellow]pymupdf not installed — treating as text file[/yellow]")
        return path.read_text(encoding="utf-8", errors="replace")


def _display_result(result) -> None:
    from rich.panel import Panel

    colour = {
        "low":      "green",
        "medium":   "yellow",
        "high":     "red",
        "critical": "bold red",
    }.get(result.risk_level, "white")

    console.print(Panel(
        f"[{colour}]Risk Level: {result.risk_level.upper()}[/{colour}]\n"
        f"Risk Score: {result.risk_score:.3f}\n"
        f"{result.summary}",
        title="TPC Screening Result",
        expand=False,
    ))

    if result.hits:
        table = Table(title="Detected Suspicious Spans", show_lines=True)
        table.add_column("Layer",     style="cyan",  width=18)
        table.add_column("Tortured",  style="red",   width=25)
        table.add_column("Canonical", style="green", width=20)
        table.add_column("Conf.",     width=6)
        table.add_column("Context",   width=40)

        for h in result.hits[:20]:
            table.add_row(
                h.get("layer", ""),
                h.get("tortured", "")[:24],
                h.get("canonical") or "(unknown)",
                f"{h.get('confidence', 0):.2f}",
                (h.get("context") or "")[:40],
            )
        console.print(table)


if __name__ == "__main__":
    app()
