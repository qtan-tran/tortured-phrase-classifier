"""
Registry Growth Analysis (RQ3)
================================
Tests whether registry growth reflects and tracks terminological drift
in the scientific literature over time.

Hjørland's prediction: if the literary warrant corpus used by KO systems
is polluted, KO systems will propagate the distortion. This module
operationalizes that prediction by testing whether registry growth is
a leading indicator of retraction waves.

Addresses RQ3: "Does registry growth reflect and track the emergence
of terminological drift in the scientific literature over time?"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


def load_registry_timeline(signals_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all signals with discovery dates for temporal analysis.

    Returns:
        DataFrame with columns: signal_id, domain, discovery_date,
        status, retracted_papers, warrant_literary, warrant_terminological,
        warrant_statistical
    """
    import yaml
    sig_dir = signals_dir or (
        Path(__file__).parent.parent.parent / "signals" / "phrases"
    )
    records = []
    for f in sorted(sig_dir.rglob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text())
            w    = data.get("warrant", {})
            records.append({
                "signal_id":           data["id"],
                "tortured":            data["tortured"],
                "canonical":           data["canonical"],
                "domain":              data["domain"],
                "status":              data.get("status", "candidate"),
                "discovery_date":      data.get("discovery_date"),
                "retracted_papers":    data.get("prevalence", {})
                                           .get("retracted_papers", 0),
                "warrant_literary":    w.get("literary", {}).get("satisfied", False),
                "warrant_terminological": w.get("terminological", {}).get("satisfied", False),
                "warrant_statistical": w.get("statistical", {}).get("satisfied", False),
                "precision_on_clean":  w.get("statistical", {}).get("precision_on_clean"),
                "recall_on_retracted": w.get("statistical", {}).get("recall_on_retracted"),
            })
        except Exception as e:
            logger.warning("Could not parse %s: %s", f.name, e)

    df = pd.DataFrame(records)
    if "discovery_date" in df.columns:
        df["discovery_date"] = pd.to_datetime(df["discovery_date"], errors="coerce")
    return df.sort_values("discovery_date")


def load_retraction_timeline(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load retraction dates from Retraction Watch public database.
    Filters for paper mill and tortured phrase related retractions.

    Note: Retraction Watch CSV available at https://retractionwatch.com/the-retraction-watch-database/
    Falls back to synthetic data if not available (for reproducibility testing).
    """
    root_dir = Path(__file__).parent.parent.parent
    local_csv = Path(csv_path) if csv_path else (root_dir / "retraction_watch.csv")

    if local_csv.exists():
        try:
            df = pd.read_csv(local_csv)
            if "RetractionDate" not in df.columns:
                raise ValueError("missing required 'RetractionDate' column")

            df["RetractionDate"] = pd.to_datetime(
                df["RetractionDate"], errors="coerce"
            )
            df = df.dropna(subset=["RetractionDate"])

            # Focus on paper-mill related cases when those labels are present.
            reason_series = (
                df.get("Reason", pd.Series("", index=df.index))
                .fillna("")
                .astype(str)
            )
            nature_series = (
                df.get("RetractionNature", pd.Series("", index=df.index))
                .fillna("")
                .astype(str)
            )
            paper_mill_mask = (
                reason_series.str.contains("paper\\s*mill", case=False, regex=True)
                | nature_series.str.contains("paper\\s*mill", case=False, regex=True)
            )

            if paper_mill_mask.any():
                df = df[paper_mill_mask].copy()
                logger.info(
                    "Loaded %d paper-mill retractions from %s",
                    len(df),
                    local_csv,
                )
            else:
                logger.info(
                    "Loaded %d retractions from %s (no explicit paper-mill labels found)",
                    len(df),
                    local_csv,
                )

            return df.sort_values("RetractionDate")
        except Exception as e:
            logger.warning("Could not parse %s: %s", local_csv, e)

    import requests
    try:
        # Try the public API
        resp = requests.get(
            "https://api.retractionwatch.com/api/v1/retractiondata",
            params={"reason": "Paper Mill", "format": "json"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)
            df["RetractionDate"] = pd.to_datetime(df.get("RetractionDate"), errors="coerce")
            return df.dropna(subset=["RetractionDate"])
    except Exception:
        pass

    # Fallback: synthetic retraction timeline based on published statistics
    logger.warning(
        "Retraction Watch API unavailable. Using synthetic timeline. "
        "Download the database from https://retractionwatch.com for real analysis."
    )
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="ME")
    # Approximate cumulative retractions based on published figures
    base  = 50
    trend = np.linspace(0, 3000, len(dates))
    noise = np.random.RandomState(42).normal(0, 50, len(dates))
    cumulative = np.maximum(0, base + trend + noise)
    return pd.DataFrame({
        "RetractionDate": dates,
        "cumulative_retractions": cumulative.cumsum(),
        "source": "synthetic",
    })


def plot_registry_vs_retractions(
    signals_dir:  Optional[Path] = None,
    retractions_csv: Optional[Path] = None,
    output_path:  str = "figures/fig_rq3_registry_growth.pdf",
    show:         bool = False,
) -> pd.DataFrame:
    """
    Figure for paper Section 6 (RQ3):
    Registry growth as a leading indicator of terminological drift emergence.

    Tests Hjørland's prediction that literary warrant corpus pollution
    propagates into KO system outputs.

    Args:
        signals_dir: Path to signal YAML files
        output_path: PDF output path for the figure
        show:        If True, display plot interactively

    Returns:
        Combined DataFrame with monthly registry and retraction data.
    """
    reg_df = load_registry_timeline(signals_dir)
    ret_df = load_retraction_timeline(retractions_csv)

    # Compute cumulative monthly registry size (confirmed signals only)
    confirmed = reg_df[reg_df["status"] == "confirmed"].copy()

    if confirmed["discovery_date"].isna().all():
        logger.warning("No discovery dates in registry — using synthetic dates")
        confirmed["discovery_date"] = pd.date_range(
            "2021-01-01", periods=len(confirmed), freq="ME"
        )

    reg_monthly = (
        confirmed.set_index("discovery_date")
        .resample("ME").size()
        .cumsum()
        .rename("registry_size")
    )

    # Compute cumulative monthly retractions
    if "RetractionDate" in ret_df.columns:
        ret_monthly = (
            ret_df.set_index("RetractionDate")
            .resample("ME").size()
            .cumsum()
            .rename("cumulative_retractions")
        )
    else:
        ret_monthly = ret_df.set_index("RetractionDate")["cumulative_retractions"]

    combined = pd.concat([reg_monthly, ret_monthly], axis=1).ffill()

    # Pearson correlation at lag 0 and lag 3 months (RQ3 analysis)
    if len(combined.dropna()) > 10:
        r_lag0 = combined["registry_size"].corr(combined["cumulative_retractions"])
        combined_lag3 = combined.copy()
        combined_lag3["registry_lagged"] = combined_lag3["registry_size"].shift(3)
        r_lag3 = combined_lag3["registry_lagged"].corr(combined_lag3["cumulative_retractions"])
        logger.info("Pearson r (lag=0): %.3f | (lag=3mo): %.3f", r_lag0, r_lag3)
        combined.attrs["pearson_r_lag0"] = r_lag0
        combined.attrs["pearson_r_lag3"] = r_lag3

    # --- Plot ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        combined.index, combined["registry_size"],
        color="#2166ac", linewidth=2.5,
        label="Registry size (confirmed signals)",
    )
    ax2.plot(
        combined.index, combined["cumulative_retractions"],
        color="#d73027", linewidth=2.5, linestyle="--",
        label="Cumulative paper mill retractions",
    )

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Registry size (confirmed signals)", color="#2166ac", fontsize=11)
    ax2.set_ylabel("Cumulative retractions (paper mills)", color="#d73027", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30)

    plt.title(
        "Figure: Registry growth as indicator of terminological drift emergence\n"
        "(RQ3: does the warrant-based registry track the literature's drift trajectory?)",
        fontsize=11
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    logger.info("Figure saved: %s", output_path)
    return combined


def export_appendix_e(combined: pd.DataFrame, output_path: str) -> None:
    """
    Export Appendix E: Registry Growth vs. Retraction Timeline data table.
    Columns: year_month, cumulative_confirmed_signals, cumulative_retractions,
             pearson_r_lag_0, pearson_r_lag_3months
    """
    df = combined.copy().reset_index()
    df.columns = [c if c != "index" else "year_month" for c in df.columns]
    df["year_month"] = df["year_month"].dt.strftime("%Y-%m")
    df["pearson_r_lag0"] = combined.attrs.get("pearson_r_lag0", float("nan"))
    df["pearson_r_lag3"] = combined.attrs.get("pearson_r_lag3", float("nan"))
    df.to_csv(output_path, index=False)
    logger.info("Appendix E exported: %s", output_path)
