from pathlib import Path

from tpc.evaluation.registry_growth import plot_registry_vs_retractions


def main() -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "retraction_watch.csv"

    combined = plot_registry_vs_retractions(
        retractions_csv=csv_path if csv_path.exists() else None,
        output_path=str(project_root / "figures" / "fig_rq3_registry_growth.pdf"),
        show=True,
    )
    print(combined)


if __name__ == "__main__":
    main()