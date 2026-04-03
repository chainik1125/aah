"""
Registry of all paper figures, loaded from figures.csv.

The CSV is the source of truth — edit it directly, and the markdown
table in used_data.md stays in sync (just open it).

Usage:
    from aah_code.currently_used_data.figure_registry import load_figures, Status

    figures = load_figures()
    todo = {k: v for k, v in figures.items() if v["status"] != Status.DONE}
"""

import csv
from enum import Enum
from pathlib import Path

_DIR = Path(__file__).resolve().parent
CSV_PATH = _DIR / "figures.csv"
MD_PATH = _DIR / "used_data.md"

LARGE_FILES = _DIR.parent / "cluster_model" / "large_files"
PAPER_PLOTS = LARGE_FILES / "paper_plots"


class Status(Enum):
    DONE = "done"
    NEEDS_DATA = "needs_data"
    NEEDS_PLOT = "needs_plot"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


def load_figures():
    """Load the FIGURES dict from figures.csv."""
    figures = {}
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key = row["key"]

            # Resolve data file paths relative to large_files/
            data_paths = []
            if row["data_files"].strip():
                for p in row["data_files"].split("; "):
                    data_paths.append(LARGE_FILES / p.strip())

            # Resolve output PDF relative to paper_plots/
            output_pdf = PAPER_PLOTS / row["output_pdf"] if row["output_pdf"].strip() else None

            figures[key] = {
                "parent_figure": row["parent_figure"],
                "paper_ref": row["paper_ref"],
                "description": row["description"],
                "status": Status(row["status"]),
                "data_files": data_paths,
                "parameters": row["parameters"],
                "run_func": row["run_func"] or None,
                "cli_cmd": row["cli_cmd"] or None,
                "merge_func": row["merge_func"] or None,
                "plot_func": row["plot_func"] or None,
                "output_pdf": output_pdf,
                "tex_name": row["tex_name"] or None,
                "notes": row["notes"],
            }
    return figures


def _check(val):
    return "Y" if val else "-"


def generate_md():
    """Generate a markdown string from the CSV (used by used_data.md rendering)."""
    figures = load_figures()

    # Group by parent figure
    from collections import OrderedDict
    groups = OrderedDict()
    for key, fig in figures.items():
        parent = fig["parent_figure"]
        if parent not in groups:
            groups[parent] = []
        groups[parent].append((key, fig))

    lines = [
        "## Figure Status",
        "",
        "Source of truth: [`figures.csv`](figures.csv) — edit that file directly.",
        "",
    ]

    for parent, entries in groups.items():
        lines.append(f"### {parent}")
        lines.append("")
        lines.append("| Paper Ref | Description | Data | PDF | Status | Data func | Plot func | Notes |")
        lines.append("|-----------|-------------|------|-----|--------|-----------|-----------|-------|")

        for key, fig in entries:
            has_data = bool(fig["data_files"]) and all(p.exists() for p in fig["data_files"])
            has_pdf = fig["output_pdf"] is not None and fig["output_pdf"].exists()
            status = fig["status"].value.upper().replace("_", " ")
            run_func = fig["run_func"].split(".")[-1] if fig["run_func"] else "-"
            plot_func = fig["plot_func"].split(".")[-1] if fig["plot_func"] else "-"

            lines.append(
                f"| {fig['paper_ref']} "
                f"| {fig['description'][:55]} "
                f"| {_check(has_data)} | {_check(has_pdf)} "
                f"| {status} "
                f"| {run_func} "
                f"| {plot_func} "
                f"| {fig['notes']} |"
            )
        lines.append("")

    lines += [
        "",
        "## Programmatic access",
        "",
        "```python",
        "from aah_code.currently_used_data.figure_registry import load_figures, Status",
        "figures = load_figures()",
        "todo = {k: v for k, v in figures.items() if v['status'] != Status.DONE}",
        "```",
        "",
        "## Key locations",
        "",
        "- **Run functions**: `aah_code/cluster_model/plots.py`",
        "- **CLI wrappers**: `aah_code/cluster_model/cluster_runner.py`",
        "- **Merge functions**: `aah_code/cluster_model/merge_comparison_results.py`",
        "- **Plot functions**: `aah_code/cluster_model/large_files/paper_plots/make_paper_plots.py`",
        "- **Data pickles**: `aah_code/cluster_model/large_files/plots/` and `large_files/partials/`",
        "- **Output PDFs**: `aah_code/cluster_model/large_files/paper_plots/`",
        "",
    ]
    return "\n".join(lines)


def update_md():
    """Write the markdown table to used_data.md."""
    MD_PATH.write_text(generate_md())
    print(f"Updated {MD_PATH}")


def summary():
    """Print a status summary table."""
    figures = load_figures()
    header = f"{'Key':<30} {'Paper Ref':<25} {'Status':<18} {'Has Data':<10}"
    print(header)
    print("-" * len(header))
    for key, fig in figures.items():
        has_data = bool(fig["data_files"]) and all(p.exists() for p in fig["data_files"])
        print(f"{key:<30} {fig['paper_ref']:<25} {fig['status'].value:<18} {'yes' if has_data else 'NO':<10}")


if __name__ == "__main__":
    update_md()
    print()
    summary()
