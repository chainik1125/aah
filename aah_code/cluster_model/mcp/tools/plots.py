"""Local plotting tools — runs make_paper_plots.py and related functions."""

import subprocess
from pathlib import Path
from typing import Optional


def make_plots(
    project_root: str,
    figures: Optional[list[str]] = None,
) -> dict:
    """
    Run make_paper_plots.py locally to regenerate figures from pickle data.

    Args:
        project_root: Path to the aah project root.
        figures: Which figures to generate. If None or ["all"], generates all.
                 Options: "hub_comparison", "v_convergence", "fixed_supercluster"
    """
    script = (
        Path(project_root)
        / "aah_code"
        / "cluster_model"
        / "large_files"
        / "paper_plots"
        / "make_paper_plots.py"
    )

    if figures and figures != ["all"]:
        # Run specific figure functions
        func_map = {
            "hub_comparison": "plot_hub_comparison",
            "v_convergence": "plot_v_convergence",
            "fixed_supercluster": "plot_fixed_supercluster",
        }
        code_lines = [
            "import sys",
            f"sys.path.insert(0, {str(project_root)!r})",
            f"sys.path.insert(0, {str(Path(project_root) / 'aah_code' / 'cluster_model' / 'large_files' / 'paper_plots')!r})",
            "from make_paper_plots import *",
            "setup_style()",
        ]
        for fig in figures:
            func_name = func_map.get(fig)
            if func_name:
                code_lines.append(f"{func_name}()")

        cmd = [
            "python", "-c", "\n".join(code_lines),
        ]
    else:
        cmd = ["python", str(script)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=120,
    )

    # Find generated files
    paper_plots_dir = script.parent
    generated = []
    if paper_plots_dir.exists():
        generated = sorted(
            str(p) for p in paper_plots_dir.glob("fig_*.*")
        )

    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "generated_files": generated,
    }


def run_local_replot(
    project_root: str,
    pickle_path: str,
    plot_relative_error: bool = True,
    show_plots: bool = False,
) -> dict:
    """
    Reload from an existing pickle and replot with different options.

    Useful for generating raw energy plots from existing relative error data.
    """
    code = f"""
import sys
sys.path.insert(0, {project_root!r})
from aah_code.cluster_model.plots import compare_filling_with_int_cluster_sizes
fig, results = compare_filling_with_int_cluster_sizes(
    int_sep_ratios_by_Nc={{2: [(1,2)]}},  # dummy, overridden by results
    U_values=[0],  # dummy
    plot_relative_error={plot_relative_error},
    show_plots={show_plots},
    save_html=True,
    save_data=False,
    results={pickle_path!r},
)
print("Replot complete")
"""
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=120,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
