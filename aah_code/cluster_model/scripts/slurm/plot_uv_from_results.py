#!/usr/bin/env python3
"""Plot merged UV sweep results using compare_U_values_with_dmrg."""

import argparse
import pickle
from pathlib import Path

from aah_code.cluster_model.plots import compare_U_values_with_dmrg


def _as_ratio(seq, label):
    if seq is None:
        raise ValueError(f"Missing {label} in merged results.")
    if isinstance(seq, (list, tuple)) and len(seq) == 2:
        return int(seq[0]), int(seq[1])
    raise ValueError(f"{label} must be a length-2 sequence, got {seq!r}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot merged UV sweep results")
    parser.add_argument("--merged", required=True, help="Path to merged pickle produced by merge_uv_results.py")
    parser.add_argument(
        "--output-dir",
        help="Directory for plot artifacts (defaults to <merged_dir>/plots)",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display Plotly window (off by default)")
    parser.add_argument("--no-html", action="store_true", help="Skip saving the interactive HTML figure")
    parser.add_argument("--no-data", action="store_true", help="Skip saving a pickle of the plotted data")
    parser.add_argument("--filename-prefix", default="U_value_energy_comparison", help="Prefix for saved files")
    return parser.parse_args()


def main():
    args = parse_args()
    merged_path = Path(args.merged).expanduser().resolve()
    if not merged_path.exists():
        raise SystemExit(f"Merged results not found: {merged_path}")

    with merged_path.open("rb") as fh:
        merged = pickle.load(fh)

    params = merged.get("parameters", {})
    cluster_sizes = merged.get("cluster_sizes")
    u_values = merged.get("U_values")
    v_values = merged.get("V_values")
    if not cluster_sizes or not u_values or not v_values:
        raise SystemExit("Merged payload missing cluster_sizes/U_values/V_values")

    v_sep_ratio = params.get("v_sep_ratio") or merged.get("v_sep_ratio")
    v_sep_ratio = _as_ratio(v_sep_ratio, "v_sep_ratio")
    int_sep_ratios = merged.get("int_sep_ratios") or params.get("int_sep_ratios")
    if int_sep_ratios is None:
        raise SystemExit("Merged payload missing int_sep_ratios")

    output_dir = Path(args.output_dir) if args.output_dir else merged_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    include_idmrg = bool(params.get("include_idmrg", True))
    include_finite_dmrg = bool(params.get("include_finite_dmrg", True))

    compare_U_values_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_ratios=int_sep_ratios,
        cluster_sizes=cluster_sizes,
        U_values=u_values,
        V_values=v_values,
        t=float(params.get("t", 1.0)),
        L=int(params.get("L", 20)),
        chi=int(params.get("chi", 32)),
        solver_method=params.get("solver_method", "sparse_ED"),
        states_retained=int(params.get("states_retained", 4)),
        output_dir=str(output_dir),
        show_plots=args.show_plots,
        save_html=not args.no_html,
        save_data=not args.no_data,
        filename_prefix=args.filename_prefix,
        log_yaxis=True,
        include_idmrg=include_idmrg,
        include_finite_dmrg=include_finite_dmrg,
        include_timing=bool(params.get("include_timing", False)),
        include_timing_plot=bool(params.get("include_timing_plot", False)),
        results=str(merged_path),
    )


if __name__ == "__main__":
    main()
