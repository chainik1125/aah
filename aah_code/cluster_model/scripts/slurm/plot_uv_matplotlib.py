#!/usr/bin/env python3
"""Render merged UV sweep results to SVG using Matplotlib."""

import argparse
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create static SVG plots for merged UV sweeps")
    parser.add_argument("--merged", required=True, help="Path to merged pickle produced by merge_uv_results.py")
    parser.add_argument(
        "--output-dir",
        help="Directory for plot artifacts (defaults to <merged_dir>/plots)",
    )
    parser.add_argument(
        "--filename-prefix",
        default="sep_uv_matplotlib",
        help="Prefix for the generated SVG filename",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for rasterization fallback (default: 150)",
    )
    return parser.parse_args()


def _get_cluster_series(cluster_block, target_v):
    if cluster_block is None:
        raise ValueError("Missing cluster energies for specified Nc")
    if str(target_v) in cluster_block:
        return np.asarray(cluster_block[str(target_v)], dtype=float)
    if target_v in cluster_block:
        return np.asarray(cluster_block[target_v], dtype=float)
    for key, values in cluster_block.items():
        try:
            if float(key) == float(target_v):
                return np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"No cluster data for V={target_v}")


def main():
    args = parse_args()
    merged_path = Path(args.merged).expanduser().resolve()
    if not merged_path.exists():
        raise SystemExit(f"Merged results not found: {merged_path}")

    with merged_path.open("rb") as fh:
        merged = pickle.load(fh)

    cluster_sizes = [int(Nc) for Nc in merged.get("cluster_sizes", [])]
    u_values = [float(u) for u in merged.get("U_values", [])]
    v_values = [float(v) for v in merged.get("V_values", [])]
    cluster_energies = merged.get("cluster_energies", {})
    idmrg_grid = np.asarray(merged.get("idmrg_energies", []), dtype=float)
    finite_dmrg_grid = np.asarray(merged.get("finite_dmrg_energies", []), dtype=float)
    if not cluster_sizes or not u_values or not v_values or not cluster_energies:
        raise SystemExit("Merged payload missing cluster_sizes/U_values/V_values/cluster_energies")

    if idmrg_grid.size and idmrg_grid.shape != (len(u_values), len(v_values)):
        raise SystemExit(
            f"idmrg_energies grid has shape {idmrg_grid.shape}, expected {(len(u_values), len(v_values))}"
        )
    if finite_dmrg_grid.size and finite_dmrg_grid.shape != (len(u_values), len(v_values)):
        raise SystemExit(
            "finite_dmrg_energies grid shape mismatch: "
            f"{finite_dmrg_grid.shape} vs {(len(u_values), len(v_values))}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else merged_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{args.filename_prefix}.svg"

    n_cols = min(3, len(v_values))
    n_rows = int(math.ceil(len(v_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, V in enumerate(v_values):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.set_title(f"V = {V:g}")
        for ci, Nc in enumerate(cluster_sizes):
            block = cluster_energies.get(str(Nc)) or cluster_energies.get(Nc)
            series = _get_cluster_series(block, V)
            color = color_cycle[ci % len(color_cycle)] if color_cycle else None
            ax.plot(u_values, series, marker="o", label=f"Nc={Nc}", color=color)

        v_idx = idx  # parallel order with v_values
        if idmrg_grid.size:
            idmrg_series = idmrg_grid[:, v_idx]
            if not np.all(np.isnan(idmrg_series)):
                ax.plot(
                    u_values,
                    idmrg_series,
                    linestyle="--",
                    linewidth=2.0,
                    color="black",
                    label="iDMRG",
                )
        if finite_dmrg_grid.size:
            finite_series = finite_dmrg_grid[:, v_idx]
            if not np.all(np.isnan(finite_series)):
                ax.plot(
                    u_values,
                    finite_series,
                    linestyle=":",
                    linewidth=2.0,
                    color="tab:red",
                    label="finite DMRG",
                )
        ax.set_xlabel("U")
        ax.set_ylabel("Energy per site")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize="small")

    # Hide unused subplots if V_values does not fill grid
    total_axes = n_rows * n_cols
    for k in range(len(v_values), total_axes):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(svg_path, format="svg", dpi=args.dpi)
    print(f"Saved Matplotlib SVG to {svg_path}")


if __name__ == "__main__":
    main()
