import argparse
import json
import os
import pickle
from pathlib import Path

from aah_code.cluster_model.plots import compare_U_values_with_dmrg


def parse_args():
    p = argparse.ArgumentParser(
        description="Run compare_U_values_with_dmrg for a single U,V pair."
    )
    p.add_argument("--params", required=True, help="Path to JSON file with parameters for this task.")
    p.add_argument("--output", required=True, help="Path to write the partial pickle.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.params, "r") as fh:
        params = json.load(fh)

    # Extract required fields with simple defaults for optional flags
    v_sep_ratio = tuple(int(x) for x in params["v_sep_ratio"])
    cluster_sizes = [int(x) for x in params["cluster_sizes"]]
    int_sep_ratios = params["int_sep_ratios"]
    if isinstance(int_sep_ratios, dict):
        norm = {}
        for k, v in int_sep_ratios.items():
            try:
                nk = int(k)
            except Exception:
                nk = k
            norm[nk] = v
        int_sep_ratios = norm
    U = float(params["U"])
    V = float(params["V"])
    t = float(params.get("t", 1.0))
    L = int(params["L"])
    chi = int(params.get("chi", 32))
    solver_method = params.get("solver_method", "sparse_ED")
    states_retained = int(params.get("states_retained", 4))
    include_idmrg = bool(params.get("include_idmrg", True))
    include_finite_dmrg = bool(params.get("include_finite_dmrg", True))
    include_timing = bool(params.get("include_timing", False))
    include_timing_plot = bool(params.get("include_timing_plot", False))

    fig, results = compare_U_values_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_ratios=int_sep_ratios,
        cluster_sizes=cluster_sizes,
        U_values=[U],
        V_values=[V],
        t=t,
        L=L,
        chi=chi,
        solver_method=solver_method,
        states_retained=states_retained,
        include_idmrg=include_idmrg,
        include_finite_dmrg=include_finite_dmrg,
        include_timing=include_timing,
        include_timing_plot=include_timing_plot,
        save_html=False,
        save_data=False,
        show_plots=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(results, fh)

    print(f"Saved partial results for U={U}, V={V} to {out_path}")


if __name__ == "__main__":
    main()
