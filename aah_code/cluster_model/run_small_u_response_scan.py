import contextlib
import io
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_density_wave_observable
from aah_code.real_space_dmrg import get_finite_dmrg_density_wave_observable


logging.getLogger().setLevel(logging.ERROR)


def signed_component(profile: np.ndarray, q_ratio: tuple[int, int]) -> float:
    p, q = q_ratio
    Q = 2.0 * np.pi * p / q
    j = np.arange(len(profile), dtype=float)
    return float(np.mean(np.cos(Q * j) * profile))


def central_difference(m_plus: float, m_minus: float, eps: float) -> float:
    return (m_plus - m_minus) / (2.0 * eps)


def run_scan():
    L = 20
    chi = 32
    t = 1.0
    filling = 1.0
    eps = 0.05
    u_values = [0.5, 1.0, 2.0, 4.0]

    schemes = [
        {
            "label": r"$N_c=2,\ \Delta_{\rm int}=\pi,\ Q=\pi$",
            "tag": "Nc2_int12_v12",
            "Nc": 2,
            "int_sep": (1, 2),
            "v_sep": (1, 2),
        },
        {
            "label": r"$N_c=4,\ \Delta_{\rm int}=\pi/2,\ Q=\pi$",
            "tag": "Nc4_int14_v12",
            "Nc": 4,
            "int_sep": (1, 4),
            "v_sep": (1, 2),
        },
        {
            "label": r"$N_c=4,\ \Delta_{\rm int}=\pi/2,\ Q=\pi/2$",
            "tag": "Nc4_int14_v14",
            "Nc": 4,
            "int_sep": (1, 4),
            "v_sep": (1, 4),
        },
    ]

    cluster_cache: dict[tuple, float] = {}
    dmrg_cache: dict[tuple, float] = {}
    results: list[dict] = []

    for scheme in schemes:
        q_ratio = scheme["v_sep"]
        series = {
            "scheme": scheme,
            "U_values": [],
            "chi_cluster": [],
            "chi_dmrg": [],
            "abs_diff": [],
            "rel_diff": [],
        }
        for U in u_values:
            mu0 = U / 2.0

            for V in (-eps, eps):
                ckey = (scheme["Nc"], scheme["int_sep"], q_ratio, U, V)
                if ckey not in cluster_cache:
                    cfg = ClusterModelConfig(
                        L=L,
                        int_cluster_size=scheme["Nc"],
                        cluster_separation_ratio=scheme["int_sep"],
                        V_separation_ratio=q_ratio,
                        physical_params=PhysicalParams(U=U, mu_0=mu0, V=V, t=t),
                        ham_lib="quspin",
                        solver_method="dense_ED",
                        states_retained="all",
                    )
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        _, profile = get_general_density_wave_observable(
                            cfg,
                            set_filling=filling,
                            temperature=1e-8,
                            return_profile=True,
                        )
                    cluster_cache[ckey] = signed_component(np.asarray(profile, dtype=float), q_ratio)

                dkey = (q_ratio, U, V)
                if dkey not in dmrg_cache:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        _, profile = get_finite_dmrg_density_wave_observable(
                            L,
                            chi,
                            U,
                            t,
                            0.0,
                            V,
                            q_ratio,
                            filling_target=filling,
                            dmrg_fixed_filling=True,
                            return_profile=True,
                        )
                    dmrg_cache[dkey] = signed_component(np.asarray(profile, dtype=float), q_ratio)

            c_minus = cluster_cache[(scheme["Nc"], scheme["int_sep"], q_ratio, U, -eps)]
            c_plus = cluster_cache[(scheme["Nc"], scheme["int_sep"], q_ratio, U, eps)]
            d_minus = dmrg_cache[(q_ratio, U, -eps)]
            d_plus = dmrg_cache[(q_ratio, U, eps)]

            chi_cluster = central_difference(c_plus, c_minus, eps)
            chi_dmrg = central_difference(d_plus, d_minus, eps)
            abs_diff = abs(chi_cluster - chi_dmrg)
            rel_diff = abs_diff / abs(chi_dmrg)

            series["U_values"].append(U)
            series["chi_cluster"].append(chi_cluster)
            series["chi_dmrg"].append(chi_dmrg)
            series["abs_diff"].append(abs_diff)
            series["rel_diff"].append(rel_diff)

        results.append(series)

    output_dir = Path("large_files/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    q_groups = {
        (1, 2): [res for res in results if res["scheme"]["v_sep"] == (1, 2)],
        (1, 4): [res for res in results if res["scheme"]["v_sep"] == (1, 4)],
    }
    titles = {
        (1, 2): r"$Q=\pi$",
        (1, 4): r"$Q=\pi/2$",
    }
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for col_idx, q_ratio in enumerate([(1, 2), (1, 4)]):
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]
        group = q_groups[q_ratio]
        if not group:
            continue

        dmrg_reference = group[0]
        ax_top.plot(
            dmrg_reference["U_values"],
            dmrg_reference["chi_dmrg"],
            color="black",
            marker="x",
            linestyle="--",
            label="Finite DMRG",
        )

        for color, series in zip(colors, group):
            ax_top.plot(
                series["U_values"],
                series["chi_cluster"],
                color=color,
                marker="o",
                label=series["scheme"]["label"],
            )
            ax_bottom.plot(
                series["U_values"],
                series["rel_diff"],
                color=color,
                marker="o",
                label=series["scheme"]["label"],
            )

        ax_top.set_title(titles[q_ratio])
        ax_top.set_ylabel(r"$\chi_Q$")
        ax_bottom.set_ylabel("Rel. error")
        ax_bottom.set_xlabel(r"$U$")
        ax_bottom.set_ylim(bottom=0.0)
        ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100*x:.0f}%"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(r"Finite-difference response benchmark at half filling, $\epsilon=0.05$")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    png_path = output_dir / "small_u_response_scan.png"
    fig.savefig(png_path, dpi=180)

    payload_path = output_dir / "small_u_response_scan.pkl"
    with payload_path.open("wb") as fh:
        pickle.dump(
            {
                "parameters": {
                    "L": L,
                    "chi": chi,
                    "t": t,
                    "filling": filling,
                    "eps": eps,
                    "U_values": u_values,
                },
                "results": results,
                "artifacts": {
                    "png": str(png_path),
                },
            },
            fh,
        )

    for series in results:
        print(series["scheme"]["label"])
        for U, chi_cluster, chi_dmrg, abs_diff, rel_diff in zip(
            series["U_values"],
            series["chi_cluster"],
            series["chi_dmrg"],
            series["abs_diff"],
            series["rel_diff"],
        ):
            print(
                f"  U={U:.2f}: chi_cluster={chi_cluster:.6f}, chi_dmrg={chi_dmrg:.6f}, "
                f"abs_diff={abs_diff:.6f}, rel_diff={100.0*rel_diff:.2f}%"
            )

    print(f"Saved plot to {png_path}")
    print(f"Saved payload to {payload_path}")


if __name__ == "__main__":
    run_scan()
