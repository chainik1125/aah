import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aah_code.real_space_dmrg import get_dmrg_static_structure_factors


def q_label(q: float) -> str:
    if abs(q) < 1e-12:
        return "0"
    frac = q / np.pi
    return f"{frac:.3g}π"


def top_modes(q_values: np.ndarray, values: np.ndarray, *, exclude_zero: bool = True, count: int = 3):
    mask = np.ones(len(q_values), dtype=bool)
    if exclude_zero:
        mask &= np.abs(q_values) > 1e-12
    idx = np.where(mask)[0]
    order = idx[np.argsort(values[idx])[::-1]]
    return [(float(q_values[i]), float(values[i])) for i in order[:count]]


def main():
    L = 32
    chi = 32
    t = 1.0
    V = 0.0
    U_values = [1.0, 2.0, 4.0]
    fillings = [
        {"label": "half", "target": 1.0},
        {"label": "quarter", "target": 0.5},
    ]

    q_values = 2.0 * np.pi * np.arange(L // 2 + 1, dtype=float) / float(L)
    output_dir = Path("large_files/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    fig, axes = plt.subplots(len(fillings), 2, figsize=(10, 6), sharex=True)

    for row_idx, filling_info in enumerate(fillings):
        label = filling_info["label"]
        filling_target = filling_info["target"]
        ax_n = axes[row_idx, 0]
        ax_s = axes[row_idx, 1]

        for U in U_values:
            res = get_dmrg_static_structure_factors(
                L,
                chi,
                U=U,
                t=t,
                V=V,
                V_sep=None,
                filling_target=filling_target,
                dmrg_fixed_filling=True,
                q_values=q_values,
            )
            top_charge = top_modes(res["q_values"], res["Nq"])
            top_spin = top_modes(res["q_values"], res["Sq"])
            results.append(
                {
                    "filling_label": label,
                    "filling_target": filling_target,
                    "U": U,
                    "q_values": res["q_values"].tolist(),
                    "Nq": res["Nq"].tolist(),
                    "Sq": res["Sq"].tolist(),
                    "top_charge_modes": top_charge,
                    "top_spin_modes": top_spin,
                }
            )

            ax_n.plot(res["q_values"] / np.pi, res["Nq"], marker="o", label=f"U={U:g}")
            ax_s.plot(res["q_values"] / np.pi, res["Sq"], marker="o", label=f"U={U:g}")

        ax_n.set_ylabel(f"{label}\nN(q)")
        ax_s.set_ylabel(f"{label}\nS(q)")
        ax_n.set_xlim(0.0, 1.0)
        ax_s.set_xlim(0.0, 1.0)
        ax_n.legend(frameon=False, fontsize=8)
        ax_s.legend(frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel(r"$q/\pi$")
    axes[-1, 1].set_xlabel(r"$q/\pi$")
    fig.suptitle("Finite DMRG static structure factors at V=0")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png_path = output_dir / "dmrg_structure_factor_scan.png"
    fig.savefig(png_path, dpi=180)

    payload_path = output_dir / "dmrg_structure_factor_scan.pkl"
    with payload_path.open("wb") as fh:
        pickle.dump(
            {
                "parameters": {
                    "L": L,
                    "chi": chi,
                    "t": t,
                    "V": V,
                    "U_values": U_values,
                    "fillings": fillings,
                },
                "results": results,
                "artifacts": {
                    "png": str(png_path),
                },
            },
            fh,
        )

    for entry in results:
        print(f"{entry['filling_label']} filling, U={entry['U']:.2f}")
        charge_desc = ", ".join(f"{q_label(q)}:{val:.4f}" for q, val in entry["top_charge_modes"])
        spin_desc = ", ".join(f"{q_label(q)}:{val:.4f}" for q, val in entry["top_spin_modes"])
        print(f"  top N(q): {charge_desc}")
        print(f"  top S(q): {spin_desc}")

    print(f"Saved plot to {png_path}")
    print(f"Saved payload to {payload_path}")


if __name__ == "__main__":
    main()
