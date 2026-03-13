import pickle
from pathlib import Path

import numpy as np

from aah_code.cluster_model.clustering import describe_probe_closure, generate_clusters
from aah_code.real_space_dmrg import get_dmrg_static_structure_factors


def q_label_from_step(step: int, L: int) -> str:
    frac = 2.0 * step / float(L)
    if abs(frac) < 1e-12:
        return "0"
    return f"{frac:.3g}π"


def top_modes(q_values: np.ndarray, values: np.ndarray, *, count: int = 4):
    order = np.argsort(values)[::-1]
    selected = []
    seen_steps = set()
    L = 2 * (len(q_values) - 1)
    for idx in order:
        q = float(q_values[idx])
        step = int(round(q * L / (2.0 * np.pi))) % L
        if step == 0 or step in seen_steps:
            continue
        seen_steps.add(step)
        selected.append((step, q, float(values[idx])))
        if len(selected) >= count:
            break
    return selected


def dispersion_spread_summary(L: int, Nc: int, int_sep_ratio, *, t: float):
    clusters = generate_clusters(L, Nc, int_sep_ratio, (1, 1)).reshape(-1, Nc)
    k_vals = -np.pi + (2.0 * np.pi / float(L)) * clusters
    eps = -2.0 * t * np.cos(k_vals)
    spread = np.max(eps, axis=1) - np.min(eps, axis=1)
    return {
        "avg_spread": float(np.mean(spread)),
        "rms_spread": float(np.sqrt(np.mean(spread ** 2))),
        "max_spread": float(np.max(spread)),
    }


def summarize_case(
    *,
    L: int,
    Nc: int,
    schemes,
    filling_label: str,
    filling_target: float,
    U: float,
    chi: int,
    t: float,
    output_rows,
):
    q_values = 2.0 * np.pi * np.arange(L // 2 + 1, dtype=float) / float(L)
    dmrg = get_dmrg_static_structure_factors(
        L,
        chi,
        U=U,
        t=t,
        V=0.0,
        V_sep=(1, 1),
        filling_target=filling_target,
        dmrg_fixed_filling=True,
        q_values=q_values,
    )

    top_charge = top_modes(dmrg["q_values"], dmrg["Nq"])
    top_spin = top_modes(dmrg["q_values"], dmrg["Sq"])

    print(f"{filling_label} filling, U={U:.2f}")
    print("  Top charge modes:")
    for step, q, val in top_charge:
        closures = [
            describe_probe_closure(L, Nc, scheme["int_sep"], probe_step=step)
            for scheme in schemes
        ]
        desc = ", ".join(
            f"{scheme['label']}:C={closure['closure_size']}"
            for scheme, closure in zip(schemes, closures)
        )
        print(f"    {q_label_from_step(step, L)} (step={step}, N={val:.4f}) -> {desc}")
        output_rows.append(
            {
                "filling": filling_label,
                "U": U,
                "channel": "charge",
                "step": step,
                "q_label": q_label_from_step(step, L),
                "weight": val,
                "closures": {
                    scheme["label"]: describe_probe_closure(L, Nc, scheme["int_sep"], probe_step=step)
                    for scheme in schemes
                },
            }
        )

    print("  Top spin modes:")
    for step, q, val in top_spin:
        closures = [
            describe_probe_closure(L, Nc, scheme["int_sep"], probe_step=step)
            for scheme in schemes
        ]
        desc = ", ".join(
            f"{scheme['label']}:C={closure['closure_size']}"
            for scheme, closure in zip(schemes, closures)
        )
        print(f"    {q_label_from_step(step, L)} (step={step}, S={val:.4f}) -> {desc}")
        output_rows.append(
            {
                "filling": filling_label,
                "U": U,
                "channel": "spin",
                "step": step,
                "q_label": q_label_from_step(step, L),
                "weight": val,
                "closures": {
                    scheme["label"]: describe_probe_closure(L, Nc, scheme["int_sep"], probe_step=step)
                    for scheme in schemes
                },
            }
        )


def main():
    L = 24
    Nc = 2
    chi = 32
    t = 1.0
    U_values = [1.0, 2.0, 4.0]
    fillings = [
        {"label": "half", "target": 1.0},
        {"label": "quarter", "target": 0.5},
    ]
    schemes = [
        {"label": "pi", "int_sep": (1, 2)},
        {"label": "pi_over_2", "int_sep": (1, 4)},
    ]

    output_dir = Path("large_files/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    scheme_spreads = {}
    print("Bare dispersion spread within each solved Nc=2 block:")
    for scheme in schemes:
        spread = dispersion_spread_summary(L, Nc, scheme["int_sep"], t=t)
        scheme_spreads[scheme["label"]] = spread
        print(
            f"  {scheme['label']}: avg={spread['avg_spread']:.4f}, "
            f"rms={spread['rms_spread']:.4f}, max={spread['max_spread']:.4f}"
        )
    print()

    rows = []
    for filling in fillings:
        for U in U_values:
            summarize_case(
                L=L,
                Nc=Nc,
                schemes=schemes,
                filling_label=filling["label"],
                filling_target=filling["target"],
                U=U,
                chi=chi,
                t=t,
                output_rows=rows,
            )

    payload = {
        "parameters": {
            "L": L,
            "Nc": Nc,
            "chi": chi,
            "t": t,
            "U_values": U_values,
            "fillings": fillings,
            "schemes": schemes,
            "scheme_dispersion_spreads": scheme_spreads,
        },
        "rows": rows,
    }

    payload_path = output_dir / "v0_mode_accessibility_scan.pkl"
    with payload_path.open("wb") as fh:
        pickle.dump(payload, fh)
    print(f"Saved payload to {payload_path}")


if __name__ == "__main__":
    main()
