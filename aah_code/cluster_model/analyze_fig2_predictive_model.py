import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from aah_code.cluster_model.clustering import generate_clusters, projected_probe_fraction
from aah_code.real_space_dmrg import get_dmrg_static_structure_factors


DEFAULT_FIG2_PATH = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/"
    "aah_code/cluster_model/large_files/plots/filling_int_cluster_comparison_L48_chi32_20260206_113516.pkl"
)


def parse_sep_key(key: str) -> Tuple[int, int]:
    p_str, q_str = key.split("_")
    return (int(p_str), int(q_str))


def wrap_dist(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def fermi_window_kinetic_balance(
    L: int,
    Nc: int,
    int_sep_ratio: Tuple[int, int],
    *,
    filling_target: float,
    sigma: float,
    t: float,
) -> float:
    clusters = generate_clusters(L, Nc, int_sep_ratio, (1, 1)).reshape(-1, Nc)
    k_vals = -np.pi + (2.0 * np.pi / float(L)) * np.arange(L, dtype=float)
    eps_vals = -2.0 * t * np.cos(k_vals)
    k_f = 0.5 * np.pi * filling_target

    weights = np.array(
        [
            math.exp(
                -(
                    min(
                        wrap_dist(float(k), +k_f),
                        wrap_dist(float(k), -k_f),
                    )
                    / sigma
                )
                ** 2
            )
            for k in k_vals
        ],
        dtype=float,
    )

    total_weight = 0.0
    total_var = 0.0
    for cluster in clusters:
        cluster = np.asarray(cluster, dtype=int)
        w = weights[cluster]
        w_sum = float(np.sum(w))
        if w_sum < 1e-14:
            continue
        e = eps_vals[cluster]
        e_mean = float(np.sum(w * e) / w_sum)
        e_var = float(np.sum(w * (e - e_mean) ** 2) / w_sum)
        total_weight += w_sum
        total_var += w_sum * e_var

    if total_weight < 1e-14:
        return float("nan")
    return total_var / total_weight


def min_max_normalize(values: Dict[Tuple, float]) -> Dict[Tuple, float]:
    finite_vals = [float(v) for v in values.values() if np.isfinite(v)]
    if not finite_vals:
        return {k: float("nan") for k in values}
    v_min = min(finite_vals)
    v_max = max(finite_vals)
    if abs(v_max - v_min) < 1e-14:
        return {k: 0.0 for k in values}
    return {k: (float(v) - v_min) / (v_max - v_min) for k, v in values.items()}


def structure_factor_cache(
    *,
    L: int,
    chi: int,
    t: float,
    U_values: Iterable[float],
    filling_modes: Dict[str, float],
) -> Dict[Tuple[str, float], Dict[str, np.ndarray]]:
    q_values = 2.0 * np.pi * np.arange(L // 2 + 1, dtype=float) / float(L)
    cache: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}
    for fill_mode, filling_target in filling_modes.items():
        for U in U_values:
            print(f"Computing DMRG structure factors: fill={fill_mode}, U={U}")
            res = get_dmrg_static_structure_factors(
                L,
                chi,
                U=float(U),
                t=t,
                V=0.0,
                V_sep=(1, 1),
                filling_target=float(filling_target),
                dmrg_fixed_filling=True,
                q_values=q_values,
            )
            cache[(fill_mode, float(U))] = {
                "q_values": np.asarray(res["q_values"], dtype=float),
                "Nq": np.asarray(res["Nq"], dtype=float),
                "Sq": np.asarray(res["Sq"], dtype=float),
            }
    return cache


def serialize_dmrg_cache(cache: Dict[Tuple[str, float], Dict[str, np.ndarray]]) -> Dict[str, Dict[str, List[float]]]:
    out = {}
    for (fill_mode, U), payload in cache.items():
        out[f"{fill_mode}:{U:g}"] = {
            "q_values": np.asarray(payload["q_values"], dtype=float).tolist(),
            "Nq": np.asarray(payload["Nq"], dtype=float).tolist(),
            "Sq": np.asarray(payload["Sq"], dtype=float).tolist(),
        }
    return out


def deserialize_dmrg_cache(payload: Dict[str, Dict[str, List[float]]]) -> Dict[Tuple[str, float], Dict[str, np.ndarray]]:
    out = {}
    for key, arrays in payload.items():
        fill_mode, u_str = key.split(":", 1)
        out[(fill_mode, float(u_str))] = {
            "q_values": np.asarray(arrays["q_values"], dtype=float),
            "Nq": np.asarray(arrays["Nq"], dtype=float),
            "Sq": np.asarray(arrays["Sq"], dtype=float),
        }
    return out


def weighted_probe_capture(
    *,
    L: int,
    Nc: int,
    int_sep_ratio: Tuple[int, int],
    q_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    mask = np.abs(q_values) > 1e-12
    if not np.any(mask):
        return float("nan")
    q_sel = np.asarray(q_values[mask], dtype=float)
    w_sel = np.asarray(weights[mask], dtype=float)
    w_sum = float(np.sum(w_sel))
    if w_sum < 1e-14:
        return float("nan")

    retained = []
    for q in q_sel:
        step = int(round(float(q) * L / (2.0 * np.pi))) % L
        retained.append(
            projected_probe_fraction(
                L,
                Nc,
                int_sep_ratio,
                probe_step=step,
                v_ratios=(1, 1),
            )["fraction"]
        )
    retained = np.asarray(retained, dtype=float)
    return float(np.sum(w_sel * retained) / w_sum)


def winner_accuracy(
    model_scores: Dict[Tuple[int, str, float, Tuple[int, int]], float],
    actual_errors: Dict[Tuple[int, str, float, Tuple[int, int]], float],
    schemes_by_nc: Dict[int, List[Tuple[int, int]]],
    *,
    U_values: Iterable[float],
    fill_modes: Iterable[str],
) -> Tuple[float, List[Dict[str, object]]]:
    hits = 0
    total = 0
    rows: List[Dict[str, object]] = []
    for Nc, schemes in schemes_by_nc.items():
        for fill_mode in fill_modes:
            for U in U_values:
                pred = sorted(
                    (model_scores[(Nc, fill_mode, float(U), scheme)], scheme)
                    for scheme in schemes
                )
                act = sorted(
                    (actual_errors[(Nc, fill_mode, float(U), scheme)], scheme)
                    for scheme in schemes
                )
                pred_best = pred[0][1]
                act_best = act[0][1]
                hits += int(pred_best == act_best)
                total += 1
                rows.append(
                    {
                        "Nc": Nc,
                        "fill_mode": fill_mode,
                        "U": float(U),
                        "pred_best": pred_best,
                        "actual_best": act_best,
                        "pred_score": float(pred[0][0]),
                        "actual_error": float(act[0][0]),
                        "match": bool(pred_best == act_best),
                    }
                )
    return (float(hits) / float(total) if total else float("nan")), rows


def probe_only_summary(
    *,
    fig2_data: Dict,
    dmrg_cache: Dict[Tuple[str, float], Dict[str, np.ndarray]],
    target_ncs: Iterable[int],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    fill_modes = ["half", "quarter"]
    finite_dmrg = np.asarray(fig2_data["finite_dmrg_energies"], dtype=float)
    u_values = [float(u) for u in fig2_data["U_values"]]

    for Nc in target_ncs:
        scheme_keys = sorted(fig2_data["cluster_energies"][str(Nc)].keys(), key=parse_sep_key)
        schemes = [parse_sep_key(key) for key in scheme_keys]
        for fill_idx, fill_mode in enumerate(fill_modes):
            for U in u_values:
                q_cache = dmrg_cache[(fill_mode, U)]
                spin_capture = {}
                charge_capture = {}
                actual_errors = {}
                u_idx = u_values.index(U)
                for scheme_key, scheme in zip(scheme_keys, schemes):
                    spin_capture[scheme] = weighted_probe_capture(
                        L=int(fig2_data["parameters"]["L"]),
                        Nc=Nc,
                        int_sep_ratio=scheme,
                        q_values=q_cache["q_values"],
                        weights=q_cache["Sq"],
                    )
                    charge_capture[scheme] = weighted_probe_capture(
                        L=int(fig2_data["parameters"]["L"]),
                        Nc=Nc,
                        int_sep_ratio=scheme,
                        q_values=q_cache["q_values"],
                        weights=q_cache["Nq"],
                    )
                    cluster_val = float(fig2_data["cluster_energies"][str(Nc)][scheme_key][fill_mode][u_idx])
                    actual_errors[scheme] = abs(cluster_val - finite_dmrg[u_idx, fill_idx])

                summaries.append(
                    {
                        "Nc": Nc,
                        "fill_mode": fill_mode,
                        "U": U,
                        "spin_ranking": sorted(
                            ((float(val), scheme) for scheme, val in spin_capture.items()),
                            reverse=True,
                        ),
                        "charge_ranking": sorted(
                            ((float(val), scheme) for scheme, val in charge_capture.items()),
                            reverse=True,
                        ),
                        "actual_ranking": sorted(
                            ((float(val), scheme) for scheme, val in actual_errors.items()),
                        ),
                    }
                )
    return summaries


def fit_predictive_model(fig2_data: Dict, dmrg_cache: Dict[Tuple[str, float], Dict[str, np.ndarray]]):
    fill_modes = ["half", "quarter"]
    filling_targets = {"half": 1.0, "quarter": 0.5}
    u_values = [float(u) for u in fig2_data["U_values"]]
    finite_dmrg = np.asarray(fig2_data["finite_dmrg_energies"], dtype=float)
    t = float(fig2_data["parameters"]["t"])
    L = int(fig2_data["parameters"]["L"])

    schemes_by_nc: Dict[int, List[Tuple[int, int]]] = {}
    actual_errors: Dict[Tuple[int, str, float, Tuple[int, int]], float] = {}
    for Nc in fig2_data["cluster_sizes"]:
        Nc = int(Nc)
        scheme_keys = sorted(fig2_data["cluster_energies"][str(Nc)].keys(), key=parse_sep_key)
        schemes_by_nc[Nc] = [parse_sep_key(key) for key in scheme_keys]
        for fill_idx, fill_mode in enumerate(fill_modes):
            for u_idx, U in enumerate(u_values):
                for scheme_key in scheme_keys:
                    scheme = parse_sep_key(scheme_key)
                    cluster_val = float(fig2_data["cluster_energies"][str(Nc)][scheme_key][fill_mode][u_idx])
                    actual_errors[(Nc, fill_mode, U, scheme)] = abs(cluster_val - finite_dmrg[u_idx, fill_idx])

    best_by_fill = {}
    sigma_grid = [np.pi / 48.0, np.pi / 24.0, np.pi / 16.0, np.pi / 12.0, np.pi / 8.0]
    weight_grid = np.arange(0.0, 1.0001, 0.05)

    for fill_mode in fill_modes:
        filling_target = filling_targets[fill_mode]
        best_result = None

        kinetic_cache_by_sigma = {}
        for sigma in sigma_grid:
            kinetic_cache = {}
            for Nc, schemes in schemes_by_nc.items():
                raw = {
                    scheme: fermi_window_kinetic_balance(
                        L,
                        Nc,
                        scheme,
                        filling_target=filling_target,
                        sigma=sigma,
                        t=t,
                    )
                    for scheme in schemes
                }
                norm = min_max_normalize({(Nc, scheme): val for scheme, val in raw.items()})
                for scheme in schemes:
                    kinetic_cache[(Nc, scheme)] = norm[(Nc, scheme)]
            kinetic_cache_by_sigma[sigma] = kinetic_cache

        probe_spin = {}
        probe_charge = {}
        for Nc, schemes in schemes_by_nc.items():
            for U in u_values:
                q_cache = dmrg_cache[(fill_mode, U)]
                spin_raw = {}
                charge_raw = {}
                for scheme in schemes:
                    spin_raw[scheme] = 1.0 - weighted_probe_capture(
                        L=L,
                        Nc=Nc,
                        int_sep_ratio=scheme,
                        q_values=q_cache["q_values"],
                        weights=q_cache["Sq"],
                    )
                    charge_raw[scheme] = 1.0 - weighted_probe_capture(
                        L=L,
                        Nc=Nc,
                        int_sep_ratio=scheme,
                        q_values=q_cache["q_values"],
                        weights=q_cache["Nq"],
                    )
                spin_norm = min_max_normalize({(Nc, U, scheme): val for scheme, val in spin_raw.items()})
                charge_norm = min_max_normalize({(Nc, U, scheme): val for scheme, val in charge_raw.items()})
                for scheme in schemes:
                    probe_spin[(Nc, U, scheme)] = spin_norm[(Nc, U, scheme)]
                    probe_charge[(Nc, U, scheme)] = charge_norm[(Nc, U, scheme)]

        for sigma in sigma_grid:
            kinetic_cache = kinetic_cache_by_sigma[sigma]
            for w_k in weight_grid:
                for w_s in weight_grid:
                    w_c = 1.0 - w_k - w_s
                    if w_c < -1e-12:
                        continue
                    w_c = max(0.0, w_c)
                    model_scores = {}
                    for Nc, schemes in schemes_by_nc.items():
                        for U in u_values:
                            for scheme in schemes:
                                score = (
                                    w_k * kinetic_cache[(Nc, scheme)]
                                    + w_s * probe_spin[(Nc, U, scheme)]
                                    + w_c * probe_charge[(Nc, U, scheme)]
                                )
                                model_scores[(Nc, fill_mode, U, scheme)] = float(score)
                    acc, rows = winner_accuracy(
                        model_scores,
                        actual_errors,
                        schemes_by_nc,
                        U_values=u_values,
                        fill_modes=[fill_mode],
                    )
                    candidate = {
                        "sigma": float(sigma),
                        "weights": {"kinetic": float(w_k), "spin": float(w_s), "charge": float(w_c)},
                        "accuracy": float(acc),
                        "rows": rows,
                    }
                    if best_result is None or candidate["accuracy"] > best_result["accuracy"]:
                        best_result = candidate
        best_by_fill[fill_mode] = best_result

    return best_by_fill, schemes_by_nc, actual_errors


def main():
    fig2_path = DEFAULT_FIG2_PATH
    with fig2_path.open("rb") as fh:
        fig2_data = pickle.load(fh)

    L = int(fig2_data["parameters"]["L"])
    chi = int(fig2_data["parameters"]["chi"])
    t = float(fig2_data["parameters"]["t"])
    u_values = [float(u) for u in fig2_data["U_values"]]
    filling_targets = {"half": 1.0, "quarter": 0.5}

    print(f"Loaded Fig. 2 cache from {fig2_path}")
    output_dir = Path("large_files/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    dmrg_cache_path = output_dir / "fig2_dmrg_structure_factor_cache.pkl"
    if dmrg_cache_path.exists():
        with dmrg_cache_path.open("rb") as fh:
            dmrg_cache = deserialize_dmrg_cache(pickle.load(fh))
        print(f"Loaded DMRG structure-factor cache from {dmrg_cache_path}")
    else:
        dmrg_cache = structure_factor_cache(
            L=L,
            chi=chi,
            t=t,
            U_values=u_values,
            filling_modes=filling_targets,
        )
        with dmrg_cache_path.open("wb") as fh:
            pickle.dump(serialize_dmrg_cache(dmrg_cache), fh)
        print(f"Saved DMRG structure-factor cache to {dmrg_cache_path}")

    probe_summaries = probe_only_summary(
        fig2_data=fig2_data,
        dmrg_cache=dmrg_cache,
        target_ncs=[2, 4],
    )

    print("\nProbe-only summary for Nc=2,4")
    for row in probe_summaries:
        spin_best = row["spin_ranking"][0][1]
        charge_best = row["charge_ranking"][0][1]
        actual_best = row["actual_ranking"][0][1]
        print(
            f"Nc={row['Nc']} fill={row['fill_mode']} U={row['U']}: "
            f"spin_best={spin_best}, charge_best={charge_best}, actual_best={actual_best}"
        )

    best_by_fill, schemes_by_nc, actual_errors = fit_predictive_model(fig2_data, dmrg_cache)

    print("\nBest fitted model by filling")
    for fill_mode, result in best_by_fill.items():
        print(
            f"{fill_mode}: accuracy={result['accuracy']:.3f}, "
            f"sigma/pi={result['sigma'] / np.pi:.4f}, weights={result['weights']}"
        )

    output_path = output_dir / "fig2_predictive_model_analysis.pkl"
    with output_path.open("wb") as fh:
        pickle.dump(
            {
                "fig2_path": str(fig2_path),
                "dmrg_cache": dmrg_cache,
                "probe_summaries": probe_summaries,
                "best_by_fill": best_by_fill,
            },
            fh,
        )
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
