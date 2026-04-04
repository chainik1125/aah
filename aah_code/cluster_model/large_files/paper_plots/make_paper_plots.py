"""
Publication-quality figures for the AAH/AAHK paper.

Reads pre-computed pickle data and generates matplotlib figures
suitable for a two-column RevTeX paper.

Usage (from venv):
    python large_files/paper_plots/make_paper_plots.py
"""

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedLocator
from pathlib import Path
from fractions import Fraction

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # large_files/
MERGED = ROOT / "cluster_runs" / "merged_results"
PARTIALS = ROOT / "partials"
PLOTS_DIR = ROOT / "plots"
OUT = Path(__file__).resolve().parent                  # paper_plots/
IMAGES = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/"
    "K_blocking/TeX/paper/AAH_AAHK_Paper/images"
)

# ── Data files with both fillings ────────────────────────────────────
DATA_BOTH_FILLINGS = {
    "1-2": PARTIALS / "filling_L48_20260126_001228" / "merged_fixed.pkl",
    "1-3": PARTIALS / "filling_L48_vsep_2pi3_20260126_112422" / "merged_20260126_112422.pkl",
    "1-4": PLOTS_DIR / "filling_cluster_size_comparison_L40_chi32_20260106_211444.pkl",
}
HUB_DATA = PLOTS_DIR / "filling_int_cluster_comparison_L48_chi32_20260206_113516.pkl"
HUB_SMALLU_DATA = PLOTS_DIR / "filling_int_cluster_comparison_L48_chi32_20260313_072204.pkl"

# Half-filling only, larger L, more V points
MERGED_HALF = {
    "1-2": MERGED / "sep_1-2_merge.pkl",
    "1-3": MERGED / "sep_1-3_merge.pkl",
    "1-4": MERGED / "sep_1-4_merge.pkl",
}

# Fixed supercluster comparison (L=120, separate files per filling)
FIXED_SC_HALF = PLOTS_DIR / "fixed_supercluster_comparison_L120_chi32_20260109_164745.pkl"
FIXED_SC_QUARTER = PLOTS_DIR / "fixed_supercluster_comparison_L120_chi32_20260113_174059.pkl"

# ── Style ────────────────────────────────────────────────────────────
FULL_WIDTH = 8.5    # wider than RevTeX for clarity
YLABEL = r"$(E^0_{\mathrm{cl}} - E^0_{\mathrm{DMRG}})\,/\,E^0_{\mathrm{DMRG}}$"
YLABEL_BETHE = r"$(E^0_{\mathrm{cl}} - E^0_{\mathrm{exact}})\,/\,E^0_{\mathrm{exact}}$"
YLABEL_FONTSIZE = 7.5


def setup_style():
    """Clean minimal style: white bg, L-shaped spines, outward ticks."""
    mpl.rcdefaults()
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.5, "legend.handletextpad": 0.4,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


# ── Colormap helpers ─────────────────────────────────────────────────
CMAP = mpl.colormaps["OrRd"]


def gradient_colors(n, cmap=CMAP, lo=0.30, hi=0.85):
    """Return n colours evenly spaced in [lo, hi] of a colormap."""
    if n == 1:
        return [cmap(hi)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


# ── Nc color mapping (warm gradient, used for V-convergence plots) ──
NC_COLORS = {}  # populated per-figure based on available Nc values


def nc_gradient(nc_list):
    """Return {nc: color} using warm gradient, largest Nc = darkest."""
    nc_sorted = sorted(nc_list)
    colors = gradient_colors(len(nc_sorted))
    return dict(zip(nc_sorted, colors))


# ── Formatting helpers ───────────────────────────────────────────────

def fmt_sep(ratio):
    """(num, denom) ratio -> LaTeX pi-fraction for legends."""
    num, den = ratio
    val = Fraction(2 * num, den)
    if val == 1:        return r"\pi"
    if val.numerator == 1: return rf"\pi/{val.denominator}"
    if val.denominator == 1: return rf"{val.numerator}\pi"
    return rf"{val.numerator}\pi/{val.denominator}"


def _save(fig, name, also_to_images=None):
    """Save figure as PDF + PNG to paper_plots/, optionally copy to images/."""
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}")
    if also_to_images:
        fig.savefig(IMAGES / also_to_images)


def _panel_label(ax, idx):
    """Place (a), (b), ... outside the top-left of the axes."""
    ax.text(-0.02, 1.02, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, fontsize=7,
            va="bottom", ha="right", fontweight="bold")


def _filling_annotations(axes, n_rows=2):
    """Add half/quarter-filling labels on the right side."""
    labels = [r"Half-filling ($\nu\!=\!1$)",
              r"Quarter-filling ($\nu\!=\!1/2$)"]
    for row in range(min(n_rows, len(labels))):
        axes[row, -1].annotate(labels[row], xy=(1.04, 0.5),
                               xycoords="axes fraction", rotation=270,
                               va="center", ha="left", fontsize=7.5)


# ═════════════════════════════════════════════════════════════════════
# Figure 2: V=0 Hubbard comparison (hub_comparison)
# ═════════════════════════════════════════════════════════════════════

def plot_hub_comparison(use_bethe=False):
    """
    Args:
        use_bethe: Reference energy source.
            False  — OBC finite DMRG from the pickle.
            'thermodynamic' or True — thermodynamic Bethe ansatz (L→∞).
            'finite' — finite periodic Bethe ansatz at the data's L.
    """
    with open(HUB_DATA, "rb") as f:
        data = pickle.load(f)

    U = np.array(data["U_values"])
    sizes = data["cluster_sizes"]
    seps = data["int_sep_ratios_by_Nc"]
    ce = data["cluster_energies"]
    fillings = ["half", "quarter"]
    filling_n = {"half": 1.0, "quarter": 0.5}
    L = data["parameters"]["L"]

    if use_bethe:
        from aah_code.bethe_ansatz import lieb_wu_energy_general_filling
        if use_bethe == 'finite':
            bethe_kw = dict(bethe_mode="finite", finite_L=L)
            save_suffix = "_bethe_finite"
        else:
            bethe_kw = {}
            save_suffix = "_bethe"
        ref = {f: np.array([lieb_wu_energy_general_filling(u, filling_n[f], **bethe_kw)
                            for u in U])
               for f in fillings}
        ylabel = YLABEL_BETHE
    else:
        fde = data["finite_dmrg_energies"]
        ref = {f: np.array([fde[ui][fi] for ui in range(len(U))])
               for fi, f in enumerate(fillings)}
        ylabel = YLABEL
        save_suffix = ""

    n_nc = len(sizes)
    fig, axes = plt.subplots(2, n_nc, figsize=(FULL_WIDTH, 3.8),
                             sharex="col", sharey="row", squeeze=False)

    for col, nc_str in enumerate(map(str, sizes)):
        nc = int(nc_str)
        sep_list = seps[nc_str]
        n_seps = len(sep_list)
        sorted_idx = sorted(range(n_seps),
                            key=lambda i: sep_list[i][0] / sep_list[i][1])
        rank = {i: r for r, i in enumerate(sorted_idx)}
        colors = gradient_colors(n_seps - 1)  # for non-maximal

        for row, filling in enumerate(fillings):
            ax = axes[row, col]
            E_ref = ref[filling]

            for si, sep in enumerate(sep_list):
                sep_key = f"{sep[0]}_{sep[1]}"
                E_cl = np.array(ce[nc_str][sep_key][filling])
                rel_err = np.abs((E_cl - E_ref) / E_ref)

                maximal = (rank[si] == n_seps - 1)
                if maximal:
                    color = "black"
                else:
                    color = colors[rank[si]]

                label = (r"$\Delta\!=\!" + fmt_sep(tuple(sep)) + r"$"
                         + (r" (max)" if maximal else ""))

                ax.plot(U, rel_err * 100,
                        marker="o", linestyle="-",
                        color=color, lw=1.6 if maximal else 1.3,
                        zorder=10 if maximal else 5 + rank[si],
                        markersize=3.5,
                        label=label if row == 0 else None)

            ax.set_ylim(bottom=-0.5)
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20, 25, 30]))
            if row == 0: ax.set_title(rf"$N_c={nc}$")
            if row == 1: ax.set_xlabel(r"$U/t$")
            if col == 0: ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)

            _panel_label(ax, row * n_nc + col)

        # Legend below bottom subplot
        handles, labels = axes[0, col].get_legend_handles_labels()
        axes[1, col].legend(handles, labels,
                            loc="upper center", bbox_to_anchor=(0.5, -0.32),
                            fontsize=6.5, ncol=1, frameon=False,
                            handlelength=1.5, handletextpad=0.4,
                            columnspacing=0.8)

    _filling_annotations(axes)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.22)
    _save(fig, f"fig_hub_comparison{save_suffix}",
          "hub_comparison_L48_Nc8.png" if not use_bethe else None)
    return fig


# ═════════════════════════════════════════════════════════════════════
# Figures 4+: V-convergence (relative error vs V)
# ═════════════════════════════════════════════════════════════════════

def plot_v_convergence(beta_str, U_subset, suffix="", image_name=None, use_pbc=False):
    """
    Relative error vs V for maximal-separation clustering at a given beta.
    Top row = half-filling, bottom = quarter-filling.
    Columns = different U values.  Lines = different Nc (warm gradient).

    Args:
        use_pbc: If True, use PBC DMRG energies ('pbc_finite_dmrg_energies')
                 instead of OBC ('finite_dmrg_energies') as the reference.
    """
    with open(DATA_BOTH_FILLINGS[beta_str], "rb") as f:
        data = pickle.load(f)

    params = data["parameters"]
    v_sep = params["v_sep_ratio"]
    L = params["L"]

    U_values = data["U_values"]
    ce = data["cluster_energies"]
    dmrg_key = "pbc_finite_dmrg_energies" if use_pbc else "finite_dmrg_energies"
    fde = data[dmrg_key]

    cluster_sizes = sorted([int(k) for k in ce.keys()])
    nc_colors = nc_gradient(cluster_sizes)

    first_nc = list(ce.keys())[0]
    V_keys = list(ce[first_nc].keys())
    V_values = np.array([float(v) for v in V_keys])

    u_mask = [i for i, u in enumerate(U_values) if u in U_subset]
    n_cols = len(u_mask)
    filling_map = {0: "half", 1: "quarter"}

    fig, axes = plt.subplots(2, n_cols, figsize=(FULL_WIDTH, 3.8),
                             sharex=True, sharey="row", squeeze=False)

    for col_idx, u_idx in enumerate(u_mask):
        U_val = U_values[u_idx]

        for row in range(2):
            ax = axes[row, col_idx]
            filling = filling_map[row]
            fill_idx = row

            E_dmrg = np.array([fde[u_idx][vi][fill_idx]
                               for vi in range(len(V_values))])

            for nc_str in ce.keys():
                nc = int(nc_str)
                E_nc = np.array([ce[nc_str][vk][filling][u_idx]
                                 for vk in V_keys])

                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = np.abs((E_nc - E_dmrg) / E_dmrg) * 100

                color = nc_colors.get(nc, "gray")
                ax.plot(V_values, rel_err, marker="o", markersize=3.5,
                        color=color, lw=1.3,
                        label=(rf"$N_c={nc}$"
                               if row == 0 and col_idx == 0 else None))

            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

            U_label = rf"$U={U_val:.0f}$" if U_val == int(U_val) else rf"$U={U_val}$"
            if row == 0: ax.set_title(U_label)
            if row == 1: ax.set_xlabel(r"$\lambda/t$")
            if col_idx == 0: ax.set_ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)

            _panel_label(ax, row * n_cols + col_idx)

    # Set shared y-limits per row from actual data range (with small padding)
    for row in range(2):
        y_max = max(ax.get_ylim()[1] for ax in axes[row, :])
        axes[row, 0].set_ylim(bottom=-0.3, top=y_max * 1.05)

    # Single legend below centre of figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02),
               fontsize=6.5, ncol=len(cluster_sizes), frameon=False,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.2)

    _filling_annotations(axes)

    beta_frac = Fraction(*v_sep)
    ref_label = "PBC" if use_pbc else "DMRG"
    pbc_params = data.get("pbc_dmrg_parameters", {})
    chi_label = pbc_params.get("chi", params.get("chi", "")) if use_pbc else params.get("chi", "")
    fig.suptitle(
        rf"$\beta = {beta_frac}$ "
        rf"($Q={fmt_sep(v_sep)}$), $L={L}$"
        + (rf", ref={ref_label} $\chi={chi_label}$" if use_pbc else ""),
        fontsize=9, y=1.02)

    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.12)

    pbc_suffix = "_pbc" if use_pbc else ""
    name = f"fig_v_convergence_{beta_str}{suffix}{pbc_suffix}"
    _save(fig, name, image_name)
    return fig


# ═════════════════════════════════════════════════════════════════════
# Figure 5: Fixed supercluster comparison
# ═════════════════════════════════════════════════════════════════════

def plot_fixed_supercluster(use_pbc=False):
    """
    Maximal vs non-maximal clustering at fixed supercluster size,
    U=5, beta=1/2.  Two rows: half-filling (top), quarter-filling (bottom).
    Uses L=120 data from separate pickle files per filling.

    Args:
        use_pbc: If True, use PBC DMRG energies ('pbc_finite_dmrg_energies')
                 instead of OBC ('finite_dmrg_energies') as the reference.
    """
    filling_files = [
        ("half", FIXED_SC_HALF),
        ("quarter", FIXED_SC_QUARTER),
    ]
    datasets = {}
    for fill_label, path in filling_files:
        with open(path, "rb") as f:
            datasets[fill_label] = pickle.load(f)

    # Both files share the same structure; use half to get common params
    ref = datasets["half"]
    U_values = np.array(ref["U_values"])
    V_values = np.array(ref["V_values"])
    sc_sizes = sorted(ref["supercluster_sizes"])
    sc_int_seps = ref["supercluster_int_seps"]
    L = ref["parameters"]["L"]

    U_target = 5.0
    u_idx = int(np.where(U_values == U_target)[0][0])

    n_cols = len(sc_sizes)
    fillings = ["half", "quarter"]
    fig, axes = plt.subplots(2, n_cols, figsize=(FULL_WIDTH, 4.2),
                             sharex=True, sharey="row", squeeze=False)

    for col, sc in enumerate(sc_sizes):
        sc_str = str(sc)
        configs = sc_int_seps[sc_str]  # list of [Nc, [p, q]]
        n_cfg = len(configs)

        # Identify maximal: largest Nc in this SC group
        max_nc = max(cfg[0] for cfg in configs)

        # Sort: maximal last so it draws on top
        sorted_configs = sorted(configs, key=lambda c: (c[0] == max_nc, c[0]))
        non_max_count = sum(1 for c in sorted_configs if c[0] != max_nc)
        nm_colors = gradient_colors(max(non_max_count, 1))

        for row, fill_label in enumerate(fillings):
            ax = axes[row, col]
            data = datasets[fill_label]
            ce = data["cluster_energies"][sc_str]
            dmrg_key = "pbc_finite_dmrg_energies" if use_pbc else "finite_dmrg_energies"
            fde = np.array(data[dmrg_key])
            E_dmrg = fde[:, u_idx]  # shape (n_V,)

            nm_ci = 0
            for ci, (nc, int_sep_list) in enumerate(sorted_configs):
                int_sep = tuple(int_sep_list)
                config_key = f"{nc}_{int_sep[0]}_{int_sep[1]}"
                if config_key not in ce:
                    continue

                E_nc = np.array(ce[config_key])[:, u_idx]  # shape (n_V,)

                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = np.abs((E_nc - E_dmrg) / E_dmrg) * 100

                is_max = (nc == max_nc)
                if is_max:
                    color = "black"
                    lw = 1.6
                else:
                    color = nm_colors[nm_ci]
                    nm_ci += 1
                    lw = 1.3

                sep_label = fmt_sep(int_sep)
                label = (rf"$N_c\!=\!{nc}$, $\Delta\!=\!{sep_label}$"
                         + (r" (max)" if is_max else ""))
                ax.plot(V_values, rel_err, marker="o", markersize=3.5,
                        color=color, lw=lw,
                        label=label if row == 0 else None,
                        zorder=10 if is_max else 5 + ci)

            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            if row == 0: ax.set_title(rf"$N_{{\mathrm{{SC}}}}={sc}$")
            if row == 1: ax.set_xlabel(r"$\lambda/t$")
            if col == 0: ax.set_ylabel(YLABEL, fontsize=YLABEL_FONTSIZE)

            _panel_label(ax, row * n_cols + col)

        # Legend below bottom subplot of each column
        handles, labels = axes[0, col].get_legend_handles_labels()
        axes[1, col].legend(handles, labels,
                            loc="upper center", bbox_to_anchor=(0.5, -0.32),
                            fontsize=6.5, ncol=1, frameon=False,
                            handlelength=1.5, handletextpad=0.4,
                            columnspacing=0.8)

    # Set shared y-limits per row from actual data range
    for row in range(2):
        y_max = max(ax.get_ylim()[1] for ax in axes[row, :])
        axes[row, 0].set_ylim(bottom=-0.5, top=y_max * 1.05)

    _filling_annotations(axes)
    pbc_params = datasets["half"].get("pbc_dmrg_parameters", {})
    chi_label = pbc_params.get("chi", "") if use_pbc else ""
    ref_note = rf", ref=PBC $\chi={chi_label}$" if use_pbc else ""
    fig.suptitle(
        rf"Fixed supercluster size, $U={U_target:.0f}$, $\beta=1/2$, $L={L}$"
        + ref_note,
        fontsize=9, y=1.02)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.22)

    pbc_suffix = "_pbc" if use_pbc else ""
    _save(fig, f"fig_fixed_supercluster{pbc_suffix}",
          "v_convergence_fixedsc_v_pi_fixedU_5.png")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Appendix Fig 2: Compressibility (ν vs μ₀) for different int seps
# ═════════════════════════════════════════════════════════════════════

COMPRESS_DATA = PLOTS_DIR / "compressibility_combined_L48_chi32_20260311_183828_dmrg_20260311_202113.pkl"
COMPRESS_SMALLU_DATA = PLOTS_DIR / "compressibility_smallU_combined_L48_chi32_20260312_173238.pkl"


def plot_hub_compressibility(pickle_path=None):
    """
    Fig 2 appendix: filling ν vs μ₀ for each (U, Nc) panel.
    Rows = U values, Columns = Nc values.
    Lines = different interaction separations (YlOrRd gradient, maximal = black).
    DMRG reference = grey dashed line.
    """
    path = Path(pickle_path) if pickle_path else COMPRESS_DATA
    with open(path, "rb") as f:
        data = pickle.load(f)

    U_values = data["U_values"]
    cluster_sizes = data["cluster_sizes"]
    int_sep_ratios = data["int_sep_ratios_by_Nc"]
    mu_arrays = {float(k): np.array(v) for k, v in data["mu_arrays"].items()}
    cf = data["cluster_fillings"]
    dmrg_fill = data.get("finite_dmrg_fillings", {})
    params = data["parameters"]
    L = params["L"]

    n_rows = len(U_values)
    n_cols = len(cluster_sizes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, 1.6 * n_rows),
        sharex="row", sharey="row", squeeze=False,
    )

    for col, nc in enumerate(cluster_sizes):
        nc_str = str(nc)
        sep_list = [tuple(s) for s in int_sep_ratios[nc_str]]
        n_seps = len(sep_list)

        # Sort by separation ratio (ascending) — maximal = last
        sorted_idx = sorted(range(n_seps),
                            key=lambda i: sep_list[i][0] / sep_list[i][1])
        rank = {i: r for r, i in enumerate(sorted_idx)}
        colors = gradient_colors(max(n_seps - 1, 1))

        for row, U in enumerate(U_values):
            ax = axes[row, col]
            u_str = str(U)
            mu_arr = mu_arrays[U]

            # Plot DMRG reference first (behind cluster lines)
            # get_gnd returns filling per site (np.mean over sites), no /L needed
            if u_str in dmrg_fill:
                dmrg_nu = np.array(dmrg_fill[u_str])
                ax.plot(mu_arr, dmrg_nu, color="0.55", ls="--", lw=1.0,
                        zorder=2,
                        label=r"DMRG" if row == 0 and col == 0 else None)

            # Plot cluster ED curves
            for si, sep in enumerate(sep_list):
                sep_key = f"{sep[0]}_{sep[1]}"
                if sep_key not in cf.get(nc_str, {}):
                    continue
                if u_str not in cf[nc_str][sep_key]:
                    continue

                nu = np.array(cf[nc_str][sep_key][u_str])

                maximal = (rank[si] == n_seps - 1)
                if maximal:
                    color = "black"
                    lw = 1.6
                else:
                    color = colors[rank[si]]
                    lw = 1.2

                label_str = (r"$\Delta\!=\!" + fmt_sep(sep) + r"$"
                             + (r" (max)" if maximal else ""))

                ax.plot(mu_arr, nu,
                        color=color, lw=lw,
                        zorder=10 if maximal else 5 + rank[si],
                        label=label_str if row == 0 else None)

            # Axis formatting
            if row == 0:
                ax.set_title(rf"$N_c={nc}$")
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mu_0/t$")
            if col == 0:
                U_label = rf"$U={U:.0f}$" if U == int(U) else rf"$U={U}$"
                ax.set_ylabel(r"$\nu$", fontsize=YLABEL_FONTSIZE)
                ax.text(-0.22, 0.5, U_label, transform=ax.transAxes,
                        fontsize=7, va="center", ha="center", rotation=90)

        # Legend below bottom subplot of each column
        handles, labels = axes[0, col].get_legend_handles_labels()
        if handles:
            axes[-1, col].legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, -0.45),
                fontsize=6, ncol=1, frameon=False,
                handlelength=1.5, handletextpad=0.4,
            )

    fig.suptitle(
        rf"Filling $\nu(\mu_0)$: cluster ED vs DMRG, $V=0$, $L={L}$",
        fontsize=9, y=1.01,
    )
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.25, wspace=0.20, bottom=0.14)

    _save(fig, "fig_hub_compressibility",
          "current/hub_compressibility_L48.pdf")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Appendix Fig 4: Compressibility (ν vs μ₀) for different Nc, finite V
# ═════════════════════════════════════════════════════════════════════

FIG4_COMPRESS_DATA = None  # populated after DMRG finishes


def plot_v_compressibility(pickle_path=None):
    """
    Fig 4 appendix: filling ν vs μ₀ for each (U, V) panel.
    Rows = U values, Columns = V values.
    Lines = different Nc (warm gradient, maximal separation).
    DMRG reference = grey dashed line.
    """
    if pickle_path is None:
        # Find the latest combined pickle
        candidates = sorted(PLOTS_DIR.glob("fig4_compressibility_combined_*.pkl"), reverse=True)
        if not candidates:
            candidates = sorted(PLOTS_DIR.glob("fig4_compressibility_ED_only_*.pkl"), reverse=True)
        if not candidates:
            print("  No Fig 4 compressibility pickle found, skipping.")
            return None
        pickle_path = candidates[0]

    path = Path(pickle_path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    U_values = data["U_values"]
    V_values = data["V_values"]
    cluster_sizes = data["cluster_sizes"]
    cf = data["cluster_fillings"]
    dmrg_fill = data.get("finite_dmrg_fillings", {})
    params = data["parameters"]
    L = params["L"]
    v_sep = tuple(params["v_sep_ratio"])

    # Reconstruct mu_arrays
    mu_arrays = {}
    for k, v in data["mu_arrays"].items():
        u_val, v_val = k.split("_")
        mu_arrays[(float(u_val), float(v_val))] = np.array(v)

    nc_colors = nc_gradient(cluster_sizes)

    n_rows = len(U_values)
    n_cols = len(V_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, 1.6 * n_rows),
        sharey="row", squeeze=False,
    )

    for row, U in enumerate(U_values):
        for col, V in enumerate(V_values):
            ax = axes[row, col]
            uv_key = f"{U}_{V}"
            mu_arr = mu_arrays.get((U, V))
            if mu_arr is None:
                continue

            # Plot DMRG reference
            # get_gnd returns filling per site (np.mean over sites), no /L needed
            if uv_key in dmrg_fill:
                dmrg_nu = np.array(dmrg_fill[uv_key])
                ax.plot(mu_arr, dmrg_nu, color="0.55", ls="--", lw=1.0,
                        zorder=2,
                        label=r"DMRG" if row == 0 and col == 0 else None)

            # Plot cluster ED curves for each Nc
            for nc in cluster_sizes:
                nc_str = str(nc)
                if nc_str not in cf or uv_key not in cf[nc_str]:
                    continue
                nu = np.array(cf[nc_str][uv_key])

                color = nc_colors[nc]
                ax.plot(mu_arr, nu, color=color, lw=1.3,
                        zorder=5 + nc,
                        label=(rf"$N_c={nc}$"
                               if row == 0 and col == 0 else None))

            # Axis formatting
            if row == 0:
                V_label = rf"$V={V:.0f}$" if V == int(V) else rf"$V={V}$"
                if V < 1e-3:
                    V_label = r"$V\!\approx\!0$"
                ax.set_title(V_label)
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mu_0/t$")
            if col == 0:
                U_label = rf"$U={U:.0f}$" if U == int(U) else rf"$U={U}$"
                ax.set_ylabel(r"$\nu$", fontsize=YLABEL_FONTSIZE)
                ax.text(-0.22, 0.5, U_label, transform=ax.transAxes,
                        fontsize=7, va="center", ha="center", rotation=90)

    # Single legend below centre
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   fontsize=6.5, ncol=len(cluster_sizes) + 1, frameon=False,
                   handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

    beta_frac = Fraction(*v_sep)
    fig.suptitle(
        rf"Filling $\nu(\mu_0)$: cluster ED vs DMRG, "
        rf"$\beta={beta_frac}$, $L={L}$",
        fontsize=9, y=1.01,
    )
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.25, wspace=0.20, bottom=0.10)

    _save(fig, "fig_v_compressibility",
          "current/v_compressibility_L48.pdf")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Appendix Fig 2d: Small-U absolute energy (V=0, U∈[0,1])
# ═════════════════════════════════════════════════════════════════════

def plot_hub_smallU_abs_energy(pickle_path=None):
    """
    App Fig 2d: Absolute energy E vs U/t for U∈[0,1], V=0.
    Same layout as plot_hub_comparison but y-axis is raw energy (not relative error).
    Rows = fillings, columns = Nc.
    Lines = interaction separations (YlOrRd gradient, maximal = black).
    DMRG reference = grey dashed.
    """
    path = Path(pickle_path) if pickle_path else HUB_SMALLU_DATA
    with open(path, "rb") as f:
        data = pickle.load(f)

    U = np.array(data["U_values"])
    sizes = data["cluster_sizes"]
    seps = data["int_sep_ratios_by_Nc"]
    ce = data["cluster_energies"]
    fde = data["finite_dmrg_energies"]
    L = data["parameters"]["L"]
    fillings = ["half", "quarter"]
    dmrg = {f: np.array([fde[ui][fi] for ui in range(len(U))])
            for fi, f in enumerate(fillings)}

    n_nc = len(sizes)
    fig, axes = plt.subplots(2, n_nc, figsize=(FULL_WIDTH, 3.8),
                             sharex="col", sharey="row", squeeze=False)

    for col, nc_str in enumerate(map(str, sizes)):
        nc = int(nc_str)
        sep_list = seps[nc_str]
        n_seps = len(sep_list)
        sorted_idx = sorted(range(n_seps),
                            key=lambda i: sep_list[i][0] / sep_list[i][1])
        rank = {i: r for r, i in enumerate(sorted_idx)}
        colors = gradient_colors(max(n_seps - 1, 1))

        for row, filling in enumerate(fillings):
            ax = axes[row, col]
            E_dmrg = dmrg[filling]

            # Plot DMRG reference
            ax.plot(U, E_dmrg, color="0.55", ls="--", lw=1.0, zorder=2,
                    label=r"DMRG" if row == 0 and col == 0 else None)

            for si, sep in enumerate(sep_list):
                sep_key = f"{sep[0]}_{sep[1]}"
                E_cl = np.array(ce[nc_str][sep_key][filling])

                maximal = (rank[si] == n_seps - 1)
                if maximal:
                    color = "black"
                else:
                    color = colors[rank[si]]

                label = (r"$\Delta\!=\!" + fmt_sep(tuple(sep)) + r"$"
                         + (r" (max)" if maximal else ""))

                ax.plot(U, E_cl, marker="o", linestyle="-",
                        color=color, lw=1.6 if maximal else 1.3,
                        zorder=10 if maximal else 5 + rank[si],
                        markersize=3.5,
                        label=label if row == 0 else None)

            if row == 0: ax.set_title(rf"$N_c={nc}$")
            if row == 1: ax.set_xlabel(r"$U/t$")
            if col == 0: ax.set_ylabel(r"$E^0/L$", fontsize=YLABEL_FONTSIZE)

            _panel_label(ax, row * n_nc + col)

        # Legend below bottom subplot
        handles, labels = axes[0, col].get_legend_handles_labels()
        axes[1, col].legend(handles, labels,
                            loc="upper center", bbox_to_anchor=(0.5, -0.32),
                            fontsize=6.5, ncol=1, frameon=False,
                            handlelength=1.5, handletextpad=0.4,
                            columnspacing=0.8)

    _filling_annotations(axes)
    fig.suptitle(
        rf"Absolute energy, $V=0$, $U\in[0,1]$, $L={L}$",
        fontsize=9, y=1.02)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.22)
    _save(fig, "fig_hub_smallU_abs_energy",
          "current/hub_smallU_abs_energy_L48.pdf")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Appendix Fig 2e: Small-U filling ν(μ₀) (V=0, U∈[0,1])
# ═════════════════════════════════════════════════════════════════════

def plot_hub_smallU_compressibility(pickle_path=None):
    """
    App Fig 2e: Filling ν vs μ₀ for U∈[0,1], V=0.
    Same layout as plot_hub_compressibility but with fine U grid.
    Rows = U values, Columns = Nc.
    Lines = interaction separations (YlOrRd gradient, maximal = black).
    DMRG reference = grey dashed.
    """
    path = Path(pickle_path) if pickle_path else COMPRESS_SMALLU_DATA
    with open(path, "rb") as f:
        data = pickle.load(f)

    U_values = data["U_values"]
    cluster_sizes = data["cluster_sizes"]
    int_sep_ratios = data["int_sep_ratios_by_Nc"]
    mu_arrays = {float(k): np.array(v) for k, v in data["mu_arrays"].items()}
    cf = data["cluster_fillings"]
    dmrg_fill = data.get("finite_dmrg_fillings", {})
    params = data["parameters"]
    L = params["L"]

    n_rows = len(U_values)
    n_cols = len(cluster_sizes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, 1.6 * n_rows),
        sharex="row", sharey="row", squeeze=False,
    )

    for col, nc in enumerate(cluster_sizes):
        nc_str = str(nc)
        sep_list = [tuple(s) for s in int_sep_ratios[nc_str]]
        n_seps = len(sep_list)

        sorted_idx = sorted(range(n_seps),
                            key=lambda i: sep_list[i][0] / sep_list[i][1])
        rank = {i: r for r, i in enumerate(sorted_idx)}
        colors = gradient_colors(max(n_seps - 1, 1))

        for row, U in enumerate(U_values):
            ax = axes[row, col]
            u_str = str(U)
            mu_arr = mu_arrays[U]

            # Plot DMRG reference
            if u_str in dmrg_fill:
                dmrg_nu = np.array(dmrg_fill[u_str])
                ax.plot(mu_arr, dmrg_nu, color="0.55", ls="--", lw=1.0,
                        zorder=2,
                        label=r"DMRG" if row == 0 and col == 0 else None)

            # Plot cluster ED curves
            for si, sep in enumerate(sep_list):
                sep_key = f"{sep[0]}_{sep[1]}"
                if sep_key not in cf.get(nc_str, {}):
                    continue
                if u_str not in cf[nc_str][sep_key]:
                    continue

                nu = np.array(cf[nc_str][sep_key][u_str])

                maximal = (rank[si] == n_seps - 1)
                if maximal:
                    color = "black"
                    lw = 1.6
                else:
                    color = colors[rank[si]]
                    lw = 1.2

                label_str = (r"$\Delta\!=\!" + fmt_sep(sep) + r"$"
                             + (r" (max)" if maximal else ""))

                ax.plot(mu_arr, nu,
                        color=color, lw=lw,
                        zorder=10 if maximal else 5 + rank[si],
                        label=label_str if row == 0 else None)

            # Axis formatting
            if row == 0:
                ax.set_title(rf"$N_c={nc}$")
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mu_0/t$")
            if col == 0:
                U_label = rf"$U={U:.2f}$"
                ax.set_ylabel(r"$\nu$", fontsize=YLABEL_FONTSIZE)
                ax.text(-0.22, 0.5, U_label, transform=ax.transAxes,
                        fontsize=7, va="center", ha="center", rotation=90)

        # Legend below bottom subplot of each column
        handles, labels = axes[0, col].get_legend_handles_labels()
        if handles:
            axes[-1, col].legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, -0.45),
                fontsize=6, ncol=1, frameon=False,
                handlelength=1.5, handletextpad=0.4,
            )

    fig.suptitle(
        rf"Filling $\nu(\mu_0)$: cluster ED vs DMRG, $V=0$, $U\in[0,1]$, $L={L}$",
        fontsize=9, y=1.01,
    )
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.25, wspace=0.20, bottom=0.14)

    _save(fig, "fig_hub_smallU_compressibility",
          "current/hub_smallU_compressibility_L48.pdf")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Appendix: Filling relative error (mean |ν_cl − ν_DMRG| over μ sweep)
# ═════════════════════════════════════════════════════════════════════

YLABEL_FILL_ERR = r"$(\nu_{\mathrm{cl}} - \nu_{\mathrm{DMRG}})\,/\,\nu_{\mathrm{DMRG}}$"


def plot_hub_filling_relerr(pickle_path=None):
    """
    Relative error of filling sweep vs μ₀ for V=0 (Fig 2 compressibility data).
    Same layout as plot_hub_compressibility: rows = U, cols = Nc, lines = Δ.
    Y-axis is |ν_cl − ν_DMRG| / ν_DMRG instead of raw ν.
    """
    path = Path(pickle_path) if pickle_path else COMPRESS_DATA
    with open(path, "rb") as f:
        data = pickle.load(f)

    U_values = data["U_values"]
    cluster_sizes = data["cluster_sizes"]
    int_sep_ratios = data["int_sep_ratios_by_Nc"]
    mu_arrays = {float(k): np.array(v) for k, v in data["mu_arrays"].items()}
    cf = data["cluster_fillings"]
    dmrg_fill = data.get("finite_dmrg_fillings", {})
    L = data["parameters"]["L"]

    n_rows = len(U_values)
    n_cols = len(cluster_sizes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, 1.6 * n_rows),
        sharex="row", sharey="row", squeeze=False,
    )

    for col, nc in enumerate(cluster_sizes):
        nc_str = str(nc)
        sep_list = [tuple(s) for s in int_sep_ratios[nc_str]]
        n_seps = len(sep_list)
        sorted_idx = sorted(range(n_seps),
                            key=lambda i: sep_list[i][0] / sep_list[i][1])
        rank = {i: r for r, i in enumerate(sorted_idx)}
        colors = gradient_colors(max(n_seps - 1, 1))

        for row, U in enumerate(U_values):
            ax = axes[row, col]
            u_str = str(U)
            mu_arr = mu_arrays[U]

            if u_str not in dmrg_fill:
                continue
            dmrg_nu = np.array(dmrg_fill[u_str])

            for si, sep in enumerate(sep_list):
                sep_key = f"{sep[0]}_{sep[1]}"
                if (nc_str not in cf or sep_key not in cf.get(nc_str, {})
                        or u_str not in cf[nc_str][sep_key]):
                    continue

                nu_cl = np.array(cf[nc_str][sep_key][u_str])

                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = np.where(
                        np.abs(dmrg_nu) > 1e-6,
                        np.abs((nu_cl - dmrg_nu) / dmrg_nu) * 100,
                        np.nan,
                    )

                maximal = (rank[si] == n_seps - 1)
                color = "black" if maximal else colors[rank[si]]
                lw = 1.6 if maximal else 1.2

                label_str = (r"$\Delta\!=\!" + fmt_sep(sep) + r"$"
                             + (r" (max)" if maximal else ""))

                ax.plot(mu_arr, rel_err,
                        color=color, lw=lw,
                        zorder=10 if maximal else 5 + rank[si],
                        label=label_str if row == 0 else None)

            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            if row == 0:
                ax.set_title(rf"$N_c={nc}$")
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mu_0/t$")
            if col == 0:
                ax.set_ylabel(YLABEL_FILL_ERR, fontsize=YLABEL_FONTSIZE)
                U_label = rf"$U={U:.0f}$" if U == int(U) else rf"$U={U}$"
                ax.text(-0.22, 0.5, U_label, transform=ax.transAxes,
                        fontsize=7, va="center", ha="center", rotation=90)

        handles, labels = axes[0, col].get_legend_handles_labels()
        if handles:
            axes[-1, col].legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, -0.45),
                fontsize=6, ncol=1, frameon=False,
                handlelength=1.5, handletextpad=0.4,
            )

    fig.suptitle(
        rf"Filling relative error, $\lambda=0$, $L={L}$",
        fontsize=9, y=1.01,
    )
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.25, wspace=0.20, bottom=0.14)
    _save(fig, "fig_hub_filling_relerr")
    return fig


def plot_v_filling_relerr(pickle_path=None):
    """
    Relative error of filling sweep vs μ₀ for finite V (Fig 4 compressibility data).
    Same layout as plot_v_compressibility: rows = U, cols = V, lines = Nc.
    Y-axis is |ν_cl − ν_DMRG| / ν_DMRG instead of raw ν.
    """
    if pickle_path is None:
        candidates = sorted(PLOTS_DIR.glob("fig4_compressibility_combined_*.pkl"), reverse=True)
        if not candidates:
            print("  No Fig 4 compressibility pickle found, skipping.")
            return None
        pickle_path = candidates[0]

    with open(Path(pickle_path), "rb") as f:
        data = pickle.load(f)

    U_values = data["U_values"]
    V_values = data["V_values"]
    cluster_sizes = data["cluster_sizes"]
    cf = data["cluster_fillings"]
    dmrg_fill = data.get("finite_dmrg_fillings", {})
    L = data["parameters"]["L"]
    v_sep = tuple(data["parameters"]["v_sep_ratio"])

    mu_arrays = {}
    for k, v in data["mu_arrays"].items():
        u_val, v_val = k.split("_")
        mu_arrays[(float(u_val), float(v_val))] = np.array(v)

    nc_colors = nc_gradient(cluster_sizes)

    n_rows = len(U_values)
    n_cols = len(V_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, 1.6 * n_rows),
        sharey="row", squeeze=False,
    )

    for row, U in enumerate(U_values):
        for col, V in enumerate(V_values):
            ax = axes[row, col]
            uv_key = f"{U}_{V}"
            mu_arr = mu_arrays.get((U, V))
            if mu_arr is None:
                continue

            if uv_key not in dmrg_fill:
                continue
            dmrg_nu = np.array(dmrg_fill[uv_key])

            for nc in cluster_sizes:
                nc_str = str(nc)
                if nc_str not in cf or uv_key not in cf[nc_str]:
                    continue
                nu_cl = np.array(cf[nc_str][uv_key])

                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = np.where(
                        np.abs(dmrg_nu) > 1e-6,
                        np.abs((nu_cl - dmrg_nu) / dmrg_nu) * 100,
                        np.nan,
                    )

                color = nc_colors[nc]
                ax.plot(mu_arr, rel_err, color=color, lw=1.3,
                        zorder=5 + nc,
                        label=(rf"$N_c={nc}$"
                               if row == 0 and col == 0 else None))

            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            if row == 0:
                V_label = rf"$\lambda={V:.0f}$" if V == int(V) else rf"$\lambda={V}$"
                if V < 1e-3:
                    V_label = r"$\lambda\!\approx\!0$"
                ax.set_title(V_label)
            if row == n_rows - 1:
                ax.set_xlabel(r"$\mu_0/t$")
            if col == 0:
                U_label = rf"$U={U:.0f}$" if U == int(U) else rf"$U={U}$"
                ax.set_ylabel(YLABEL_FILL_ERR, fontsize=YLABEL_FONTSIZE)
                ax.text(-0.22, 0.5, U_label, transform=ax.transAxes,
                        fontsize=7, va="center", ha="center", rotation=90)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   fontsize=6.5, ncol=len(cluster_sizes), frameon=False,
                   handlelength=1.5, handletextpad=0.4, columnspacing=1.0)

    beta_frac = Fraction(*v_sep)
    fig.suptitle(
        rf"Filling relative error, $\beta={beta_frac}$, $L={L}$",
        fontsize=9, y=1.01,
    )
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.25, wspace=0.20, bottom=0.10)
    _save(fig, "fig_v_filling_relerr")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Appendix: Absolute energy plots (raw E/L instead of relative error)
# ═════════════════════════════════════════════════════════════════════

YLABEL_ABS = r"$E^0 / L$"


def plot_hub_abs_energy():
    """Absolute energy E/L vs U for V=0 (Fig 2 data, non-relative)."""
    with open(HUB_DATA, "rb") as f:
        data = pickle.load(f)

    U = np.array(data["U_values"])
    sizes = data["cluster_sizes"]
    seps = data["int_sep_ratios_by_Nc"]
    ce = data["cluster_energies"]
    fillings = ["half", "quarter"]
    L = data["parameters"]["L"]

    fde = data["finite_dmrg_energies"]
    ref = {f: np.array([fde[ui][fi] for ui in range(len(U))])
           for fi, f in enumerate(fillings)}

    n_nc = len(sizes)
    fig, axes = plt.subplots(2, n_nc, figsize=(FULL_WIDTH, 3.8),
                             sharex="col", sharey="row", squeeze=False)

    for col, nc_str in enumerate(map(str, sizes)):
        nc = int(nc_str)
        sep_list = seps[nc_str]
        n_seps = len(sep_list)
        sorted_idx = sorted(range(n_seps),
                            key=lambda i: sep_list[i][0] / sep_list[i][1])
        rank = {i: r for r, i in enumerate(sorted_idx)}
        colors = gradient_colors(n_seps - 1)

        for row, filling in enumerate(fillings):
            ax = axes[row, col]
            E_ref = ref[filling]

            # Plot DMRG reference
            ax.plot(U, E_ref, color="grey", ls="--", lw=1.0, zorder=1,
                    label="DMRG" if row == 0 and col == 0 else None)

            for si, sep in enumerate(sep_list):
                sep_key = f"{sep[0]}_{sep[1]}"
                E_cl = np.array(ce[nc_str][sep_key][filling])

                maximal = (rank[si] == n_seps - 1)
                color = "black" if maximal else colors[rank[si]]

                label = (r"$\Delta\!=\!" + fmt_sep(tuple(sep)) + r"$"
                         + (r" (max)" if maximal else ""))

                ax.plot(U, E_cl, marker="o", linestyle="-",
                        color=color, lw=1.6 if maximal else 1.3,
                        zorder=10 if maximal else 5 + rank[si],
                        markersize=3.5,
                        label=label if row == 0 else None)

            ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20, 25, 30]))
            if row == 0: ax.set_title(rf"$N_c={nc}$")
            if row == 1: ax.set_xlabel(r"$U/t$")
            if col == 0: ax.set_ylabel(YLABEL_ABS, fontsize=YLABEL_FONTSIZE)
            _panel_label(ax, row * n_nc + col)

        handles, labels = axes[0, col].get_legend_handles_labels()
        axes[1, col].legend(handles, labels,
                            loc="upper center", bbox_to_anchor=(0.5, -0.32),
                            fontsize=6.5, ncol=1, frameon=False,
                            handlelength=1.5, handletextpad=0.4,
                            columnspacing=0.8)

    _filling_annotations(axes)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.22)
    _save(fig, "fig_hub_abs_energy")
    return fig


def plot_v_abs_energy(beta_str, V_subset=None, suffix=""):
    """Absolute energy E/L vs U for Fig 4 data.

    Axes swapped from main text: x-axis = U, columns = V values,
    lines = different Nc.
    """
    with open(DATA_BOTH_FILLINGS[beta_str], "rb") as f:
        data = pickle.load(f)

    params = data["parameters"]
    v_sep = params["v_sep_ratio"]
    L = params["L"]

    U_values = np.array(data["U_values"])
    ce = data["cluster_energies"]
    fde = data["finite_dmrg_energies"]

    cluster_sizes = sorted([int(k) for k in ce.keys()])
    nc_colors = nc_gradient(cluster_sizes)

    first_nc = list(ce.keys())[0]
    V_keys = list(ce[first_nc].keys())
    V_values = np.array([float(v) for v in V_keys])

    if V_subset is not None:
        v_mask = [i for i, v in enumerate(V_values) if v in V_subset]
    else:
        v_mask = list(range(len(V_values)))
    n_cols = len(v_mask)
    filling_map = {0: "half", 1: "quarter"}

    fig, axes = plt.subplots(2, n_cols, figsize=(FULL_WIDTH, 3.8),
                             sharex=True, sharey="row", squeeze=False)

    for col_idx, v_idx in enumerate(v_mask):
        V_val = V_values[v_idx]
        V_key = V_keys[v_idx]

        for row in range(2):
            ax = axes[row, col_idx]
            filling = filling_map[row]
            fill_idx = row

            # DMRG ref: fde[u_idx][v_idx][fill_idx] → sweep over U
            E_dmrg = np.array([fde[u_idx][v_idx][fill_idx]
                               for u_idx in range(len(U_values))])

            ax.plot(U_values, E_dmrg, color="grey", ls="--", lw=1.0,
                    zorder=1,
                    label="DMRG" if row == 0 and col_idx == 0 else None)

            for nc_str in ce.keys():
                nc = int(nc_str)
                # ce[nc][V_key][filling][u_idx] → sweep over U
                E_nc = np.array(ce[nc_str][V_key][filling])

                color = nc_colors.get(nc, "gray")
                ax.plot(U_values, E_nc, marker="o", markersize=3.5,
                        color=color, lw=1.3,
                        label=(rf"$N_c={nc}$"
                               if row == 0 and col_idx == 0 else None))

            V_label = rf"$\lambda={V_val:.0f}$" if V_val == int(V_val) else rf"$\lambda={V_val}$"
            if V_val < 1e-3:
                V_label = r"$\lambda\approx 0$"
            if row == 0: ax.set_title(V_label)
            if row == 1: ax.set_xlabel(r"$U/t$")
            if col_idx == 0: ax.set_ylabel(YLABEL_ABS, fontsize=YLABEL_FONTSIZE)
            _panel_label(ax, row * n_cols + col_idx)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02),
               fontsize=6.5, ncol=len(cluster_sizes) + 1, frameon=False,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.2)

    _filling_annotations(axes)

    beta_frac = Fraction(*v_sep)
    fig.suptitle(
        rf"$\beta = {beta_frac}$ ($Q={fmt_sep(v_sep)}$), $L={L}$",
        fontsize=9, y=1.02)

    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.12)
    _save(fig, f"fig_v_abs_energy_{beta_str}{suffix}")
    return fig


def plot_fixed_supercluster_abs_energy(V_subset=None):
    """Absolute energy E/L vs U for Fig 5 data.

    Axes swapped from main text: x-axis = U, columns = V values,
    lines = different (Nc, Δ) configurations at the largest supercluster size.
    """
    filling_files = [
        ("half", FIXED_SC_HALF),
        ("quarter", FIXED_SC_QUARTER),
    ]
    datasets = {}
    for fill_label, path in filling_files:
        with open(path, "rb") as f:
            datasets[fill_label] = pickle.load(f)

    ref = datasets["half"]
    U_values = np.array(ref["U_values"])
    V_values = np.array(ref["V_values"])
    sc_sizes = sorted(ref["supercluster_sizes"])
    sc_int_seps = ref["supercluster_int_seps"]
    L = ref["parameters"]["L"]

    if V_subset is not None:
        v_mask = [i for i, v in enumerate(V_values) if v in V_subset]
    else:
        v_mask = list(range(len(V_values)))
    n_cols = len(v_mask)

    # Use the largest supercluster size for this plot
    sc_target = sc_sizes[-1]
    sc_str = str(sc_target)
    configs = sc_int_seps[sc_str]
    max_nc = max(cfg[0] for cfg in configs)
    sorted_configs = sorted(configs, key=lambda c: (c[0] == max_nc, c[0]))
    non_max_count = sum(1 for c in sorted_configs if c[0] != max_nc)
    nm_colors = gradient_colors(max(non_max_count, 1))

    fillings = ["half", "quarter"]
    fig, axes = plt.subplots(2, n_cols, figsize=(FULL_WIDTH, 4.2),
                             sharex=True, sharey="row", squeeze=False)

    for col_idx, v_idx in enumerate(v_mask):
        V_val = V_values[v_idx]

        for row, fill_label in enumerate(fillings):
            ax = axes[row, col_idx]
            data = datasets[fill_label]
            ce = data["cluster_energies"][sc_str]
            fde = np.array(data["finite_dmrg_energies"])
            # fde shape: (n_V, n_U) → pick row v_idx
            E_dmrg = fde[v_idx, :]

            ax.plot(U_values, E_dmrg, color="grey", ls="--", lw=1.0,
                    zorder=1,
                    label="DMRG" if row == 0 and col_idx == 0 else None)

            nm_ci = 0
            for ci, (nc, int_sep_list) in enumerate(sorted_configs):
                int_sep = tuple(int_sep_list)
                config_key = f"{nc}_{int_sep[0]}_{int_sep[1]}"
                if config_key not in ce:
                    continue

                # ce[config] shape: (n_V, n_U) → pick row v_idx
                E_nc = np.array(ce[config_key])[v_idx, :]

                is_max = (nc == max_nc)
                if is_max:
                    color = "black"
                    lw = 1.6
                else:
                    color = nm_colors[nm_ci]
                    nm_ci += 1
                    lw = 1.3

                sep_label = fmt_sep(int_sep)
                label = (rf"$N_c\!=\!{nc}$, $\Delta\!=\!{sep_label}$"
                         + (r" (max)" if is_max else ""))
                ax.plot(U_values, E_nc, marker="o", markersize=3.5,
                        color=color, lw=lw,
                        label=label if row == 0 else None,
                        zorder=10 if is_max else 5 + ci)

            V_label = rf"$\lambda={V_val:.0f}$" if V_val == int(V_val) else rf"$\lambda={V_val}$"
            if V_val < 1e-3:
                V_label = r"$\lambda\approx 0$"
            if row == 0: ax.set_title(V_label)
            if row == 1: ax.set_xlabel(r"$U/t$")
            if col_idx == 0: ax.set_ylabel(YLABEL_ABS, fontsize=YLABEL_FONTSIZE)
            _panel_label(ax, row * n_cols + col_idx)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02),
               fontsize=6.5, ncol=len(sorted_configs) + 1, frameon=False,
               handlelength=1.5, handletextpad=0.4, columnspacing=0.8)

    _filling_annotations(axes)
    fig.suptitle(
        rf"$N_{{\mathrm{{SC}}}}={sc_target}$, $\beta=1/2$, $L={L}$",
        fontsize=9, y=1.02)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.18, wspace=0.32, bottom=0.18)
    _save(fig, "fig_fixed_supercluster_abs_energy")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Legacy functions (original Fig 4 & Fig 5 format, preserved for ref)
# ═════════════════════════════════════════════════════════════════════

_LEGACY_NC_COLORS = {
    2: "#4477AA", 3: "#228833", 4: "#EE8833",
    6: "#EE6677", 8: "#AA3377", 9: "#66CCEE",
    10: "#BBBBBB", 12: "#CCBB44",
}
_LEGACY_MARKERS = ["o", "s", "^", "D", "v", "P", "<", ">", "X", "h"]
_LEGACY_LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def plot_v_convergence_legacy(beta_str, U_subset, suffix="", image_name=None):
    """Original Fig 4 format: Tol bright per-Nc colors, varied markers."""
    with open(DATA_BOTH_FILLINGS[beta_str], "rb") as f:
        data = pickle.load(f)
    params = data["parameters"]
    v_sep = params["v_sep_ratio"]
    L = params["L"]
    U_values = data["U_values"]
    ce = data["cluster_energies"]
    fde = data["finite_dmrg_energies"]
    first_nc = list(ce.keys())[0]
    V_keys = list(ce[first_nc].keys())
    V_values = np.array([float(v) for v in V_keys])
    u_mask = [i for i, u in enumerate(U_values) if u in U_subset]
    n_cols = len(u_mask)
    fig, axes = plt.subplots(2, n_cols, figsize=(7.0, 3.0),
                             sharex=True, squeeze=False)
    filling_map = {0: "half", 1: "quarter"}
    for col_idx, u_idx in enumerate(u_mask):
        U = U_values[u_idx]
        for row in range(2):
            ax = axes[row, col_idx]
            filling = filling_map[row]
            E_dmrg = np.array([fde[u_idx][vi][row]
                               for vi in range(len(V_values))])
            for nc_str in ce.keys():
                nc = int(nc_str)
                E_nc = np.array([ce[nc_str][vk][filling][u_idx]
                                 for vk in V_keys])
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = np.abs((E_nc - E_dmrg) / E_dmrg) * 100
                color = _LEGACY_NC_COLORS.get(nc, "gray")
                ax.plot(V_values, rel_err, marker="o", markersize=3,
                        color=color, lw=1.2,
                        label=(rf"$N_c={nc}$"
                               if row == 0 and col_idx == 0 else None))
            ax.set_ylim(bottom=-0.3)
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            U_label = rf"$U={U:.0f}$" if U == int(U) else rf"$U={U}$"
            if row == 0: ax.set_title(U_label)
            if row == 1: ax.set_xlabel(r"$\lambda/t$")
            if col_idx == 0: ax.set_ylabel("Relative error")
            idx = row * n_cols + col_idx
            ax.text(0.04, 0.95, f"({chr(ord('a') + idx)})",
                    transform=ax.transAxes, fontsize=7,
                    va="top", ha="left", fontweight="bold")
    axes[0, 0].legend(loc="upper right")
    for row, lbl in enumerate([r"$n=1$", r"$n=1/2$"]):
        axes[row, -1].annotate(lbl, xy=(1.04, 0.5),
                               xycoords="axes fraction", rotation=270,
                               va="center", ha="left", fontsize=8)
    beta_frac = Fraction(*v_sep)
    fig.suptitle(rf"$\beta = {beta_frac}$ ($Q={fmt_sep(v_sep)}$), $L={L}$",
                 fontsize=9, y=1.02)
    fig.subplots_adjust(hspace=0.15, wspace=0.25)
    name = f"fig_v_convergence_{beta_str}{suffix}_legacy"
    _save(fig, name, image_name)
    return fig


def plot_fixed_supercluster_legacy():
    """Original Fig 5 format: Tol bright colors, varied markers/linestyles."""
    with open(MERGED_HALF["1-2"], "rb") as f:
        data = pickle.load(f)
    U_values = np.array(data["U_values"])
    V_values = np.array(data["V_values"])
    ce = data["cluster_energies"]
    fde = np.array(data["finite_dmrg_energies"])
    U_target = 5.0
    u_idx = int(np.where(U_values == U_target)[0][0])
    sc_configs = {
        2: [(2, (1, 2), True)],
        4: [(4, (1, 4), True), (2, (1, 4), False)],
        6: [(6, (1, 6), True), (2, (1, 6), False)],
        8: [(8, (1, 8), True), (4, (1, 8), False), (2, (1, 8), False)],
    }
    sc_sizes = sorted(sc_configs.keys())
    n_cols = len(sc_sizes)
    fig, axes = plt.subplots(1, n_cols, figsize=(7.0, 1.8),
                             sharex=True, squeeze=False)
    for col, sc in enumerate(sc_sizes):
        ax = axes[0, col]
        E_dmrg = fde[u_idx]
        for ci, (nc, int_sep, is_max) in enumerate(sc_configs[sc]):
            if nc not in ce:
                continue
            E_nc = np.array([ce[nc][V][u_idx] if V in ce[nc] else np.nan
                             for V in V_values])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.abs((E_nc - E_dmrg) / E_dmrg) * 100
            color = "black" if is_max else _LEGACY_NC_COLORS.get(nc, "gray")
            ls = "-" if is_max else _LEGACY_LS[ci]
            marker = "s" if is_max else _LEGACY_MARKERS[ci]
            lw = 1.4 if is_max else 1.0
            sep_label = fmt_sep(int_sep)
            label = (rf"$N_c\!=\!{nc}$, $\Delta\!=\!{sep_label}$"
                     + (r" \textbf{(max)}" if is_max else ""))
            ax.plot(V_values, rel_err, marker=marker, markersize=3,
                    color=color, ls=ls, lw=lw, label=label)
        ax.set_ylim(bottom=-0.5)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_title(rf"$N_{{SC}}={sc}$")
        ax.set_xlabel(r"$\lambda/t$")
        if col == 0: ax.set_ylabel("Relative error")
        ax.legend(loc="best", fontsize=5)
        ax.text(0.04, 0.95, f"({chr(ord('a') + col)})",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left", fontweight="bold")
    fig.suptitle(rf"Fixed supercluster size, $U={U_target:.0f}$, $\beta=1/2$, "
                 r"half-filling", fontsize=9, y=1.04)
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "fig_fixed_supercluster_legacy")
    return fig


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    OUT.mkdir(exist_ok=True)
    print("Generating publication figures...")

    print("  [1/10] Hub comparison (V=0)...")
    plot_hub_comparison()

    print("  [2/10] V-convergence beta=1/2 (low U)...")
    plot_v_convergence("1-2", U_subset=[0.0, 1.0, 2.0, 3.0],
                       image_name="v_convergence_pi_relerror.png")

    print("  [3/10] V-convergence beta=1/2 (high U)...")
    plot_v_convergence("1-2", U_subset=[5.0, 7.0, 10.0, 20.0],
                       suffix="_highU",
                       image_name="v_convergence_pi_relerror_highU.png")

    print("  [4/10] V-convergence beta=1/3...")
    plot_v_convergence("1-3", U_subset=[0.0, 1.0, 3.0, 5.0],
                       image_name="v_convergence_2pi3_relerror.png")

    print("  [5/10] V-convergence beta=1/4...")
    plot_v_convergence("1-4", U_subset=[0.0, 1.0, 3.0, 5.0],
                       image_name="v_convergence_pi4_48.png")

    print("  [6/10] Fixed supercluster comparison...")
    plot_fixed_supercluster()

    print("  [7/10] Hub compressibility (appendix)...")
    plot_hub_compressibility()

    print("  [8/10] V-convergence compressibility (appendix)...")
    plot_v_compressibility()

    print("  [9/10] Small-U absolute energy (appendix)...")
    plot_hub_smallU_abs_energy()

    print("  [10/10] Small-U compressibility (appendix)...")
    plot_hub_smallU_compressibility()

    print(f"\nAll figures saved to {OUT}")
    plt.close("all")


if __name__ == "__main__":
    main()
