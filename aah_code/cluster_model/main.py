"""
File for running mainline scripts
Eventually this will just be simple call to the ClusterModel class
but for now I just want to get a single spectrum run going.
"""

import os
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        return iterable

from aah_code.cluster_model.compare_old_new_line_plots import (
    compare_old_new_line_plots,
    get_old_method_results,
    save_line_plots,
)
from aah_code.cluster_model.convergence_workflow import (
    SweepGrid,
    run_convergence_study,
)
from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.cluster_model.plots import compare_int_seps_with_dmrg, compare_cluster_sizes_with_dmrg, compare_U_values_with_dmrg, compare_filling_with_int_cluster_sizes, compare_compressibility_with_int_cluster_sizes, compare_compressibility_cluster_sizes, compare_compressibility_fixed_supercluster
from aah_code.real_space_dmrg import run_dmrg_method
import pickle

def three_way_comparison_with_dmrg(
    int_sep_ratio: Tuple[int, int],
    v_sep_ratio: Tuple[int, int],
    U_values: np.ndarray,
    V_values: np.ndarray,
    t: float = 1.0,
    L: int = 20,
    Nc: int = 2,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    output_dir: str = 'large_files/plots',
    show_plots: bool = True
):
    """
    Three-way comparison: Old QSpin, New General QSpin, and DMRG
    Creates line plots with all three methods for comparison.
    
    Note: DMRG calculation is independent of int_sep_ratio (only affected by V pattern)
    For v_sep_ratio = (1,2), this corresponds to π modulation (staggered V)
    """
    
    print("=" * 60)
    print("Three-Way Comparison: Old vs New QSpin vs DMRG")
    print("=" * 60)
    print(f"System: L={L}, Nc={Nc}, t={t}")
    print(f"New method: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print(f"DMRG: chi={chi}")
    print(f"U values: {U_values}")
    print(f"V values: {V_values}")
    
    # Storage for all results
    all_results = {}
    
    for V in tqdm(V_values, desc="V values"):
        all_results[V] = {
            'energies_old': [],
            'energies_new': [],
            'energies_dmrg': [],
            'fillings_old': [],
            'fillings_new': [],
            'fillings_dmrg': []
        }
        
        for U in U_values:
            mu_0 = U / 2  # Half-filling
            
            # Old QSpin method (selected based on ratios)
            energy_old_per_site, filling_old_per_site = get_old_method_results(
                U=U, V=V, t=t, L=L, Nc=Nc,
                int_sep_ratio=int_sep_ratio,
                v_sep_ratio=v_sep_ratio
            )
            energy_old_subtracted = energy_old_per_site + mu_0 * filling_old_per_site
            
            # New General QSpin method
            physical_params = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
            run_config = ClusterModelConfig(
                L=L,
                int_cluster_size=Nc,
                cluster_separation_ratio=int_sep_ratio,
                V_separation_ratio=v_sep_ratio,
                ham_lib='quspin',
                physical_params=physical_params,
                model_bc='periodic',
                int_cluster_bc='periodic',
                super_cluster_bc='periodic'
            )
            system_expectations_new, _ = get_general_expectations(run_config)
            energy_new, filling_new, _ = system_expectations_new
            energy_new_subtracted = (energy_new + mu_0 * filling_new) / L
            filling_new_per_site = filling_new / L
            
            # DMRG method
            # Note: DMRG now uses v_sep_ratio for arbitrary modulation
            energy_dmrg, filling_dmrg, _ = run_dmrg_method(U, mu_0, V, v_sep_ratio, t, L, chi)
            # DMRG returns per-site quantities already
            energy_dmrg_subtracted = energy_dmrg + mu_0 * filling_dmrg
            
            # Store results
            all_results[V]['energies_old'].append(energy_old_subtracted)
            all_results[V]['energies_new'].append(energy_new_subtracted)
            all_results[V]['energies_dmrg'].append(energy_dmrg_subtracted)
            all_results[V]['fillings_old'].append(filling_old_per_site)
            all_results[V]['fillings_new'].append(filling_new_per_site)
            all_results[V]['fillings_dmrg'].append(filling_dmrg)
    
    # Create plots
    figures = create_three_way_plots(U_values, V_values, all_results, output_dir, show_plots)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for V in V_values:
        energies_old = np.array(all_results[V]['energies_old'])
        energies_new = np.array(all_results[V]['energies_new'])
        energies_dmrg = np.array(all_results[V]['energies_dmrg'])
        
        old_vs_dmrg = np.nanmean(np.abs(energies_old - energies_dmrg))
        new_vs_dmrg = np.nanmean(np.abs(energies_new - energies_dmrg))
        old_vs_new = np.nanmean(np.abs(energies_old - energies_new))
        
        print(f"V={V:.2f}:")
        print(f"  |Old-DMRG|={old_vs_dmrg:.6f}, |New-DMRG|={new_vs_dmrg:.6f}, |Old-New|={old_vs_new:.6f}")
    
    return figures, all_results


def create_three_way_plots(U_values, V_values, all_results, output_dir='large_files/plots', show_plots=True):
    """Create line plots comparing all three methods."""
    
    # Group V values into chunks of 3
    n_v_per_fig = 3
    n_figures = np.ceil(len(V_values) / n_v_per_fig).astype(int)
    figures = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * n_v_per_fig
        end_idx = min(start_idx + n_v_per_fig, len(V_values))
        current_V_values = V_values[start_idx:end_idx]
        n_cols = len(current_V_values)
        
        # Create subplot titles
        energy_titles = [f'Energy/site vs U (V={V:.2f})' for V in current_V_values]
        filling_titles = [f'Filling/site vs U (V={V:.2f})' for V in current_V_values]
        
        fig = make_subplots(
            rows=2, cols=n_cols,
            subplot_titles=energy_titles + filling_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Define colors and styles
        colors = {
            'Old QSpin': 'blue',
            'New General': 'red',
            'DMRG': 'green'
        }
        
        for col_idx, V in enumerate(current_V_values):
            col = col_idx + 1
            
            # Energy plots (top row)
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['energies_old'],
                    mode='lines+markers', 
                    name='Old QSpin',
                    line=dict(color=colors['Old QSpin'], width=2),
                    marker=dict(size=6),
                    showlegend=(col_idx == 0)
                ),
                row=1, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['energies_new'],
                    mode='lines+markers', 
                    name='New General',
                    line=dict(color=colors['New General'], width=2, dash='dash'),
                    marker=dict(size=6, symbol='square'),
                    showlegend=(col_idx == 0)
                ),
                row=1, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['energies_dmrg'],
                    mode='lines+markers', 
                    name='DMRG',
                    line=dict(color=colors['DMRG'], width=2, dash='dot'),
                    marker=dict(size=6, symbol='diamond'),
                    showlegend=(col_idx == 0)
                ),
                row=1, col=col
            )
            
            # Filling plots (bottom row)
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['fillings_old'],
                    mode='lines+markers', 
                    name='Old QSpin',
                    line=dict(color=colors['Old QSpin'], width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['fillings_new'],
                    mode='lines+markers', 
                    name='New General',
                    line=dict(color=colors['New General'], width=2, dash='dash'),
                    marker=dict(size=6, symbol='square'),
                    showlegend=False
                ),
                row=2, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['fillings_dmrg'],
                    mode='lines+markers', 
                    name='DMRG',
                    line=dict(color=colors['DMRG'], width=2, dash='dot'),
                    marker=dict(size=6, symbol='diamond'),
                    showlegend=False
                ),
                row=2, col=col
            )
            
            # Update axes labels
            fig.update_xaxes(title_text='U', row=1, col=col)
            fig.update_xaxes(title_text='U', row=2, col=col)
            
            if col == 1:
                fig.update_yaxes(title_text='Energy/site', row=1, col=col)
                fig.update_yaxes(title_text='Filling/site', row=2, col=col)
        
        # Update layout
        title_text = f'Three-Way Comparison: Old vs New vs DMRG (Page {fig_idx+1}/{n_figures})'
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center'),
            height=700,
            width=400 * n_cols,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified'
        )
        
        # Save and/or show
        filename = f'three_way_comparison_page_{fig_idx+1}.html'
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved figure to {filepath}")
        
        if show_plots:
            fig.show()
        
        figures.append(fig)
    
    return figures


def old_new_quspin_comparison():
    """Run the old two-way comparison without DMRG."""
    # Define parameter ranges
    U_values = np.linspace(0, 5, 10)
    V_values = np.array([0.0, 1.0, 2.0, 3.0])
    
    # Generate, show, and save line plots
    figures = save_line_plots(
        U_values=U_values,
        V_values=V_values,
        t=1.0,
        L=20,
        Nc=2,
        int_sep_ratio=(1, 2),
        v_sep_ratio=(1, 2),
        output_prefix='old_vs_new_line_plots',
        output_dir='large_files/plots',
        save_html=True,
        show_plots=True
    )
    
    print("\nComparison complete!")
    return figures


def run_convergence_example(U_values,V_values):
    """
    Example entry point demonstrating the convergence workflow.
    """
    golden_ratio = 2.0 / (np.sqrt(5.0) - 1.0)
    # The AA modulation only depends on beta modulo 1, so we drop the integer part
    # to obtain the usual sequence of Hurwitz approximants (1, 1/2, 2/3, 3/5, ...).
    beta = golden_ratio % 1.0
    beta=1/4
    max_supercluster_size = 8

    #U_values = np.unique(
    #    np.concatenate([np.linspace(0, 1, 3)])
    #)
    
    #V_values = [1e-6, 1.0, 2.0]

    sweep = SweepGrid(
        x_axis={"U": U_values},
        varying_parameter={"V": V_values},
        fixed_parameter={"t": 1.0},
        solver_method="sparse_ED",
        chi=32,
        states_retained=6,
        include_idmrg=False,
        include_finite_dmrg=True,
    )

    manifest = run_convergence_study(
        beta,
        max_supercluster_size=max_supercluster_size,
        cluster_sizes=[2,4,8],
        sweep=sweep,
        base_L=24,
        max_beta_denominator=max_supercluster_size,
        output_root="large_files/runs/convergence_study",
    )

    print("Convergence study complete.")
    print(f"Golden ratio beta_raw={golden_ratio:.12f}, beta_used={beta:.12f}")
    print(f"Artifacts stored under: {manifest['output_root']}")
    print("Manifest entries:")
    for entry in manifest["entries"]:
        beta_ratio = entry["beta_ratio"]
        status = entry.get("status", "completed")
        if status == "completed":
            artifacts = entry.get("artifacts", {})
            print(
                f"  Nc={entry['Nc']} | beta={beta_ratio[0]}/{beta_ratio[1]} "
                f"| artifacts -> {artifacts.get('task_dir', 'n/a')}"
            )
        else:
            reason = entry.get("reason", "unspecified")
            print(
                f"  Nc={entry['Nc']} | beta={beta_ratio[0]}/{beta_ratio[1]} "
                f"| status={status.upper()} | reason: {reason}"
            )


if __name__ == "__main__":
    
    U_values=[0,1,5,50]
    V_values=[1e-6,5e-1,2,10]

    #U_values=[1,10]
    #V_values=[1,5]
    # run_convergence_example(U_values,V_values)



    # Legacy manual workflow reference:
    L = 24
    Nc = 8
    fixed_t = 1
    states_retained = 6
    U_values = [0,1]#[0,1,2,3,5,10,15,20,30,100]#np.unique(np.concatenate([np.linspace(0, 1, 3), np.linspace(2, 8, 3)]))
    V_values = [1e-6,1e-1,2e-1,5e-1,1,2,3,4]#[1e-6,1,10]#[1e-6, 1.0, 2.0,10]
    t_values = [0, 0.5, 1.0]
    v_sep_ratio = (1, 8)
    solver_method = "sparse_ED"
    
    int_sep_list = [(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8)]
    
    # allowed_m = enumerate_m(L, Nc, [int(L * v_sep_ratio[0] / v_sep_ratio[1])], 8)
    # print(f"allowed_m: {allowed_m}")
    # for m in allowed_m:
    #     clusters = generate_clusters(L, Nc, (m, L), v_sep_ratio)
    #     print(
    #         f\"PARAMETERS: L={L}, m={m}, "
    #         f\"V step n={int(L * v_sep_ratio[0] / v_sep_ratio[1])}, "
    #         f\"cluster_shapes={clusters.shape}\"
    #     )
    
    # int_seps = [(m, L) for m in allowed_m]
    
    # chi=64
    # compare_int_seps_with_dmrg(
    #     v_sep_ratio=(1,3),
    #     int_sep_list=[(1,3),(1,6)],
    #     x_axis={"U": [0,1e-1,5e-1,1,5,10,]},
    #     varying_parameter={"V": [1e-6,5e-1,1]},
    #     fixed_parameter={"t": fixed_t},
    #     L=L,
    #     Nc=3,
    #     chi=32,
    #     solver_method=solver_method,
    #     states_retained=states_retained,
    #     include_idmrg=True,
    #     include_finite_dmrg=True,
    # )

    #Non sensible large_files/plots/cluster_size_convergence_L36_chi32_20251114_194522.pkl

#     fig, results = compare_cluster_sizes_with_dmrg(
#     v_sep_ratio=(1, 2),
#     int_sep_ratios={4:(1,4),8:(1,8)},
#     cluster_sizes=[4,8],
#     U_values=[0,1e-1, 5e-1, 1.0, 5,10],
#     V_values=[1e-6],
#     L=32,
#     t=1.0,
#     solver_method='sparse_ED',
#     states_retained=6,
#     chi=32,
#     log_yaxis=False,
#     results='large_files/plots/cluster_size_convergence_L24_chi32_20251114_191504.pkl',
#     save_html=True,
#     show_plots=True,
#     plot_relative_error=False
# )
#     exit()



       
    #up to size 10 plot: large_files/plots/U_value_energy_comparison_L40_chi64_20251208_155050.pkl, took 40m
    
    # Example usage of the new function:

    data_path='large_files/plots/filling_int_cluster_comparison_L24_chi32_20251228_220850.pkl'
    #'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/partials/filling_L48_20260126_001228/merged_20260126_001228.pkl'
    #'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/plots/U_value_energy_comparison_L24_chi32_20251219_233103.pkl'
    #'/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/cluster_runs/merged_results/sep_1-4_merge.pkl'

    with open(data_path, 'rb') as f:
        results_data = pickle.load(f)
        # print("Loaded results data keys:", results_data.keys())
    U_values=results_data['U_values']
    # V_values=results_data['V_values']
    int_sep_ratios=results_data['int_sep_ratios_by_Nc']
    cluster_sizes=results_data['cluster_sizes']
    print(f"params: {results_data['parameters']}")
    print(f"U_values: {U_values}")
    
    

    # fig, results = compare_U_values_with_dmrg(
    #     v_sep_ratio=(1, 2),
    #     int_sep_ratios={2:(1,2),4:(1,4),6:(1,6)},
    #     cluster_sizes=[2,4,6],
    #     U_values=[0, 1, 2,3,4,5,7, 10,20,30],
    #     V_values=[1e-6, 1,2,3,5],
    #     L=24,
    #     t=-1.0,
    #     solver_method='sparse_ED',
    #     states_retained=6,
    #     chi=32,
    #     log_yaxis=False,  # Plot energies on linear scale
    #     include_idmrg=False,
    #     include_finite_dmrg=True,
    #     save_data=True,
    #     save_html=False,
    #     show_plots=True,
    #     include_timing=False,
    #     include_timing_plot=False,
    #     plot_relative_error=False,
    #     #results=data_path,  # Load existing results
    #     set_filling=1/2,
    #     dmrg_fixed_filling=1/2
    # )
    # exit()s

    # Example usage of compare_filling_with_int_cluster_sizes:
    # Compares half-filling (top row) vs quarter-filling (bottom row)
    # across different cluster sizes (columns) and interaction separations (lines)
    # Note: Each cluster size has its own compatible int_sep_ratios
    #
    fig, results = compare_filling_with_int_cluster_sizes(
        int_sep_ratios_by_Nc={
            2: [(1, 2), (1, 4), (1, 8),(1,12),(1,24)],   # π, π/2, π/4 for N_c=2
            3: [(1, 3), (1, 6),(1,12),(1,24)],            # π/3, 2π/3 for N_c=3
            4: [(1, 4), (1, 8),(1,12)],            # π/2, π/4 for N_c=4
            # 5: [(1,5),(1,10),(1,15)],
            6: [(1, 6), (1, 12),(1,24)],
            8: [(1, 8),(1,16),(1,24)],
        },
        U_values=[0,1,2,3,4,5,7,10,20,30],#[0, 1, 2,3,4,5,7, 10,20,30],          # x-axis
        V=0.0,                              # Fixed V (uses v_sep=(1,1) automatically when V≈0)
        L=48,
        t=-1.0,
        solver_method='sparse_ED',
        states_retained=6,
        chi=32,
        include_idmrg=False,
        include_finite_dmrg=True,
        save_data=True,
        save_html=False,
        show_plots=True,
        cols_per_page=6,
        plot_relative_error=True,
        results='large_files/plots/filling_int_cluster_comparison_L48_chi32_20260206_113516.pkl',
        # 'large_files/plots/filling_int_cluster_comparison_L48_chi32_20260206_113516.pkl' # nice data that demonstrates why you need 1/16 in the N_c=8 case! Will re-run at larger L!
        # 'large_files/plots/filling_int_cluster_comparison_L120_chi32_20260205_184801.pkl', #really nice results at 120 but larger cluster sizes missing the smaller one.
        # 'large_files/plots/filling_int_cluster_comparison_L120_chi32_20260205_184801.pkl',
        # 'large_files/plots/filling_int_cluster_comparison_L24_chi32_20251228_220850.pkl'
    )
    exit()

    # Example usage of compare_compressibility_with_int_cluster_sizes:
    # Compressibility plot: filling (n) vs mu_0 at various U values
    # - Rows: different U values
    # - Columns: different cluster sizes (N_c)
    # - Lines: different interaction separations
    # - Generates both filling and relative error matplotlib figures (PDF + SVG)
    # fig, results = compare_compressibility_with_int_cluster_sizes(
    #     int_sep_ratios_by_Nc={
    #         2: [(1, 2), (1, 4), (1, 8), (1, 12), (1, 24)],
    #         3: [(1, 3), (1, 6), (1, 12), (1, 24)],
    #         4: [(1, 4), (1, 8), (1, 12)],
    #         # 6: [(1, 6), (1, 12), (1, 24)],
    #         # 8: [(1, 8), (1, 16), (1, 24)],
    #     },
    #     U_values=[0.1, 1,2, 5, 10],
    #     n_mu_points=10,
    #     mu_range_factor=1.0,
    #     mu_min_range=1.0,
    #     V=0.0,
    #     L=48,
    #     t=-1.0,
    #     solver_method='sparse_ED',
    #     states_retained=6,
    #     chi=32,
    #     include_idmrg=False,
    #     include_finite_dmrg=True,
    #     save_data=True,
    #     save_html=False,
    #     show_plots=True,
    #     save_pdf=True,
    #     results=None,
    # )
    # exit()
    #results
    # N_c=4,6,8 comparison for L=72
    # large_files/plots/filling_int_cluster_comparison_L72_chi32_20251231_111030.pkl
    # N_c=4,8 L=80
    # Saved data to large_files/plots/filling_int_cluster_comparison_L80_chi32_20251231_115713.pkl
    # L=120, N_c=2,3,4,6
    # Saved data to large_files/plots/filling_int_cluster_comparison_L120_chi32_20260107_125158.pkl




    # Example usage of compare_filling_cluster_sizes:
    # Compares half-filling (top row) vs quarter-filling (bottom row)
    # - Columns: different V values
    # - X-axis: U values
    # - Lines: different cluster sizes (N_c)
    #
    # from aah_code.cluster_model.plots import compare_filling_cluster_sizes
    # fig, results = compare_filling_cluster_sizes(
    #     v_sep_ratio=(1, 2),                    # π spacing for V
    #     int_sep_ratios={                       # int_sep ratio for each cluster size
    #         2: (1, 2),   # π for N_c=3
    #         4: (1,4),   # 
    #         # 4: (1, 4),   # π/2 for N_c=4
    #         # 6: (1, 6),   # π/3 for N_c=6
    #         # 8: (1, 8),   # π/4 for N_c=8
    #         # 9: (1, 9),   # π/4 for N_c=8
    #     },
    #     cluster_sizes=[2,4],         # lines (different N_c)
    #     U_values=[1,10],#[0, 1, 2,3,4,5,7, 10,20,30],          # x-axis
    #     V_values=[1e-6],              # columns (different V)
    #     L=48,
    #     t=1.0,
    #     solver_method='sparse_ED',
    #     states_retained=6,
    #     chi=64,
    #     include_idmrg=True,
    #     include_finite_dmrg=True,
    #     save_data=True,
    #     save_html=False,
    #     results=None,
    #     # large_files/plots/filling_int_cluster_comparison_L120_chi32_20260205_184801.pkl
    #     # 'large_files/plots/filling_cluster_size_comparison_L48_chi32_20260205_105334.pkl',
    #     # 'large_files/plots/filling_cluster_size_comparison_L48_chi32_20260204_213153.pkl',
    #     # '/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/partials/filling_L48_20260126_001228/merged_fixed.pkl',
    #     # '/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/partials/filling_L48_vsep_2pi3_20260126_112422/merged_20260126_112422.pkl',
    #     # '/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/partials/filling_L48_20260126_001228/merged_fixed.pkl',
    #     #'large_files/plots/filling_cluster_size_comparison_L24_chi32_20260106_171531.pkl',
    #     show_plots=True,
    #     plot_relative_error=True,
    #     log_yaxis=False,
    #     axes=('U','V'),
    #     cols_per_page=6,
    #     shared_yaxis='page',
    #     compute_localization=True
        
    # )

    # exit()


    #beta=1/2
    # really nice results showing V convergence for \beta=\pi both at half and quarter filling!
    # '/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah/aah_code/cluster_model/large_files/partials/filling_L48_20260126_001228/merged_fixed.pkl',
    # large_files/plots/filling_cluster_size_comparison_L48_chi32_20260125_230523.pkl



    # N_c=3
    # large_files/plots/filling_cluster_size_comparison_L36_chi32_20260106_192546.pkl

    # Example usage of compare_fixed_supercluster:
    # Compares different (Nc, int_sep) pairs that give the same supercluster size.
    # - Rows: Different V values
    # - Columns: Different supercluster sizes
    # - X-axis: U values
    # - Lines: Different (Nc, int_sep) pairs for that supercluster size
    #          (maximal separation Nc=SC, int_sep=(1,SC) is shown in black)
    #
    
    #
    

    # from aah_code.cluster_model.plots import plot_supercluster_filling_comparison

    # fig, info = plot_supercluster_filling_comparison(
    #     'large_files/plots/fixed_supercluster_comparison_L120_chi32_20260109_164745.pkl', 
    #     'large_files/plots/fixed_supercluster_comparison_L120_chi32_20260113_174059.pkl',
    #     U_fixed=10.0,
    #     plot_relative_error=True,
    #     log_yaxis=False,
    #     shared_yaxis=True
    # )

    # fig.show()
    
    from aah_code.cluster_model.plots import compare_fixed_supercluster, compatible_int_seps, compute_supercluster_size
    # # Option 1: Auto-generate (Nc, int_sep) pairs using supercluster_sizes and max_seps
    fig, results = compare_fixed_supercluster(
        v_sep_ratio=(1, 2),
        U_values=[0, 1, 2,3,4,5,7, 10,20,30],
        V_values=[1e-6, 1,2,5],
        supercluster_sizes=[2,4,6,8],          # Auto-generate pairs for these SC sizes
        max_seps=4,                            # Max 3 (Nc, int_sep) pairs per SC size
        L=120,
        t=1.0,
        set_filling=1.0,
        solver_method='sparse_ED',
        chi=32,
        save_data=True,
        save_html=False,
        include_finite_dmrg=True,
        include_idmrg=False,
        rows_per_page=1,                       # 3 V values per page
        cols_per_page=4,                       # 3 supercluster sizes per page
        plot_relative_error=True,
        log_yaxis=False,
        axes=['V','U'],
        results='large_files/plots/fixed_supercluster_comparison_L120_chi32_20260109_164745.pkl'
    )
    exit()



    # 'large_files/plots/fixed_supercluster_comparison_L120_chi32_20260109_164745.pkl', - beta = pi, 1/2 filling
    #'large_files/plots/fixed_supercluster_comparison_L120_chi32_20260113_174059.pkl', - beta = pi, 1/4 filling
    #
    # # Option 2: Manually specify (Nc, int_sep) pairs
    # fig, results = compare_fixed_supercluster(
    #     v_sep_ratio=(1, 2),                    # π spacing for V (n=L/2=12 for L=24)
    #     U_values=[0, 2, 4, 8],
    #     V_values=[0.5, 1.0, 2.0],
    #     supercluster_int_seps={
    #         # supercluster_size -> list of (Nc, int_sep_ratio) pairs
    #         # Each (Nc, int_sep) must:
    #         #   1. Produce the claimed supercluster size
    #         #   2. Have Nc compatible with int_sep (qm % Nc == 0)
    #         # Format: (Nc, (p, q)) where int_sep_ratio = (p, q)
    #         #
    #         # SC=4: int_sep=(1,4) -> m=6, gcd(24,6,12)=6, SC=4
    #         #       qm = L/gcd(L,m) = 24/6 = 4
    #         #       Compatible Nc: 1, 2, 4 (must divide qm=4)
    #         4: [
    #             (4, (1, 4)),  # Nc=4, maximal separation (black)
    #             (2, (1, 4)),  # Nc=2, same int_sep
    #         ],
    #         # SC=6: int_sep=(1,3) -> m=8, gcd(24,8,12)=4, SC=6
    #         #       qm = 24/4 = 6, compatible Nc: 1, 2, 3, 6
    #         #       int_sep=(1,6) -> m=4, gcd(24,4,12)=4, SC=6
    #         #       qm = 24/4 = 6, compatible Nc: 1, 2, 3, 6
    #         6: [
    #             (6, (1, 6)),  # Nc=6, maximal separation (black)
    #             (3, (1, 3)),  # Nc=3, int_sep=(1,3)
    #             (2, (1, 6)),  # Nc=2, int_sep=(1,6)
    #         ],
    #         # SC=8: int_sep=(1,8) -> m=3, gcd(24,3,12)=3, SC=8
    #         #       qm = 24/3 = 8, compatible Nc: 1, 2, 4, 8
    #         8: [
    #             (8, (1, 8)),  # Nc=8, maximal separation (black)
    #             (4, (1, 8)),  # Nc=4
    #             (2, (1, 8)),  # Nc=2
    #         ],
    #     },
    #     L=24,
    #     t=1.0,
    #     set_filling=1.0,
    #     solver_method='sparse_ED',
    #     chi=32,
    #     include_finite_dmrg=True,
    #     plot_relative_error=False,
    #     rows_per_page=3,                       # 3 V values per page
    #     cols_per_page=3,                       # 3 supercluster sizes per page
    # )

    # Example usage of compare_compressibility_cluster_sizes:
    # Compressibility (filling vs mu_0) for non-zero V, lines = different Nc
    fig, results = compare_compressibility_cluster_sizes(
        v_sep_ratio=(1, 2),                    # π spacing for V
        int_sep_ratios={                       # int_sep ratio for each cluster size
            2: (1, 2),   # π for N_c=2
            4: (1, 4),   # π/2 for N_c=4
            6: (1,6),
            8: (1,8)
        },
        cluster_sizes=[2, 4],                  # lines (different N_c)
        U_values=[1,2,5, 10],                      # rows
        V_values=[1e-6, 1.0,3,5],                   # columns
        n_mu_points=10,
        mu_range_factor=1.0,
        mu_min_range=1.0,
        t=1.0,
        L=48,
        chi=32,
        solver_method='sparse_ED',
        states_retained=6,
        include_idmrg=False,
        include_finite_dmrg=True,
        save_data=True,
        save_html=False,
        show_plots=True,
        save_pdf=True,
        results=None,
        axes=('U', 'V'),                       # rows=U, cols=V
    )
    exit()

    # Example usage of compare_compressibility_fixed_supercluster:
    # Compressibility (filling vs mu_0) for fixed supercluster sizes
    # fig_list, results = compare_compressibility_fixed_supercluster(
    #     v_sep_ratio=(1, 2),
    #     U_values=[0.1, 1],
    #     V_values=[0.5],
    #     supercluster_sizes=[4, 6],
    #     n_mu_points=10,
    #     mu_range_factor=1.0,
    #     mu_min_range=1.0,
    #     t=1.0,
    #     L=48,
    #     chi=32,
    #     solver_method='dense_ED',
    #     show_plots=False,
    #     save_pdf=True,
    #     axes=('U', 'V'),  # rows=U, pages=V
    # )
