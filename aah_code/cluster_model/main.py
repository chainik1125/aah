"""
File for running mainline scripts
Eventually this will just be simple call to the ClusterModel class
but for now I just want to get a single spectrum run going.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        return iterable
import os

from compare_old_new_line_plots import save_line_plots, compare_old_new_line_plots, get_old_method_results
from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.real_space_dmrg import run_dmrg_method
from aah_code.cluster_model.plots import compare_int_seps_with_dmrg

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


if __name__ == "__main__":
    L=84
    Nc=3
    #t=0.0
    states_retained=6
    U_values=np.linspace(0,3,3)
    V_values=[1e-6,1,2]
    t_values=[0,1/2,1]
    v_sep_ratio=(1,6)
    solver_method='sparse_ED'

    
    #old_new_quspin_comparison()
    # three_way_comparison_with_dmrg(
    #     L=L,
    #     Nc=Nc,
    #     int_sep_ratio=int_sep_ratio,
    #     v_sep_ratio=v_sep_ratio,
    #     U_values=U_values,
    #     V_values=V_values,
    #     solver_method=solver_method
    # )
    
    # Compare different int_sep configurations with DMRG for fixed v_sep

    int_sep_list=[(1,3),(1,6)]

    compare_int_seps_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_list=int_sep_list,
        x_axis={'t':t_values},
        varying_parameter={'V':V_values},
        fixed_parameter={'U':0.0},
        L=L,
        Nc=Nc,
        solver_method=solver_method,
        states_retained=states_retained,
        include_idmrg=True,          # Include infinite DMRG (default)
        include_finite_dmrg=True    # Optionally include finite DMRG
    )



    # compare_int_seps_with_dmrg(
    #     v_sep_ratio=v_sep_ratio,
    #     int_sep_list=int_sep_list
    #     U_values=U_values,
    #     V_values=V_values,
    #     L=L,
    #     Nc=Nc,
    #     t=t,
    #     solver_method=solver_method,
    #     states_retained=states_retained
    # )