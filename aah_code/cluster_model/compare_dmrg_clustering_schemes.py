"""
Compare DMRG with multiple clustering schemes for the generalized cluster model.
DMRG provides the reference solution, while different int_sep ratios represent
different clustering approximations.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.real_space_dmrg import run_dmrg_method


def compare_dmrg_with_clustering_schemes(
    U_values: np.ndarray,
    V_values: np.ndarray,
    v_sep_ratio: Tuple[int, int],
    int_sep_list: List[Tuple[int, int]],
    t: float = 1.0,
    L: int = 20,
    Nc: int = 2,
    chi: int = 32,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True
):
    """
    Compare DMRG with multiple clustering schemes.
    
    DMRG is computed once for each (U, V) pair with the given v_sep_ratio.
    Multiple clustering schemes are compared, each with different int_sep_ratio
    but the same v_sep_ratio.
    
    Parameters
    ----------
    U_values : np.ndarray
        Array of Hubbard U values
    V_values : np.ndarray
        Array of V interaction values
    v_sep_ratio : tuple
        V-term separation ratio (p, q) used for both DMRG and clustering
    int_sep_list : list of tuples
        List of interaction separation ratios to compare
    t : float
        Hopping parameter
    L : int
        System size
    Nc : int
        Cluster size
    chi : int
        Bond dimension for DMRG
    output_dir : str
        Directory to save plots
    show_plots : bool
        Whether to display plots in browser
        
    Returns
    -------
    figures : list
        List of plotly figures
    all_results : dict
        Dictionary containing all computed results
    """
    
    print("=" * 60)
    print("Compare DMRG with Multiple Clustering Schemes")
    print("=" * 60)
    print(f"System: L={L}, Nc={Nc}, t={t}")
    print(f"V-separation: v_sep={v_sep_ratio} (used by both DMRG and clustering)")
    print(f"Clustering schemes to compare: {int_sep_list}")
    print(f"DMRG: chi={chi}")
    print(f"U values: {U_values}")
    print(f"V values: {V_values}")
    
    # Storage for all results
    all_results = {}
    
    # Main loop over V values with progress bar
    for V in tqdm(V_values, desc="V values", position=0):
        all_results[V] = {
            'energies_dmrg': [],
            'fillings_dmrg': [],
            'energies_clustering': {str(int_sep): [] for int_sep in int_sep_list},
            'fillings_clustering': {str(int_sep): [] for int_sep in int_sep_list}
        }
        
        # Inner loop over U values with progress bar
        for U in tqdm(U_values, desc=f"U values (V={V:.3f})", position=1, leave=False):
            mu_0 = U / 2  # Half-filling
            
            # Run DMRG once for this (U, V) pair
            tqdm.write(f"  Running DMRG for U={U:.2f}, V={V:.3f}")
            energy_dmrg, filling_dmrg, _ = run_dmrg_method(
                U=U, 
                mu_0=mu_0, 
                V=V, 
                V_sep=v_sep_ratio,  # Use v_sep_ratio for DMRG
                t=t, 
                system_size=L, 
                chi=chi
            )
            # DMRG returns per-site quantities already
            energy_dmrg_subtracted = energy_dmrg + mu_0 * filling_dmrg
            
            all_results[V]['energies_dmrg'].append(energy_dmrg_subtracted)
            all_results[V]['fillings_dmrg'].append(filling_dmrg)
            
            # Run each clustering scheme
            for int_sep_idx, int_sep_ratio in enumerate(int_sep_list):
                # Add description for current clustering scheme
                tqdm.write(f"  Running clustering scheme {int_sep_idx+1}/{len(int_sep_list)}: int_sep={int_sep_ratio}")
                
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
                
                system_expectations, _ = get_general_expectations(run_config)
                energy, filling, _ = system_expectations
                energy_subtracted = (energy + mu_0 * filling) / L
                filling_per_site = filling / L
                
                all_results[V]['energies_clustering'][str(int_sep_ratio)].append(energy_subtracted)
                all_results[V]['fillings_clustering'][str(int_sep_ratio)].append(filling_per_site)
    
    # Create plots
    figures = create_comparison_plots(
        U_values, V_values, all_results, int_sep_list, 
        v_sep_ratio, output_dir, show_plots
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for V in V_values:
        print(f"\nV={V:.2f}:")
        energies_dmrg = np.array(all_results[V]['energies_dmrg'])
        
        for int_sep in int_sep_list:
            energies_cluster = np.array(all_results[V]['energies_clustering'][str(int_sep)])
            error = np.nanmean(np.abs(energies_cluster - energies_dmrg))
            print(f"  int_sep={int_sep}: |Cluster-DMRG|={error:.6f}")
    
    return figures, all_results


def create_comparison_plots(
    U_values, V_values, all_results, int_sep_list, v_sep_ratio,
    output_dir='large_files/plots', show_plots=True
):
    """Create line plots comparing DMRG with multiple clustering schemes."""
    
    # Group V values into chunks of 3
    n_v_per_fig = 3
    n_figures = np.ceil(len(V_values) / n_v_per_fig).astype(int)
    figures = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
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
        
        for col_idx, V in enumerate(current_V_values):
            col = col_idx + 1
            
            # DMRG reference (black, solid line)
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['energies_dmrg'],
                    mode='lines+markers', 
                    name='DMRG',
                    line=dict(color='black', width=2),
                    marker=dict(size=6),
                    showlegend=(col_idx == 0)
                ),
                row=1, col=col
            )
            
            # Add each clustering scheme
            for i, int_sep in enumerate(int_sep_list):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=U_values, 
                        y=all_results[V]['energies_clustering'][str(int_sep)],
                        mode='lines+markers', 
                        name=f'Cluster {int_sep}',
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=6, symbol='square'),
                        showlegend=(col_idx == 0)
                    ),
                    row=1, col=col
                )
            
            # Filling plots (bottom row)
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['fillings_dmrg'],
                    mode='lines+markers', 
                    name='DMRG',
                    line=dict(color='black', width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=2, col=col
            )
            
            for i, int_sep in enumerate(int_sep_list):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=U_values, 
                        y=all_results[V]['fillings_clustering'][str(int_sep)],
                        mode='lines+markers', 
                        name=f'Cluster {int_sep}',
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=6, symbol='square'),
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
        title_text = f'DMRG vs Clustering Schemes (v_sep={v_sep_ratio}, Page {fig_idx+1}/{n_figures})'
        
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
        filename = f'dmrg_vs_clustering_v_sep_{v_sep_ratio[0]}_{v_sep_ratio[1]}_page_{fig_idx+1}.html'
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved figure to {filepath}")
        
        if show_plots:
            fig.show()
        
        figures.append(fig)
    
    return figures


if __name__ == "__main__":
    # Test with v_sep=(1,6) and int_sep_list=[(1,6)]
    U_values = np.array([1.0])  # Just 1 U value for very quick test
    V_values = np.array([1e-6])  # Single small V value for quick test
    v_sep_ratio = (1, 6)  # Period-6 modulation
    int_sep_list = [(1, 6), (1, 3), (1, 2)]  # Test multiple clustering schemes
    
    print("Running test comparison...")
    figures, results = compare_dmrg_with_clustering_schemes(
        U_values=U_values,
        V_values=V_values,
        v_sep_ratio=v_sep_ratio,
        int_sep_list=int_sep_list,
        t=1.0,
        L=12,  # Use L=12 for testing (divisible by 6)
        Nc=2,
        chi=16,  # Reduced chi for faster test
        output_dir='large_files/plots',
        show_plots=False  # Don't open browser during test
    )
    
    print("\nTest complete!")