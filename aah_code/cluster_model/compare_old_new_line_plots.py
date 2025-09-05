"""
Line plot comparison between Old QSpin (mismatched) and New General QSpin implementation.
Based on compare_methods_line_plots from hamiltonian.py
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from typing import Tuple, Optional, List
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        return iterable
import math
import os
from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations, test_quick_mismatched
from aah_code.hamiltonian import HamiltonianParams
from aah_code.main import run_cluster_method

# Set plotly to browser renderer (not notebook)
pio.renderers.default = 'browser'

# Silence verbose logging
import logging
logging.getLogger('aah').setLevel(logging.WARNING)


def get_old_method_results(
    U: float,
    V: float, 
    t: float,
    L: int,
    Nc: int,
    int_sep_ratio: Tuple[int, int],
    v_sep_ratio: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Select and run the appropriate old method based on the separation ratios.
    
    Returns
    -------
    energy_per_site, filling_per_site : tuple
        Energy and filling per site from the old method
    """
    mu_0 = U / 2  # Half-filling
    
    # Determine which old method to use based on ratios
    # test_quick_mismatched: int_sep = L/4, v_sep = L/2
    # run_cluster_method: cluster_k_generator = L/2 (π separation)
    
    if v_sep_ratio == (1, 2) and int_sep_ratio == (1, 4):
        # This matches test_quick_mismatched configuration
        print("  Using test_quick_mismatched (int_sep=L/4, v_sep=L/2)")
        physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
        system_expectations, _ = test_quick_mismatched(
            lattice_points=L,
            cluster_size=Nc,
            physical_params=physical_params,
            ham_lib='quspin'
        )
        energy, filling, _ = system_expectations
        return energy / L, filling / L
        
    elif v_sep_ratio == (1, 2) and int_sep_ratio == (1, 2):
        # This matches run_cluster_method configuration (π separation)
        print("  Using run_cluster_method (π separation)")
        energy, filling = run_cluster_method(
            U=U, mu_0=mu_0, V=V, t=t, 
            system_size=L, 
            ham_lib='quspin'
        )
        # run_cluster_method returns total values, need per-site
        return energy / L, filling / L
        
    else:
        # For other configurations, use test_quick_mismatched as default
        # but warn the user
        print(f"  Warning: No exact old method match for int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
        print("  Using test_quick_mismatched as fallback")
        physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
        system_expectations, _ = test_quick_mismatched(
            lattice_points=L,
            cluster_size=Nc,
            physical_params=physical_params,
            ham_lib='quspin'
        )
        energy, filling, _ = system_expectations
        return energy / L, filling / L


def compare_old_new_line_plots(
    U_values: np.ndarray,
    V_values: np.ndarray,
    t: float = 1.0,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    precomputed_results: Optional[dict] = None
) -> List[go.Figure]:
    """
    Create line plots comparing old QSpin vs new general QSpin for varying U at fixed V values.
    Each figure shows 2x3 subplots (energy top row, filling bottom row).
    If more than 3 V values, creates multiple figures.
    
    The old method is automatically selected based on the separation ratios:
    - int_sep=(1,4), v_sep=(1,2): Uses test_quick_mismatched (L/4, L/2 separations)
    - int_sep=(1,2), v_sep=(1,2): Uses run_cluster_method (π separation)
    - Other configurations: Uses test_quick_mismatched as fallback with warning
    
    Parameters
    ----------
    U_values : array-like
        Array of Hubbard U interaction values to test
    V_values : array-like  
        Array of V interaction values to test
    t : float
        Hopping parameter (default: 1.0)
    L : int
        System size (default: 8)
    Nc : int
        Cluster size (default: 2)
    int_sep_ratio : tuple
        Interaction cluster separation ratio for new method
    v_sep_ratio : tuple
        V-term separation ratio for new method
    precomputed_results : dict, optional
        Pre-computed results to avoid re-solving
        
    Returns
    -------
    figures : list of plotly.graph_objects.Figure
        List of interactive plotly figures with line plot comparisons
    """
    
    # Group V values into chunks of 3
    n_v_per_fig = 3
    n_figures = math.ceil(len(V_values) / n_v_per_fig)
    figures = []
    
    # Use pre-computed results if provided, otherwise compute
    if precomputed_results is not None:
        print("Using pre-computed results for line plots")
        all_results = precomputed_results
    else:
        print("=" * 60)
        print("Computing line plots for Old vs New QSpin")
        print("=" * 60)
        print(f"System: L={L}, Nc={Nc}, t={t}")
        print(f"New method clustering: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
        print(f"U values: {U_values}")
        print(f"V values: {V_values}")
        
        # Storage for all results
        all_results = {}
        
        for V in tqdm(V_values, desc="V values"):
            # Initialize storage for this V
            all_results[V] = {
                'energies_old': [],
                'energies_new': [],
                'fillings_old': [],
                'fillings_new': []
            }
            
            for U in U_values:
                mu_0 = U / 2  # Half-filling
                
                try:
                    # Old QSpin method (selected based on ratios)
                    energy_old_per_site, filling_old_per_site = get_old_method_results(
                        U=U, V=V, t=t, L=L, Nc=Nc,
                        int_sep_ratio=int_sep_ratio,
                        v_sep_ratio=v_sep_ratio
                    )
                    # Add mu_0 term back for comparison
                    energy_old_subtracted = energy_old_per_site + mu_0 * filling_old_per_site
                    filling_old_normalized = filling_old_per_site
                    
                    # New General QSpin method
                    physical_params_new = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=Nc,
                        cluster_separation_ratio=int_sep_ratio,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params_new,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic'
                    )
                    system_expectations_new, _ = get_general_expectations(run_config)
                    energy_new, filling_new, _ = system_expectations_new
                    # Add mu_0 term back for comparison
                    energy_new_subtracted = (energy_new + mu_0 * filling_new) / L
                    filling_new_normalized = filling_new / L
                    
                    # Store results
                    all_results[V]['energies_old'].append(energy_old_subtracted)
                    all_results[V]['energies_new'].append(energy_new_subtracted)
                    all_results[V]['fillings_old'].append(filling_old_normalized)
                    all_results[V]['fillings_new'].append(filling_new_normalized)
                    
                except Exception as e:
                    print(f"Error at U={U}, V={V}: {e}")
                    all_results[V]['energies_old'].append(np.nan)
                    all_results[V]['energies_new'].append(np.nan)
                    all_results[V]['fillings_old'].append(np.nan)
                    all_results[V]['fillings_new'].append(np.nan)
    
    # Create figures
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
            'New General': 'red'
        }
        
        for col_idx, V in enumerate(current_V_values):
            col = col_idx + 1
            
            # Energy plots (top row)
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['energies_old'],
                    mode='lines+markers', 
                    name='Old QSpin (Mismatched)',
                    line=dict(color=colors['Old QSpin'], width=2),
                    marker=dict(size=6),
                    showlegend=(col_idx == 0),
                    hovertemplate='U=%{x}<br>E/site=%{y:.4f}<extra></extra>'
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
                    showlegend=(col_idx == 0),
                    hovertemplate='U=%{x}<br>E/site=%{y:.4f}<extra></extra>'
                ),
                row=1, col=col
            )
            
            # Filling plots (bottom row)
            fig.add_trace(
                go.Scatter(
                    x=U_values, 
                    y=all_results[V]['fillings_old'],
                    mode='lines+markers', 
                    name='Old QSpin (Mismatched)',
                    line=dict(color=colors['Old QSpin'], width=2),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate='U=%{x}<br>n/site=%{y:.4f}<extra></extra>'
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
                    showlegend=False,
                    hovertemplate='U=%{x}<br>n/site=%{y:.4f}<extra></extra>'
                ),
                row=2, col=col
            )
            
            # Update axes labels
            fig.update_xaxes(title_text='U', row=1, col=col)
            fig.update_xaxes(title_text='U', row=2, col=col)
            
            if col == 1:  # Only add y-axis labels on leftmost column
                fig.update_yaxes(title_text='Energy/site', row=1, col=col)
                fig.update_yaxes(title_text='Filling/site', row=2, col=col)
        
        # Update layout
        title_text = f'Old QSpin vs New General: Line Plot Comparison (Page {fig_idx+1}/{n_figures})'
        subtitle_text = f'L={L}, Nc={Nc}, t={t}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio}'
        
        fig.update_layout(
            title=dict(
                text=f'{title_text}<br><sub>{subtitle_text}</sub>',
                x=0.5,
                xanchor='center'
            ),
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
        
        figures.append(fig)
    
    return figures, all_results


def save_line_plots(
    U_values: np.ndarray,
    V_values: np.ndarray,
    t: float = 1.0,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    output_prefix: str = 'old_vs_new_line_plots',
    output_dir: str = 'large_files/plots',
    save_html: bool = True,
    show_plots: bool = True
):
    """
    Generate and display/save line plot comparisons.
    
    Parameters
    ----------
    U_values, V_values, t, L, Nc, int_sep_ratio, v_sep_ratio : 
        Same as compare_old_new_line_plots
    output_prefix : str
        Prefix for output HTML files
    output_dir : str
        Directory to save HTML files (default: 'large_files/plots')
    save_html : bool
        Whether to save HTML files (default: True)
    show_plots : bool
        Whether to display plots in browser (default: True)
    """
    
    figures, all_results = compare_old_new_line_plots(
        U_values=U_values,
        V_values=V_values,
        t=t,
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio
    )
    
    # Create output directory if saving HTML
    if save_html:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save and/or show each figure
    for i, fig in enumerate(figures):
        if save_html:
            filename = f'{output_prefix}_page_{i+1}.html'
            filepath = os.path.join(output_dir, filename)
            fig.write_html(filepath)
            print(f"Saved figure to {filepath}")
        
        if show_plots:
            fig.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for V in V_values:
        energies_old = np.array(all_results[V]['energies_old'])
        energies_new = np.array(all_results[V]['energies_new'])
        fillings_old = np.array(all_results[V]['fillings_old'])
        fillings_new = np.array(all_results[V]['fillings_new'])
        
        energy_diff = np.nanmean(np.abs(energies_new - energies_old))
        filling_diff = np.nanmean(np.abs(fillings_new - fillings_old))
        
        print(f"V={V:.2f}: Mean |ΔE/site|={energy_diff:.6f}, Mean |Δn/site|={filling_diff:.6f}")
    
    return figures


if __name__ == "__main__":
    # Define parameter ranges
    U_values = np.linspace(0, 10, 11)  # 11 U values from 0 to 10
    V_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # 6 V values
    
    # Generate, show, and save line plots
    figures = save_line_plots(
        U_values=U_values,
        V_values=V_values,
        t=1.0,
        L=8,
        Nc=2,
        int_sep_ratio=(1, 4),
        v_sep_ratio=(1, 2),
        output_prefix='old_vs_new_line_plots',
        output_dir='large_files/plots',  # Save to large_files/plots
        save_html=True,  # Save HTML files
        show_plots=True  # Display in browser
    )
    
    print("\nComparison complete!")