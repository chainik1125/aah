"""
Plotting functions for comparing different cluster model setups with iDMRG.
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


def compare_int_seps_with_dmrg(
    v_sep_ratio: Tuple[int, int],
    int_sep_list: List[Tuple[int, int]],
    U_values: np.ndarray,
    V_values: np.ndarray,
    t: float = 1.0,
    L: int = 20,
    Nc: int = 2,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True
):
    """
    Compare different int_sep setups with iDMRG for a fixed v_sep.
    
    Args:
        v_sep_ratio: Fixed V separation ratio (e.g., (1,2) for staggered)
        int_sep_list: List of int_sep ratios to compare (e.g., [(1,2), (1,3), (1,4)])
        U_values: Array of U values to sweep
        V_values: Array of V values to test
        t: Hopping parameter
        L: System size
        Nc: Cluster size
        chi: DMRG bond dimension
        solver_method: Method for solving ('dense_ED' or 'sparse_ED')
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    
    Returns:
        figures: List of plotly figures
        all_results: Dictionary with all computed results
    """
    
    print("=" * 60)
    print(f"Comparing {len(int_sep_list)} int_sep configurations with iDMRG")
    print("=" * 60)
    print(f"System: L={L}, Nc={Nc}, t={t}")
    print(f"Fixed v_sep={v_sep_ratio}")
    print(f"Int_sep configurations: {int_sep_list}")
    print(f"DMRG: chi={chi}")
    print(f"U values: {U_values}")
    print(f"V values: {V_values}")
    
    # Storage for all results
    all_results = {}
    
    for V in tqdm(V_values, desc="V values", position=0, leave=True, ncols=80):
        all_results[V] = {
            'energies_dmrg': [],
            'fillings_dmrg': []
        }
        
        # Initialize storage for each int_sep configuration
        for int_sep in int_sep_list:
            int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
            all_results[V][f'energies_{int_sep_key}'] = []
            all_results[V][f'fillings_{int_sep_key}'] = []
        
        for U in tqdm(U_values, desc=f"  U (V={V:.2f})", position=1, leave=False, ncols=80):
            mu_0 = U / 2  # Half-filling
            
            # DMRG calculation (same for all int_sep, only depends on v_sep)
            energy_dmrg, filling_dmrg, _ = run_dmrg_method(U, mu_0, V, v_sep_ratio, t, L, chi)
            energy_dmrg_subtracted = energy_dmrg + mu_0 * filling_dmrg
            
            all_results[V]['energies_dmrg'].append(energy_dmrg_subtracted)
            all_results[V]['fillings_dmrg'].append(filling_dmrg)
            
            # Calculate for each int_sep configuration
            for int_sep in int_sep_list:
                int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                
                physical_params = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
                run_config = ClusterModelConfig(
                    L=L,
                    int_cluster_size=Nc,
                    cluster_separation_ratio=int_sep,
                    V_separation_ratio=v_sep_ratio,
                    ham_lib='quspin',
                    physical_params=physical_params,
                    model_bc='periodic',
                    int_cluster_bc='periodic',
                    super_cluster_bc='periodic',
                    solver_method=solver_method,
                    states_retained=states_retained
                )
                
                system_expectations, _ = get_general_expectations(run_config)
                energy, filling, _ = system_expectations
                energy_subtracted = (energy + mu_0 * filling) / L
                filling_per_site = filling / L
                
                all_results[V][f'energies_{int_sep_key}'].append(energy_subtracted)
                all_results[V][f'fillings_{int_sep_key}'].append(filling_per_site)
    
    print("\n")  # Add spacing after progress bars
    
    # Create plots
    figures = create_int_sep_comparison_plots(
        U_values, V_values, all_results, int_sep_list, v_sep_ratio, 
        output_dir, show_plots
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics (Mean Absolute Differences from DMRG)")
    print("=" * 60)
    
    for V in V_values:
        print(f"\nV={V:.2f}:")
        energies_dmrg = np.array(all_results[V]['energies_dmrg'])
        
        for int_sep in int_sep_list:
            int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
            energies_method = np.array(all_results[V][f'energies_{int_sep_key}'])
            mae = np.nanmean(np.abs(energies_method - energies_dmrg))
            print(f"  int_sep={int_sep}: MAE={mae:.6f}")
    
    return figures, all_results


def create_int_sep_comparison_plots(
    U_values, V_values, all_results, int_sep_list, v_sep_ratio,
    output_dir='large_files/plots', show_plots=True
):
    """Create line plots comparing different int_sep configurations with DMRG."""
    
    # Group V values into chunks of 3
    n_v_per_fig = 3
    n_figures = np.ceil(len(V_values) / n_v_per_fig).astype(int)
    figures = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for different methods (reserve red for DMRG)
    color_palette = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    colors = {'DMRG': 'red'}
    
    # Helper function to format separation as fraction of π
    def format_sep_as_pi(sep_tuple):
        """Convert separation ratio to π fraction notation."""
        numerator = 2 * sep_tuple[0]
        denominator = sep_tuple[1]
        if numerator == denominator:
            return 'π'
        elif numerator == 1:
            return f'π/{denominator}'
        else:
            from fractions import Fraction
            frac = Fraction(numerator, denominator)
            if frac.denominator == 1:
                return f'{frac.numerator}π'
            return f'{frac.numerator}π/{frac.denominator}'
    
    for idx, int_sep in enumerate(int_sep_list):
        int_sep_key = f'int_sep {format_sep_as_pi(int_sep)}'
        colors[int_sep_key] = color_palette[idx % len(color_palette)]
    
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
            
            # Plot DMRG results
            # Energy plot (top row)
            fig.add_trace(
                go.Scatter(
                    x=U_values,
                    y=all_results[V]['energies_dmrg'],
                    mode='lines+markers',
                    name='DMRG',
                    line=dict(color=colors['DMRG'], width=3),
                    marker=dict(size=8, symbol='diamond'),
                    showlegend=(col_idx == 0)
                ),
                row=1, col=col
            )
            
            # Filling plot (bottom row)
            fig.add_trace(
                go.Scatter(
                    x=U_values,
                    y=all_results[V]['fillings_dmrg'],
                    mode='lines+markers',
                    name='DMRG',
                    line=dict(color=colors['DMRG'], width=3),
                    marker=dict(size=8, symbol='diamond'),
                    showlegend=False
                ),
                row=2, col=col
            )
            
            # Plot each int_sep configuration
            for idx, int_sep in enumerate(int_sep_list):
                int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                label = f'int_sep {format_sep_as_pi(int_sep)}'
                
                # Check if int_sep matches v_sep for marker selection
                if int_sep == v_sep_ratio:
                    marker_symbol = 'diamond'  # Same as DMRG
                else:
                    marker_symbol = ['circle', 'square', 'triangle-up', 'x', 'cross'][idx % 5]
                
                # Energy plot (top row)
                fig.add_trace(
                    go.Scatter(
                        x=U_values,
                        y=all_results[V][f'energies_{int_sep_key}'],
                        mode='lines+markers',
                        name=label,
                        line=dict(
                            color=colors[label],
                            width=2
                            # Removed dash parameter - all lines are solid now
                        ),
                        marker=dict(size=6, symbol=marker_symbol),
                        showlegend=(col_idx == 0)
                    ),
                    row=1, col=col
                )
                
                # Filling plot (bottom row)
                fig.add_trace(
                    go.Scatter(
                        x=U_values,
                        y=all_results[V][f'fillings_{int_sep_key}'],
                        mode='lines+markers',
                        name=label,
                        line=dict(
                            color=colors[label],
                            width=2
                            # Removed dash parameter - all lines are solid now
                        ),
                        marker=dict(size=6, symbol=marker_symbol),
                        showlegend=False
                    ),
                    row=2, col=col
                )
            
            # Update axes labels
            fig.update_xaxes(title_text='U', row=1, col=col)
            fig.update_xaxes(title_text='U', row=2, col=col)
            
            # Set y-axis range for filling plots
            fig.update_yaxes(range=[0, 2], row=2, col=col)
            
            if col == 1:
                fig.update_yaxes(title_text='Energy/site', row=1, col=col)
                fig.update_yaxes(title_text='Filling/site', row=2, col=col)
        
        # Update layout
        title_text = (f'Int_sep Comparison with DMRG (v_sep={format_sep_as_pi(v_sep_ratio)}, '
                     f'Page {fig_idx+1}/{n_figures})')
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center'),
            height=700,
            width=400 * n_cols + 150,  # Extra width for right-side legend
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            ),
            hovermode='x unified'
        )
        
        # Save and/or show
        v_sep_str = f"{v_sep_ratio[0]}_{v_sep_ratio[1]}"
        filename = f'int_sep_comparison_v_sep_{v_sep_str}_page_{fig_idx+1}.html'
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved figure to {filepath}")
        
        if show_plots:
            fig.show()
        
        figures.append(fig)
    
    return figures


if __name__ == "__main__":
    # Example usage
    L = 12
    Nc = 3
    U_values = np.linspace(0, 6, 7)
    V_values = np.array([1e-8, 0.5, 1.0, 2.0])
    v_sep_ratio = (1, 6)
    int_sep_list = [(1, 6), (1, 4), (1, 3)]
    
    figures, results = compare_int_seps_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_list=int_sep_list,
        U_values=U_values,
        V_values=V_values,
        L=L,
        Nc=Nc,
        solver_method='sparse_ED',
        chi=32,
        show_plots=True
    )