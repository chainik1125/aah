"""
Comparison between Old QSpin (mismatched) and New General QSpin implementation
with plotly visualization.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional
from tqdm import tqdm
from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations, test_quick_mismatched
from aah_code.hamiltonian import HamiltonianParams

# Silence verbose logging
import logging
logging.getLogger('aah').setLevel(logging.WARNING)


def compare_old_new_quspin(
    U_values: np.ndarray,
    V_values: np.ndarray,
    t: float = 1.0,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    save_html: bool = True,
    output_file: str = 'old_vs_new_quspin_comparison.html'
) -> go.Figure:
    """
    Compare old QSpin (mismatched) vs new general QSpin and create plotly visualization.
    
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
    save_html : bool
        Whether to save the figure as HTML (default: True)
    output_file : str
        Output HTML filename
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure with comparison heatmaps
    """
    
    n_U = len(U_values)
    n_V = len(V_values)
    
    # Initialize storage grids
    energy_old_grid = np.zeros((n_U, n_V))
    energy_new_grid = np.zeros((n_U, n_V))
    filling_old_grid = np.zeros((n_U, n_V))
    filling_new_grid = np.zeros((n_U, n_V))
    
    print("=" * 60)
    print("Comparing Old QSpin vs New General QSpin")
    print("=" * 60)
    print(f"System: L={L}, Nc={Nc}, t={t}")
    print(f"New method clustering: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print(f"Testing {n_U} U values × {n_V} V values")
    print("=" * 60)
    
    # Run parameter scan
    for i, U in enumerate(tqdm(U_values, desc='U values')):
        for j, V in enumerate(V_values):
            mu_0 = U / 2  # Half-filling
            
            try:
                # Old QSpin method
                physical_params_old = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
                system_expectations_old, _ = test_quick_mismatched(
                    lattice_points=L,
                    cluster_size=Nc,
                    physical_params=physical_params_old,
                    ham_lib='quspin'
                )
                energy_old, filling_old, _ = system_expectations_old
                energy_old_per_site = energy_old / L
                filling_old_per_site = filling_old / L
                
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
                energy_new_per_site = energy_new / L
                filling_new_per_site = filling_new / L
                
                # Store results
                energy_old_grid[i, j] = energy_old_per_site
                energy_new_grid[i, j] = energy_new_per_site
                filling_old_grid[i, j] = filling_old_per_site
                filling_new_grid[i, j] = filling_new_per_site
                
            except Exception as e:
                print(f"Error at U={U}, V={V}: {e}")
                energy_old_grid[i, j] = np.nan
                energy_new_grid[i, j] = np.nan
                filling_old_grid[i, j] = np.nan
                filling_new_grid[i, j] = np.nan
    
    # Calculate differences
    energy_diff = energy_new_grid - energy_old_grid
    filling_diff = filling_new_grid - filling_old_grid
    
    # Calculate percentage differences
    energy_pct_diff = np.where(
        np.abs(energy_old_grid) > 1e-12,
        100 * energy_diff / np.abs(energy_old_grid),
        100 * energy_diff
    )
    filling_pct_diff = np.where(
        np.abs(filling_old_grid) > 1e-12,
        100 * filling_diff / np.abs(filling_old_grid),
        100 * filling_diff
    )
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Old QSpin: Energy/site', 'New General: Energy/site', 'Difference: Energy/site',
            'Old QSpin: Filling/site', 'New General: Filling/site', 'Difference: Filling/site',
            'Energy % Difference', 'Filling % Difference', 'Absolute Energy Difference'
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # Row 1: Energy comparisons
    fig.add_trace(
        go.Heatmap(
            z=energy_old_grid,
            x=V_values,
            y=U_values,
            colorscale='Viridis',
            text=np.round(energy_old_grid, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>E/site=%{z:.4f}<extra></extra>',
            colorbar=dict(title='E/site', x=0.28, y=0.83, len=0.25)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=energy_new_grid,
            x=V_values,
            y=U_values,
            colorscale='Viridis',
            text=np.round(energy_new_grid, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>E/site=%{z:.4f}<extra></extra>',
            showscale=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=energy_diff,
            x=V_values,
            y=U_values,
            colorscale='RdBu',
            zmid=0,
            text=np.round(energy_diff, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>ΔE/site=%{z:.4f}<extra></extra>',
            colorbar=dict(title='ΔE/site', x=1.02, y=0.83, len=0.25)
        ),
        row=1, col=3
    )
    
    # Row 2: Filling comparisons
    fig.add_trace(
        go.Heatmap(
            z=filling_old_grid,
            x=V_values,
            y=U_values,
            colorscale='Plasma',
            text=np.round(filling_old_grid, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>n/site=%{z:.4f}<extra></extra>',
            colorbar=dict(title='n/site', x=0.28, y=0.5, len=0.25)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=filling_new_grid,
            x=V_values,
            y=U_values,
            colorscale='Plasma',
            text=np.round(filling_new_grid, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>n/site=%{z:.4f}<extra></extra>',
            showscale=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=filling_diff,
            x=V_values,
            y=U_values,
            colorscale='RdBu',
            zmid=0,
            text=np.round(filling_diff, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>Δn/site=%{z:.4f}<extra></extra>',
            colorbar=dict(title='Δn/site', x=1.02, y=0.5, len=0.25)
        ),
        row=2, col=3
    )
    
    # Row 3: Percentage and absolute differences
    fig.add_trace(
        go.Heatmap(
            z=energy_pct_diff,
            x=V_values,
            y=U_values,
            colorscale='RdBu',
            zmid=0,
            text=np.round(energy_pct_diff, 1),
            texttemplate='%{text}%',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>Error=%{z:.2f}%<extra></extra>',
            colorbar=dict(title='Error %', x=0.28, y=0.17, len=0.25)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=filling_pct_diff,
            x=V_values,
            y=U_values,
            colorscale='RdBu',
            zmid=0,
            text=np.round(filling_pct_diff, 1),
            texttemplate='%{text}%',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>Error=%{z:.2f}%<extra></extra>',
            showscale=False
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=np.abs(energy_diff),
            x=V_values,
            y=U_values,
            colorscale='Reds',
            text=np.round(np.abs(energy_diff), 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            hovertemplate='U=%{y}<br>V=%{x}<br>|ΔE/site|=%{z:.4f}<extra></extra>',
            colorbar=dict(title='|ΔE/site|', x=1.02, y=0.17, len=0.25)
        ),
        row=3, col=3
    )
    
    # Update axes labels
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title_text='V', row=i, col=j)
            fig.update_yaxes(title_text='U', row=i, col=j)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Old QSpin vs New General QSpin Comparison<br><sub>L={L}, Nc={Nc}, t={t}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio}</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=900,
        width=1200,
        showlegend=False
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Max absolute energy difference: {np.nanmax(np.abs(energy_diff)):.6f}")
    print(f"Mean absolute energy difference: {np.nanmean(np.abs(energy_diff)):.6f}")
    print(f"Max absolute filling difference: {np.nanmax(np.abs(filling_diff)):.6f}")
    print(f"Mean absolute filling difference: {np.nanmean(np.abs(filling_diff)):.6f}")
    print(f"Max energy percentage error: {np.nanmax(np.abs(energy_pct_diff)):.2f}%")
    print(f"Max filling percentage error: {np.nanmax(np.abs(filling_pct_diff)):.2f}%")
    
    # Save to HTML if requested
    if save_html:
        fig.write_html(output_file)
        print(f"\nFigure saved to: {output_file}")
    
    return fig


if __name__ == "__main__":
    # Define parameter ranges
    U_values = np.linspace(0, 10, 6)  # 6 U values from 0 to 10
    V_values = np.linspace(0, 5, 6)   # 6 V values from 0 to 5
    
    # Run comparison
    fig = compare_old_new_quspin(
        U_values=U_values,
        V_values=V_values,
        t=1.0,
        L=8,
        Nc=2,
        int_sep_ratio=(1, 4),
        v_sep_ratio=(1, 2),
        save_html=True,
        output_file='old_vs_new_quspin_comparison.html'
    )
    
    # Display completion message
    print("\nComparison complete!")