"""
Three-way comparison: Old QSpin vs New General Cluster vs DMRG
Based on quspin_vs_tenpy_heatmap but comparing old and new cluster methods.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from aah_code.cluster_model.full_spectrum_custom import run_general_cluster_method
from aah_code.hamiltonian import test_quick_mismatched, HamiltonianParams
from aah_code.main import run_dmrg_method

# Silence verbose logging
import logging
logging.getLogger('aah').setLevel(logging.WARNING)


def old_quspin_vs_new_general_heatmap(
    U_values, 
    V_values, 
    system_size=20,
    t=1.0, 
    include_dmrg=True, 
    chi=32,
    int_sep_ratio=(1, 10),  # For L=20, use L/10 to match old method
    v_sep_ratio=(1, 2)      # π modulation
):
    """
    Create a heatmap comparing Old QSpin (test_quick_mismatched) vs New General Cluster Method.
    Optionally includes DMRG as a third reference method.
    
    Parameters
    ----------
    U_values : array-like
        Array of Hubbard U interaction values to test
    V_values : array-like  
        Array of V interaction values to test
    system_size : int, optional
        Size of the system (default: 20)
    t : float, optional
        Hopping parameter (default: 1.0)
    include_dmrg : bool, optional
        Whether to include DMRG comparison (default: True)
    chi : int, optional
        Bond dimension for DMRG (default: 32)
    int_sep_ratio : tuple
        Interaction cluster separation for new method
    v_sep_ratio : tuple
        V-term separation for new method
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure with energy and filling error heatmaps
    """
    
    # Initialize grids for storing results
    energy_old_grid = np.zeros((len(U_values), len(V_values)))
    energy_new_grid = np.zeros((len(U_values), len(V_values)))
    filling_old_grid = np.zeros((len(U_values), len(V_values)))
    filling_new_grid = np.zeros((len(U_values), len(V_values)))
    
    if include_dmrg:
        energy_dmrg_grid = np.zeros((len(U_values), len(V_values)))
        filling_dmrg_grid = np.zeros((len(U_values), len(V_values)))
    
    print("Comparing Old QSpin vs New General Cluster Method...")
    print(f"System size: {system_size}, t: {t}")
    print(f"New method clustering: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print(f"Testing {len(U_values)} U values × {len(V_values)} V values")
    
    for U_index, U in enumerate(tqdm(U_values, desc='U values')):
        for V_index, V in enumerate(V_values):
            mu_0 = U / 2  # Half-filling condition
            
            print(f"\n--- U = {U:.2f}, V = {V:.2f}, μ₀ = {mu_0:.2f} ---")
            
            # Old QSpin method (test_quick_mismatched)
            print("Running old QSpin method...")
            physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
            system_expectations, _ = test_quick_mismatched(
                lattice_points=system_size, 
                cluster_size=2, 
                physical_params=physical_params
            )
            energy_old, filling_old, _ = system_expectations
            energy_old_subtracted = energy_old + (mu_0 * filling_old)
            
            # New General Cluster method
            print("Running new general cluster method...")
            energy_new, filling_new = run_general_cluster_method(
                U=U, mu_0=mu_0, V=V, t=t,
                L=system_size, Nc=2,
                int_sep_ratio=int_sep_ratio,
                v_sep_ratio=v_sep_ratio,
                use_simple_ham=True  # Always use simple ham for now (V terms have indexing issue)
            )
            energy_new_subtracted = energy_new + (mu_0 * filling_new)
            
            # DMRG method (if requested)
            if include_dmrg:
                print("Running DMRG method...")
                energy_dmrg, filling_dmrg, psi_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
                # For infinite DMRG, energy and filling are already per-site
                energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)
                
                energy_dmrg_grid[U_index, V_index] = energy_dmrg_subtracted
                filling_dmrg_grid[U_index, V_index] = filling_dmrg
            
            # Store results in grids (as per-site quantities)
            energy_old_grid[U_index, V_index] = energy_old_subtracted / system_size
            energy_new_grid[U_index, V_index] = energy_new_subtracted / system_size
            filling_old_grid[U_index, V_index] = filling_old / system_size
            filling_new_grid[U_index, V_index] = filling_new / system_size
            
            print(f"Old QSpin:  E/site={energy_old_subtracted/system_size:.3f}, n/site={filling_old/system_size:.3f}")
            print(f"New method: E/site={energy_new_subtracted/system_size:.3f}, n/site={filling_new/system_size:.3f}")
            
            if include_dmrg:
                print(f"DMRG:       E/site={energy_dmrg_subtracted:.3f}, n/site={filling_dmrg:.3f}")
    
    # Calculate percentage differences
    percentage_diff_energy_new_old = 100 * (energy_new_grid - energy_old_grid) / np.abs(energy_old_grid)
    percentage_diff_filling_new_old = 100 * (filling_new_grid - filling_old_grid) / np.abs(filling_old_grid)
    
    # Handle division by zero
    percentage_diff_energy_new_old = np.where(
        np.abs(energy_old_grid) < 1e-12, 
        100 * (energy_new_grid - energy_old_grid), 
        percentage_diff_energy_new_old
    )
    percentage_diff_filling_new_old = np.where(
        np.abs(filling_old_grid) < 1e-12,
        100 * (filling_new_grid - filling_old_grid),
        percentage_diff_filling_new_old
    )

    if include_dmrg:
        # Calculate DMRG comparisons
        percentage_diff_energy_old_dmrg = 100 * (energy_old_grid - energy_dmrg_grid) / np.abs(energy_dmrg_grid)
        percentage_diff_filling_old_dmrg = 100 * (filling_old_grid - filling_dmrg_grid) / np.abs(filling_dmrg_grid)
        percentage_diff_energy_new_dmrg = 100 * (energy_new_grid - energy_dmrg_grid) / np.abs(energy_dmrg_grid)
        percentage_diff_filling_new_dmrg = 100 * (filling_new_grid - filling_dmrg_grid) / np.abs(filling_dmrg_grid)
        
        # Handle division by zero for DMRG
        percentage_diff_energy_old_dmrg = np.where(
            np.abs(energy_dmrg_grid) < 1e-12, 
            100 * (energy_old_grid - energy_dmrg_grid), 
            percentage_diff_energy_old_dmrg
        )
        percentage_diff_filling_old_dmrg = np.where(
            np.abs(filling_dmrg_grid) < 1e-12,
            100 * (filling_old_grid - filling_dmrg_grid),
            percentage_diff_filling_old_dmrg
        )
        percentage_diff_energy_new_dmrg = np.where(
            np.abs(energy_dmrg_grid) < 1e-12, 
            100 * (energy_new_grid - energy_dmrg_grid), 
            percentage_diff_energy_new_dmrg
        )
        percentage_diff_filling_new_dmrg = np.where(
            np.abs(filling_dmrg_grid) < 1e-12,
            100 * (filling_new_grid - filling_dmrg_grid),
            percentage_diff_filling_new_dmrg
        )
        
        # Calculate max error for unified color scale
        max_energy_error = max(
            np.max(np.abs(percentage_diff_energy_new_old)), 
            np.max(np.abs(percentage_diff_energy_old_dmrg)),
            np.max(np.abs(percentage_diff_energy_new_dmrg))
        )
        
        # Create 3x2 subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'New vs Old: Energy (%)', 'Old vs DMRG: Energy (%)', 'New vs DMRG: Energy (%)',
                'New vs Old: Filling (%)', 'Old vs DMRG: Filling (%)', 'New vs DMRG: Filling (%)'
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Energy heatmaps with unified color scale (Row 1)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_energy_new_old,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                showscale=False,
                text=np.round(percentage_diff_energy_new_old, 1),
                texttemplate='%{text}%',
                textfont={"size": 8},
                hovertemplate='U=%{y}<br>V=%{x}<br>Error=%{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_energy_old_dmrg,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                showscale=False,
                text=np.round(percentage_diff_energy_old_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_energy_new_dmrg,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                colorbar=dict(title='Energy Error (%)', x=1.02, y=0.75, len=0.4),
                text=np.round(percentage_diff_energy_new_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=1, col=3
        )
        
        # Filling heatmaps (Row 2)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_filling_new_old,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                showscale=False,
                text=np.round(percentage_diff_filling_new_old, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_filling_old_dmrg,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                showscale=False,
                text=np.round(percentage_diff_filling_old_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_filling_new_dmrg,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                colorbar=dict(title='Filling Error (%)', x=1.02, y=0.25, len=0.4),
                text=np.round(percentage_diff_filling_new_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f'Three-way Comparison: Old QSpin vs New General vs DMRG (L={system_size}, t={t}, χ={chi})',
            height=800,
            width=1200
        )
        
        # Update axes for all subplots
        for row in [1, 2]:
            for col in [1, 2, 3]:
                fig.update_xaxes(title_text='V', row=row, col=col)
                fig.update_yaxes(title_text='U', row=row, col=col)
    
    else:
        # Create 2-panel subplot layout (without DMRG)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Energy Error: New vs Old (%)', 'Filling Error: New vs Old (%)'],
            horizontal_spacing=0.15
        )
        
        # Energy heatmap
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_energy_new_old,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Energy Error (%)', x=0.45),
                text=np.round(percentage_diff_energy_new_old, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hovertemplate='U=%{y}<br>V=%{x}<br>Error=%{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Filling heatmap
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_filling_new_old,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Filling Error (%)', x=1.02),
                text=np.round(percentage_diff_filling_new_old, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hovertemplate='U=%{y}<br>V=%{x}<br>Error=%{z:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Old QSpin vs New General Method (L={system_size})',
            height=500,
            width=1000
        )
        
        fig.update_xaxes(title_text='V')
        fig.update_yaxes(title_text='U')
    
    return fig


if __name__ == "__main__":
    # Test with L=20
    print("Testing three-way comparison for L=20")
    print("=" * 70)
    
    # Define parameter ranges
    U_values = np.linspace(0, 4, 5)  # 5x5 grid for reasonable computation time
    V_values = np.linspace(0, 2, 5)
    
    # Run comparison
    fig = old_quspin_vs_new_general_heatmap(
        U_values=U_values,
        V_values=V_values,
        system_size=20,
        t=1.0,
        include_dmrg=True,
        chi=32,
        int_sep_ratio=(1, 10),  # L/10 for L=20
        v_sep_ratio=(1, 2)      # π modulation
    )
    
    # Save figure
    fig.write_html("three_way_comparison_L20.html")
    print("\nFigure saved as 'three_way_comparison_L20.html'")
    
    # Also create a version without DMRG for faster testing
    print("\nCreating version without DMRG...")
    fig_no_dmrg = old_quspin_vs_new_general_heatmap(
        U_values=U_values,
        V_values=V_values,
        system_size=20,
        t=1.0,
        include_dmrg=False,
        int_sep_ratio=(1, 10),
        v_sep_ratio=(1, 2)
    )
    
    fig_no_dmrg.write_html("comparison_L20_no_dmrg.html")
    print("Figure saved as 'comparison_L20_no_dmrg.html'")