"""
Comparison functions for different physics implementations.

This module provides functions to compare different physics implementations,
particularly focusing on QuSpin vs TenPy for the same physical models.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from aah_code.main import run_cluster_method, run_dmrg_method


# Set Plotly to use browser renderer to avoid nbformat issues
import plotly.io as pio
pio.renderers.default = "browser"

def quspin_vs_tenpy_heatmap(U_values, V_values, system_size=10, t=1.0, include_dmrg=False, chi=32, mu_subtraction=True):
    """
    Create a heatmap comparing QuSpin vs TenPy implementations for pi-pi modulation case.
    
    This function compares the QuSpin pi-pi implementation against the standard TenPy
    cluster method implementation across a parameter space of U and V values.
    Optionally includes DMRG as a third reference method.
    
    Parameters
    ----------
    U_values : array-like
        Array of Hubbard U interaction values to test
    V_values : array-like  
        Array of V interaction values to test
    system_size : int, optional
        Size of the system (default: 10)
    t : float, optional
        Hopping parameter (default: 1.0)
    include_dmrg : bool, optional
        Whether to include DMRG comparison (default: False)
    chi : int, optional
        Bond dimension for DMRG (default: 32)
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure with energy and filling error heatmaps
    """
    
    # Initialize grids for storing results
    energy_tenpy_grid = np.zeros((len(U_values), len(V_values)))
    energy_quspin_grid = np.zeros((len(U_values), len(V_values)))
    filling_tenpy_grid = np.zeros((len(U_values), len(V_values)))
    filling_quspin_grid = np.zeros((len(U_values), len(V_values)))
    
    if include_dmrg:
        energy_dmrg_grid = np.zeros((len(U_values), len(V_values)))
        filling_dmrg_grid = np.zeros((len(U_values), len(V_values)))
    
    print("Comparing QuSpin vs TenPy implementations...")
    print(f"System size: {system_size}, t: {t}")
    print(f"Testing {len(U_values)} U values × {len(V_values)} V values")
    
    for U_index, U in enumerate(tqdm(U_values, desc='U values')):
        for V_index, V in enumerate(V_values):
            mu_0 = U / 2  # Half-filling condition
            
            print(f"\n--- U = {U:.2f}, V = {V:.2f}, μ₀ = {mu_0:.2f} ---")
            
            # TenPy method (standard cluster method)
            print("Running TenPy cluster method...")
            energy_tenpy, filling_tenpy = run_cluster_method(
                U=U, mu_0=mu_0, V=V, t=t, system_size=system_size, ham_lib='tenpy'
            )
            energy_tenpy_subtracted = energy_tenpy + (mu_0 * filling_tenpy)
            
            # QuSpin method (pi-pi modulation)
            print("Running QuSpin pi-pi method...")
            energy_quspin, filling_quspin = run_cluster_method(
                U=U, mu_0=mu_0, V=V, t=t, system_size=system_size, ham_lib='quspin'
            )
            energy_quspin_subtracted = energy_quspin + (mu_0 * filling_quspin)
            
            
            # DMRG method (if requested)
            if include_dmrg:
                print("Running DMRG method...")
                energy_dmrg, filling_dmrg, psi_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
                energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)
                
                energy_dmrg_grid[U_index, V_index] = energy_dmrg_subtracted
                filling_dmrg_grid[U_index, V_index] = filling_dmrg
            
            # Store results in grids
            energy_tenpy_grid[U_index, V_index] = energy_tenpy_subtracted
            energy_quspin_grid[U_index, V_index] = energy_quspin_subtracted
            filling_tenpy_grid[U_index, V_index] = filling_tenpy / system_size
            filling_quspin_grid[U_index, V_index] = filling_quspin / system_size
            
            print(f"TenPy:  E={energy_tenpy:.3f}, n={filling_tenpy/system_size:.3f},E_sub: {energy_tenpy_subtracted/system_size:.3f}")
            print(f"QuSpin: E={energy_quspin:.3f}, n={filling_quspin/system_size:.3f},E_sub: {energy_quspin_subtracted/system_size:.3f}")
            
            if include_dmrg:
                print(f"DMRG:   E={energy_dmrg:.3f}, n={filling_dmrg:.3f},E_sub: {energy_dmrg_subtracted:.3f}")
            
    
    #Calculate per-site density
    energy_tenpy_grid = energy_tenpy_grid / system_size
    energy_quspin_grid = energy_quspin_grid / system_size
    # Calculate percentage differences 
    percentage_diff_grid_energy_quspin_tenpy = 100 * (energy_quspin_grid - energy_tenpy_grid) / np.abs(energy_tenpy_grid)
    percentage_diff_grid_filling_quspin_tenpy = 100 * (filling_quspin_grid - filling_tenpy_grid) / np.abs(filling_tenpy_grid)
    
    # Handle cases where TenPy result is zero to avoid division by zero
    percentage_diff_grid_energy_quspin_tenpy = np.where(
        np.abs(energy_tenpy_grid) < 1e-12, 
        100 * (energy_quspin_grid - energy_tenpy_grid), 
        percentage_diff_grid_energy_quspin_tenpy
    )
    percentage_diff_grid_filling_quspin_tenpy = np.where(
        np.abs(filling_tenpy_grid) < 1e-12,
        100 * (filling_quspin_grid - filling_tenpy_grid),
        percentage_diff_grid_filling_quspin_tenpy
    )

    if include_dmrg:
        # Calculate DMRG comparisons
        percentage_diff_grid_energy_quspin_dmrg = 100 * (energy_quspin_grid - energy_dmrg_grid) / np.abs(energy_dmrg_grid)
        percentage_diff_grid_filling_quspin_dmrg = 100 * (filling_quspin_grid - filling_dmrg_grid) / np.abs(filling_dmrg_grid)
        percentage_diff_grid_energy_tenpy_dmrg = 100 * (energy_tenpy_grid - energy_dmrg_grid) / np.abs(energy_dmrg_grid)
        percentage_diff_grid_filling_tenpy_dmrg = 100 * (filling_tenpy_grid - filling_dmrg_grid) / np.abs(filling_dmrg_grid)
        
        # Handle division by zero for DMRG
        percentage_diff_grid_energy_quspin_dmrg = np.where(
            np.abs(energy_dmrg_grid) < 1e-12, 
            100 * (energy_quspin_grid - energy_dmrg_grid), 
            percentage_diff_grid_energy_quspin_dmrg
        )
        percentage_diff_grid_filling_quspin_dmrg = np.where(
            np.abs(filling_dmrg_grid) < 1e-12,
            100 * (filling_quspin_grid - filling_dmrg_grid),
            percentage_diff_grid_filling_quspin_dmrg
        )
        percentage_diff_grid_energy_tenpy_dmrg = np.where(
            np.abs(energy_dmrg_grid) < 1e-12, 
            100 * (energy_tenpy_grid - energy_dmrg_grid), 
            percentage_diff_grid_energy_tenpy_dmrg
        )
        percentage_diff_grid_filling_tenpy_dmrg = np.where(
            np.abs(filling_dmrg_grid) < 1e-12,
            100 * (filling_tenpy_grid - filling_dmrg_grid),
            percentage_diff_grid_filling_tenpy_dmrg
        )

        

        
        # Calculate max energy error for unified color scale
        max_energy_error = max(np.max(np.abs(percentage_diff_grid_energy_quspin_tenpy)), 
                               np.max(np.abs(percentage_diff_grid_energy_quspin_dmrg)),
                               np.max(np.abs(percentage_diff_grid_energy_tenpy_dmrg)))
        
        # Create 3x2 subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['QuSpin vs TenPy: Energy (%)', 'QuSpin vs DMRG: Energy (%)', 'TenPy vs DMRG: Energy (%)',
                           'QuSpin vs TenPy: Filling (%)', 'QuSpin vs DMRG: Filling (%)', 'TenPy vs DMRG: Filling (%)'],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Energy heatmaps with unified color scale (Row 1)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy_quspin_tenpy,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                showscale=False,
                text=np.round(percentage_diff_grid_energy_quspin_tenpy, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy_quspin_dmrg,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                showscale=False,
                text=np.round(percentage_diff_grid_energy_quspin_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy_tenpy_dmrg,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                colorbar=dict(title='Energy Error (%)', x=1.02, y=0.75, len=0.4),
                text=np.round(percentage_diff_grid_energy_tenpy_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=1, col=3
        )
        
        # Filling heatmaps (Row 2)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling_quspin_tenpy,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                showscale=False,
                text=np.round(percentage_diff_grid_filling_quspin_tenpy, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling_quspin_dmrg,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                showscale=False,
                text=np.round(percentage_diff_grid_filling_quspin_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling_tenpy_dmrg,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                colorbar=dict(title='Filling Error (%)', x=1.02, y=0.25, len=0.4),
                text=np.round(percentage_diff_grid_filling_tenpy_dmrg, 1),
                texttemplate='%{text}%',
                textfont={"size": 8}
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f'Three-way Comparison: QuSpin π-π vs TenPy vs DMRG (L={system_size}, t={t}, χ={chi})',
            height=800,
            width=1200
        )
        
        # Update axes for all subplots
        for row in [1, 2]:
            for col in [1, 2, 3]:
                fig.update_xaxes(title_text='V', row=row, col=col)
                fig.update_yaxes(title_text='U', row=row, col=col)
    
    else:
        # Create 2-panel subplot layout (original)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Energy Error: QuSpin vs TenPy (%)', 'Filling Error: QuSpin vs TenPy (%)'],
            horizontal_spacing=0.15
        )
        
        # Energy heatmap
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy_quspin_tenpy,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Energy Error (%)', x=0.45),
                text=np.round(percentage_diff_grid_energy_quspin_tenpy, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='U=%{y}<br>V=%{x}<br>Energy Error=%{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Filling heatmap  
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling_quspin_tenpy,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Filling Error (%)', x=1.02),
                text=np.round(percentage_diff_grid_filling_quspin_tenpy, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='U=%{y}<br>V=%{x}<br>Filling Error=%{z:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'QuSpin π-π vs TenPy Cluster Method Comparison (L={system_size}, t={t})',
            height=500,
            width=900
        )
        
        # Update axes
        fig.update_xaxes(title_text='V', row=1, col=1)
        fig.update_yaxes(title_text='U', row=1, col=1)
        fig.update_xaxes(title_text='V', row=1, col=2) 
        fig.update_yaxes(title_text='U', row=1, col=2)
    
    return fig


def detailed_comparison_at_point(U, V, system_size=10, t=1.0):
    """
    Perform detailed comparison between QuSpin and TenPy at a single parameter point.
    
    Parameters
    ----------
    U : float
        Hubbard U interaction value
    V : float
        V interaction value  
    system_size : int, optional
        Size of the system (default: 10)
    t : float, optional
        Hopping parameter (default: 1.0)
        
    Returns
    -------
    dict
        Dictionary containing detailed comparison results
    """
    mu_0 = U / 2  # Half-filling condition
    
    print(f"Detailed comparison at U={U:.2f}, V={V:.2f}, μ₀={mu_0:.2f}")
    print(f"System size: {system_size}, t: {t}")
    
    # TenPy method
    print("\n=== TenPy Cluster Method ===")
    energy_tenpy, filling_tenpy = run_cluster_method(
        U=U, mu_0=mu_0, V=V, t=t, system_size=system_size, ham_lib='tenpy'
    )
    energy_tenpy_subtracted = energy_tenpy + mu_0 * filling_tenpy
    
    # QuSpin method  
    print("\n=== QuSpin π-π Method ===")
    energy_quspin, filling_quspin = run_cluster_method(
        U=U, mu_0=mu_0, V=V, t=t, system_size=system_size, ham_lib='quspin'
    )
    energy_quspin_subtracted = energy_quspin + mu_0 * filling_quspin
    
    # Calculate differences
    energy_diff = energy_quspin_subtracted - energy_tenpy_subtracted
    filling_diff = filling_quspin/system_size - filling_tenpy/system_size
    
    energy_rel_diff = 100 * energy_diff / np.abs(energy_tenpy_subtracted) if np.abs(energy_tenpy_subtracted) > 1e-12 else np.inf
    filling_rel_diff = 100 * filling_diff / np.abs(filling_tenpy/system_size) if np.abs(filling_tenpy/system_size) > 1e-12 else np.inf
    
    results = {
        'parameters': {'U': U, 'V': V, 'mu_0': mu_0, 't': t, 'L': system_size},
        'tenpy': {
            'energy': energy_tenpy,
            'energy_subtracted': energy_tenpy_subtracted, 
            'filling': filling_tenpy,
            'filling_per_site': filling_tenpy/system_size
        },
        'quspin': {
            'energy': energy_quspin,
            'energy_subtracted': energy_quspin_subtracted,
            'filling': filling_quspin, 
            'filling_per_site': filling_quspin/system_size
        },
        'differences': {
            'energy_absolute': energy_diff,
            'filling_absolute': filling_diff,
            'energy_relative_percent': energy_rel_diff,
            'filling_relative_percent': filling_rel_diff
        }
    }
    
    print(f"\n=== Comparison Results ===")
    print(f"TenPy:  E={energy_tenpy_subtracted:.6f}, n={filling_tenpy/system_size:.6f}")
    print(f"QuSpin: E={energy_quspin_subtracted:.6f}, n={filling_quspin/system_size:.6f}")
    print(f"ΔE = {energy_diff:.6f} ({energy_rel_diff:.3f}%)")
    print(f"Δn = {filling_diff:.6f} ({filling_rel_diff:.3f}%)")
    
    return results


def get_site_resolved_spectra(U, V, system_size=10, t=1.0):
    """
    Get the full site-resolved number spectra from both TenPy and QuSpin before thermodynamic averaging.
    
    This function extracts the raw eigenvalue and particle number spectra from the cluster method
    calculations, providing detailed state-by-state information that gets averaged out in the 
    standard comparison functions.
    
    Parameters
    ----------
    U : float
        Hubbard U interaction value
    V : float
        V interaction value  
    system_size : int, optional
        Size of the system (default: 10)
    t : float, optional
        Hopping parameter (default: 1.0)
        
    Returns
    -------
    dict
        Dictionary containing raw spectra with keys:
        - 'parameters': Input parameters used
        - 'tenpy': TenPy results with 'energy_spectrum', 'number_spectrum', 'k_points'
        - 'quspin': QuSpin results with 'energy_spectrum', 'number_spectrum', 'k_points'
        - 'cluster_info': Information about cluster structure
    """
    mu_0 = U / 2  # Half-filling condition
    
    print(f"Getting site-resolved spectra at U={U:.2f}, V={V:.2f}, μ₀={mu_0:.2f}")
    print(f"System size: {system_size}, t: {t}")
    
    # Import required modules
    from aah_code.clusters import ClusterExperiment
    from aah_code.hamiltonian import FullSpectrum
    from aah_code.global_params import StatesParams, HamiltonianParams
    
    # Create cluster experiment (same as in run_cluster_method)
    cluster_size = 2  # 2-site clusters
    cluster_k_generator = system_size // 2  # π separation case
    
    cluster_experiment = ClusterExperiment(
        cluster_size=cluster_size,
        lattice_points=system_size,
        cluster_k_generator=cluster_k_generator,
    )
    
    # Generate k-points grid and clusters
    k_points = cluster_experiment.generate_clusters()
    
    # Set up parameters
    state_params = StatesParams(spin_states=2)
    physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
    
    # Initialize results dictionary
    results = {
        'parameters': {'U': U, 'V': V, 'mu_0': mu_0, 't': t, 'L': system_size},
        'cluster_info': {
            'cluster_size': cluster_size,
            'cluster_k_generator': cluster_k_generator,
            'n_clusters': k_points.shape[0],
            'total_k_points': k_points.size
        }
    }
    
    # Get TenPy spectra
    print("=== Getting TenPy Full Spectrum ===")
    tenpy_spectrum_object = FullSpectrum(k_points, state_params, physical_params, ham_lib='tenpy')
    tenpy_k_points, tenpy_energy_spectrum, tenpy_number_spectrum, tenpy_spin_spectrum = tenpy_spectrum_object.get_full_spectrum()
    
    results['tenpy'] = {
        'energy_spectrum': tenpy_energy_spectrum,  # List of eigenvalue arrays per cluster
        'number_spectrum': tenpy_number_spectrum,  # List of particle number arrays per cluster  
        'k_points': tenpy_k_points,
        'spin_spectrum': tenpy_spin_spectrum  # Spin spectrum data
    }
    
    # Get QuSpin spectra
    print("=== Getting QuSpin Full Spectrum ===")
    quspin_spectrum_object = FullSpectrum(k_points, state_params, physical_params, ham_lib='quspin')
    quspin_k_points, quspin_energy_spectrum, quspin_number_spectrum, quspin_spin_spectrum = quspin_spectrum_object.get_full_spectrum()
    
    results['quspin'] = {
        'energy_spectrum': quspin_energy_spectrum,  # List of eigenvalue arrays per cluster
        'number_spectrum': quspin_number_spectrum,  # List of particle number arrays per cluster
        'k_points': quspin_k_points,
        'spin_spectrum': quspin_spin_spectrum  # Spin spectrum data
    }
    
    # Print summary information
    print("\n=== Spectrum Summary ===")
    print(f"Number of clusters: {results['cluster_info']['n_clusters']}")
    print(f"Cluster size: {cluster_size}")
    
    # Show some example spectra details
    if len(results['tenpy']['energy_spectrum']) > 0:
        tenpy_example = results['tenpy']['energy_spectrum'][0]
        quspin_example = results['quspin']['energy_spectrum'][0]
        
        print(f"First cluster - TenPy:  {len(tenpy_example)} energy states")
        print(f"First cluster - QuSpin: {len(quspin_example)} energy states") 
        
        if len(tenpy_example) > 0 and len(quspin_example) > 0:
            print(f"TenPy  ground state energy: {tenpy_example[0]:.6f}")
            print(f"QuSpin ground state energy: {quspin_example[0]:.6f}")
            
    # Check for particle number differences in each cluster
    print("\n=== Particle Number Analysis ===")
    for i, (tenpy_n, quspin_n) in enumerate(zip(results['tenpy']['number_spectrum'], 
                                                results['quspin']['number_spectrum'])):
        print(f"Cluster {i}: TenPy {len(tenpy_n)} states, QuSpin {len(quspin_n)} states")
        if len(tenpy_n) != len(quspin_n):
            print(f"  ⚠️  State count mismatch in cluster {i}")
        
        # Compare particle number ranges
        if len(tenpy_n) > 0 and len(quspin_n) > 0:
            tenpy_n_range = f"{np.min(tenpy_n):.1f}-{np.max(tenpy_n):.1f}"
            quspin_n_range = f"{np.min(quspin_n):.1f}-{np.max(quspin_n):.1f}" 
            print(f"  TenPy  particle range: {tenpy_n_range}")
            print(f"  QuSpin particle range: {quspin_n_range}")
    
    return results


if __name__ == "__main__":
    # Example usage: small parameter space scan
    U_values = np.linspace(0, 5, 5)
    V_values = np.linspace(1e-8, 5, 5) 
    
    # print("Creating QuSpin vs TenPy comparison heatmap...")
    # fig = quspin_vs_tenpy_heatmap(U_values, V_values, system_size=20)
    

    

    # #try to find where the number spectra are different
    # results_dict=get_site_resolved_spectra(U=10.0, V=1e-6, system_size=20)
    # tenpy_number_spectrum=results_dict["tenpy"]["number_spectrum"]
    # quspin_number_spectrum=results_dict["quspin"]["number_spectrum"]


    # print(results_dict.keys())
    
    # print(f'tenpy gs numbers: {tenpy_number_spectrum[5,0,:]}')
    # print(f'quspin gs numbers: {quspin_number_spectrum[5,:4,:]}')
    # print(f'quspin k points: {results_dict["quspin"]["k_points"][5,:4,:]}')
    # print(f'quspin spectrum: {results_dict["quspin"]["energy_spectrum"][5,:4]}')
    # print(f'tenpy spectrum: {results_dict["tenpy"]["energy_spectrum"][5,:4]}')

    # print()
    
    # print("\n" + "="*50)
    # print("Creating three-way comparison with DMRG...")
    fig_3way = quspin_vs_tenpy_heatmap(U_values, V_values, system_size=100, include_dmrg=True, chi=32, mu_subtraction=True)
    fig_3way.show()
    
    # Example usage: detailed comparison at specific point
    #print("\n" + "="*50)
    #detailed_results = detailed_comparison_at_point(U=1.0, V=0.5, system_size=6)
    
    # Example usage: get site-resolved spectra
    #print("\n" + "="*50)
    #print("Getting site-resolved spectra...")
    #spectra_results = get_site_resolved_spectra(U=1.0, V=0.5, system_size=20)