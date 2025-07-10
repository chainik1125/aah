"""
Comparison script between Real-Space DMRG and Cluster Method
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.hamiltonian import FullSpectrum, Hubbard1D, SpectrumSolver
from aah_code.global_params import StatesParams, HamiltonianParams
from aah_code.real_space_dmrg import get_gnd


def run_twosite(U,mu_0,V,t=1,system_size=10):
    #TODO: Add averaging over twisted boundary conditions
    two_site_hamiltonian_spin_zero=np.array([[U+V,-t,t,0],
                                            [-t,0,0,t],
                                            [t,0,0,-t],
                                            [0,t,-t,U-V]])
    two_site_hamiltonian_spin_zero=two_site_hamiltonian_spin_zero-2*(mu_0)*np.eye(np.shape(two_site_hamiltonian_spin_zero)[0])

    eigvals,eigvecs=np.linalg.eigh(two_site_hamiltonian_spin_zero)

    subtracted_eigvals=eigvals+2*mu_0

    return subtracted_eigvals[0]/2

def run_cluster_method(U, mu_0, V=0, t=1, system_size=10):
    """
    Run cluster method calculation using corrected FullSpectrum class
    
    Args:
        U: Hubbard interaction strength
        mu_0: Chemical potential
        V: Staggered potential (default 0)
        t: Hopping parameter (default 1)
        system_size: Number of lattice sites
    
    Returns:
        (energy, filling): Total energy and filling
    """
    # Create grid and partition into clusters (proper cluster method)
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
    
    # Now use the corrected FullSpectrum class
    full_spectrum_object = FullSpectrum(k_points, state_params, physical_params)
    cluster_spectra = full_spectrum_object.get_full_spectrum()
    
    # Get ground state expectations (zero temperature)
    system_expectations, cluster_expectations = full_spectrum_object.get_cluster_thermodynamic_expectations(
        cluster_spectra, temperature=None
    )
    
    energy, filling, spin = system_expectations
    
    return energy, filling

def run_dmrg_method(U, mu_0, V=0, t=1, system_size=10, chi=32):
    """
    Run real-space DMRG calculation
    
    Args:
        U: Hubbard interaction strength
        mu_0: Chemical potential
        V: Staggered potential (default 0)
        t: Hopping parameter (default 1)
        system_size: Number of lattice sites
        chi: Bond dimension for DMRG
    
    Returns:
        (energy, filling): Total energy and filling
    """
    energy, psi, filling = get_gnd(L=system_size, chi=chi, U=U, t=t, mu=mu_0, V=V)
    
    return energy, filling

def compare_half_filling_U_fixed_V(U_values,mu_subtraction:bool=True,V:float=0):
    """
    Comparison across range of U values at half-filling
    """
    print("Starting DMRG vs Cluster Method Comparison")
    
    # Parameters
    system_size = 10
    #U_values = np.array([0,5])#np.array([0,1,2,3,4,5])  # Range of U values
     
    t = 1  # Hopping parameter
    chi = 32  # Bond dimension for DMRG
    
    print(f"System size: {system_size}")
    print(f"U values: {U_values}")
    print(f"Bond dimension (DMRG): {chi}")
    print("Running at half-filling (μ₀ = U/2) for each U")
    
    # Store results
    energies_cluster = []
    energies_dmrg = []
    energies_twosite=[]
    fillings_cluster = []
    fillings_dmrg = []
    fillings_twosite=[]#TODO: implement twosite filling for mu_0 away from haf-filled value
    mu_0_values=[]
    
    
    
    print("\nRunning calculations...")
    for U in tqdm(U_values, desc="U values"):
        mu_0 = U / 2  # Half-filling condition
        mu_0_values.append(mu_0)

        print(f"\n--- U = {U}, μ₀ = {mu_0} ---")
        
        # Cluster method
        print("Running cluster method...")
        energy_cluster, filling_cluster = run_cluster_method(U, mu_0, V, t, system_size)
        energies_cluster.append(energy_cluster)
        fillings_cluster.append(filling_cluster)
        
        # DMRG method
        print("Running DMRG method...")
        energy_dmrg, filling_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
        energies_dmrg.append(energy_dmrg)
        fillings_dmrg.append(filling_dmrg)

        #twosite
        energy_twosite=run_twosite(U,mu_0,V,t,system_size)
        filling_twosite=1 #by definition - wrong when mu_0 away from half-filling
        energies_twosite.append(energy_twosite)
        fillings_twosite.append(filling_twosite)

        
        print(f"Cluster: E={energy_cluster:.3f}, n={filling_cluster:.3f}")
        print(f"DMRG:    E={energy_dmrg:.3f}, n={filling_dmrg:.3f}")
        print(f"Two site:    E={energy_twosite:.3f}, n={2:.3f}")
    
    # Convert to numpy arrays
    energies_cluster = np.array(energies_cluster)
    energies_dmrg = np.array(energies_dmrg)
    energies_twosite=np.array(energies_twosite)
    fillings_cluster = np.array(fillings_cluster)
    fillings_dmrg = np.array(fillings_dmrg)
    fillings_twosite=np.array(fillings_twosite)
    
    # Create comparison plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            ['Energy Density vs U (Half-Filling)',
            'Energy percentage difference',
            'Filling vs U (Half-Filling)', 
            'Filling percentage difference']
        ),
        #specs=[[{"secondary_y": False, "colspan": 2}, None],
        #       [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    #Add back in the mu part to control for this part of the energies
    if mu_subtraction:
        total_energies_cluster_subtracted=energies_cluster+np.array(mu_0_values)*filling_cluster
        total_energies_dmrg_subtracted=energies_dmrg+np.array(mu_0_values)*(fillings_dmrg*system_size)
    else:
        total_energies_cluster_subtracted=energies_cluster
        total_energies_dmrg_subtracted=energies_dmrg

    percentage_difference_energies=100*(total_energies_cluster_subtracted-total_energies_dmrg_subtracted)/np.abs(total_energies_dmrg_subtracted)
    percentage_difference_energies_twosite=100*(energies_twosite*system_size-total_energies_dmrg_subtracted)/np.abs(total_energies_dmrg_subtracted)
    percentage_difference_fillings=100*(fillings_cluster/system_size-fillings_dmrg)/np.abs(fillings_dmrg)
        
    # Plot 1: Energy Density vs U (spanning two columns)
    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=total_energies_cluster_subtracted/system_size,
            mode='lines+markers',
            name='Cluster Method',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

        
    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=total_energies_dmrg_subtracted/system_size,
            mode='lines+markers',
            name='DMRG',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )


    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=energies_twosite,
            mode='lines+markers',
            name='twosite',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=percentage_difference_energies,
            mode='lines+markers',
            name='Energies error %',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=percentage_difference_energies_twosite,
            mode='lines+markers',
            name='Energies error %',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Plot 2: Filling vs U (bottom left)
    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=fillings_cluster/system_size,
            mode='lines+markers',
            name='Cluster Method',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=fillings_dmrg,
            mode='lines+markers',
            name='DMRG',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 3: Energy Difference vs U (bottom right)
    energy_diff = np.abs(total_energies_cluster_subtracted - total_energies_dmrg_subtracted)
    fig.add_trace(
        go.Scatter(
            x=U_values,
            y=percentage_difference_fillings,
            mode='lines+markers',
            name='Fillings error %',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'DMRG vs Cluster Method Comparison (L={system_size}, Half-Filling, V={V})',
        #height=800,
        #width=1200,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="U", row=1, col=1)
    fig.update_yaxes(title_text="Energy Density", row=1, col=1)

    fig.update_yaxes(title_text="Energy error (%)", row=1, col=2, tickformat=".1f", ticksuffix="%")
    
    fig.update_xaxes(title_text="U", row=2, col=1)
    fig.update_yaxes(title_text="Filling Density", row=2, col=1)
    
    fig.update_xaxes(title_text="U", row=2, col=2)
    fig.update_yaxes(title_text="Filling error (%)", row=2, col=2, tickformat=".1f", ticksuffix="%")
    
    # Show the plot
    fig.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for i, U in enumerate(U_values):
        print(f"U = {U}:")
        print(f"  Cluster: E = {energies_cluster[i]:.6f}, n = {fillings_cluster[i]:.6f}")
        print(f"  DMRG:    E = {energies_dmrg[i]:.6f}, n = {fillings_dmrg[i]:.6f}")
        print(f"  Diff:    ΔE = {100*abs(energies_cluster[i] - energies_dmrg[i])/abs(energies_dmrg[i]):.2f}, Δn = {abs(fillings_cluster[i] - fillings_dmrg[i]):.6f}")
        print()

def U_V_difference_heatmap(U_values,V_values,include_twosite=False):
    """
    heatmap_dict[(U,V):((corrected_energy_cluster,filling_cluster),(energy_dmrg,filling_dmrg))]
    """
    heatmap_dict={}
    energy_cluster_grid=np.zeros((len(U_values),len(V_values)))
    energy_dmrg_grid=np.zeros((len(U_values),len(V_values)))
    filling_cluster_grid=np.zeros((len(U_values),len(V_values)))
    filling_dmrg_grid=np.zeros((len(U_values),len(V_values)))
    
    if include_twosite:
        energy_twosite_grid=np.zeros((len(U_values),len(V_values)))
        filling_twosite_grid=np.zeros((len(U_values),len(V_values)))

    
    system_size=10
    t=1
    chi=32

    for U_index,U in enumerate(tqdm(U_values,desc='U values')):
        for V_index,V in enumerate(V_values):
                    mu_0 = U / 2  # Half-filling condition
                    

                    print(f"\n--- U = {U}, μ₀ = {mu_0} ---")
                    
                    # Cluster method
                    print("Running cluster method...")
                    energy_cluster, filling_cluster = run_cluster_method(U, mu_0, V, t, system_size)
                    energy_cluster_subtracted=energy_cluster+mu_0*filling_cluster
                    
                    cluster_tuple=(energy_cluster_subtracted,filling_cluster/system_size)

                    energy_cluster_grid[U_index,V_index]=energy_cluster_subtracted
                    filling_cluster_grid[U_index,V_index]=filling_cluster/system_size
                    
                    # DMRG method
                    print("Running DMRG method...")
                    energy_dmrg, filling_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
                    energy_dmrg_subtracted=energy_dmrg+(mu_0*filling_dmrg*system_size)

                    dmrg_tuple=(energy_dmrg_subtracted,filling_dmrg)

                    energy_dmrg_grid[U_index,V_index]=energy_dmrg_subtracted
                    filling_dmrg_grid[U_index,V_index]=filling_dmrg

                    if include_twosite:
                        # Two-site method
                        print("Running two-site method...")
                        energy_twosite = run_twosite(U, mu_0, V, t, system_size)
                        filling_twosite = 1.0  # Two-site at half-filling
                        energy_twosite_grid[U_index,V_index]=energy_twosite*system_size
                        filling_twosite_grid[U_index,V_index]=filling_twosite
                        print(f"Two-site: E={energy_twosite:.3f}")

                    #heatmap_dict[(U,V)]=(cluster_tuple,dmrg_tuple)

                    
                    
                    print(f"Cluster: E={energy_cluster:.3f}, n={filling_cluster:.3f}")
                    print(f"DMRG:    E={energy_dmrg:.3f}, n={filling_dmrg:.3f}")
    
    percentage_diff_grid_energy=100*(energy_cluster_grid-energy_dmrg_grid)/np.abs(energy_dmrg_grid)
    percentage_diff_grid_filling=100*(filling_cluster_grid-filling_dmrg_grid)/np.abs(filling_dmrg_grid)
    
    if include_twosite:
        percentage_diff_grid_energy_twosite=100*(energy_twosite_grid-energy_dmrg_grid)/np.abs(energy_dmrg_grid)
        percentage_diff_grid_filling_twosite=100*(filling_twosite_grid-filling_dmrg_grid)/np.abs(filling_dmrg_grid)
        
        # Calculate max energy error for unified color scale
        max_energy_error = max(np.max(np.abs(percentage_diff_grid_energy)), 
                              np.max(np.abs(percentage_diff_grid_energy_twosite)))
        
        fig=make_subplots(
            rows=2, cols=2,
            subplot_titles=['Cluster vs DMRG: Energy Error (%)', 'Two-site vs DMRG: Energy Error (%)',
                           'Cluster vs DMRG: Filling Error (%)', 'Two-site vs DMRG: Filling Error (%)'],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )
    else:
        fig=make_subplots(
            rows=1, cols=2,
            subplot_titles=['Energy Error (%)', 'Filling Error (%)'],
            horizontal_spacing=0.1
        )

    if include_twosite:
        # Energy heatmaps with unified color scale (Row 1)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                showscale=False,
                text=np.round(percentage_diff_grid_energy, 0).astype(int),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy_twosite,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmin=-max_energy_error,
                zmax=max_energy_error,
                zmid=0,
                colorbar=dict(title='Energy Error (%)', x=1.02, y=0.75, len=0.4),
                text=np.round(percentage_diff_grid_energy_twosite, 0).astype(int),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=1, col=2
        )
        
        # Filling heatmaps (Row 2)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                showscale=False,
                text=np.round(percentage_diff_grid_filling, 0).astype(int),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling_twosite,
                x=V_values,
                y=U_values,
                colorscale='Viridis',
                colorbar=dict(title='Filling Error (%)', x=1.02, y=0.25, len=0.4),
                text=np.round(percentage_diff_grid_filling_twosite, 0).astype(int),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=2, col=2
        )
    else:
        # Energy heatmap (Cluster vs DMRG)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_energy,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Energy Error (%)', x=0.45),
                text=np.round(percentage_diff_grid_energy, 0).astype(int),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=1, col=1
        )
        
        # Filling heatmap (Cluster vs DMRG)
        fig.add_trace(
            go.Heatmap(
                z=percentage_diff_grid_filling,
                x=V_values,
                y=U_values,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Filling Error (%)', x=1.02),
                text=np.round(percentage_diff_grid_filling, 0).astype(int),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=1, col=2
        )
    
    # Update layout
    title_text = 'DMRG vs Cluster Method: Parameter Space Comparison'
    if include_twosite:
        title_text += ' (with Two-site)'
    
    fig.update_layout(
        title=title_text,
        height=800 if include_twosite else 500,
        #width=1000
    )
    
    # Update axes
    fig.update_xaxes(title_text='V', row=1, col=1)
    fig.update_yaxes(title_text='U', row=1, col=1)
    fig.update_xaxes(title_text='V', row=1, col=2)
    fig.update_yaxes(title_text='U', row=1, col=2)
    
    if include_twosite:
        fig.update_xaxes(title_text='V', row=2, col=1)
        fig.update_yaxes(title_text='U', row=2, col=1)
        fig.update_xaxes(title_text='V', row=2, col=2)
        fig.update_yaxes(title_text='U', row=2, col=2)
    
    return fig


if __name__ == "__main__":
    U_values=np.linspace(0,5,10)
    V_values=np.linspace(0,5,10)
    U_V_difference_heatmap(U_values,V_values,include_twosite=True).show()

    #compare_half_filling_U_fixed_V(U_values,V=2)