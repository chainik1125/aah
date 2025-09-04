"""
Run scripts for the general cluster method with arbitrary tilings.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from aah_code.cluster_model.model_ham import make_cluster_ham
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d


def run_general_cluster_method(
    U: float, 
    mu_0: float, 
    V: float = 0, 
    t: float = 1, 
    L: int = 10,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 2),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    ham_lib: str = 'quspin',
    temperature: Optional[float] = None
) -> Tuple[float, float]:
    """
    Run the generalized cluster method with arbitrary cluster tilings.
    
    Parameters
    ----------
    U : float
        Hubbard interaction strength
    mu_0 : float
        Chemical potential
    V : float
        Staggered potential amplitude
    t : float
        Hopping parameter
    L : int
        Total system size (number of k-points)
    Nc : int
        Cluster size
    int_sep_ratio : Tuple[int, int]
        Interaction cluster separation as (p, q) where separation = p/q * L
    v_sep_ratio : Tuple[int, int]
        V-term separation as (p, q) where separation = p/q * L
    ham_lib : str
        Hamiltonian library to use ('quspin' or 'tenpy')
    temperature : Optional[float]
        Temperature for thermodynamic expectations (None for T=0)
        
    Returns
    -------
    energy : float
        Total energy per site
    filling : float
        Average filling per site
    """
    
    print(f"Running general cluster method with L={L}, Nc={Nc}")
    print(f"int_sep_ratio={int_sep_ratio}, v_sep_ratio={v_sep_ratio}")
    
    # Generate all superclusters
    all_superclusters = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
    num_superclusters = all_superclusters.shape[0]
    
    print(f"Generated {num_superclusters} superclusters")
    print(f"Supercluster shape: {all_superclusters.shape}")
    
    # Store results for each supercluster
    energies = []
    fillings = []
    
    # Process each supercluster
    for sc_idx in range(num_superclusters):
        supercluster_idxs = all_superclusters[sc_idx]
        # Convert to k-space but keep the shape as expected by make_cluster_ham
        supercluster_k_full = convert_site_clusters_to_k(all_superclusters[sc_idx:sc_idx+1], L)
        supercluster_k = supercluster_k_full[0]
        
        print(f"\nProcessing supercluster {sc_idx + 1}/{num_superclusters}")
        print(f"  Sites: {supercluster_idxs}")
        print(f"  k-values/pi: {supercluster_k/np.pi}")
        
        # Create Hamiltonian for this supercluster
        # Pass the supercluster as it expects: shape (num_blocks, Nc)
        H, basis = make_cluster_ham(
            supercluster_k, 
            supercluster_idxs, 
            t, V, U, mu_0, 
            L, Nc, 
            int_sep_ratio, 
            v_sep_ratio, 
            ham_lib
        )
        
        # Get eigenvalues and eigenvectors
        H_matrix = H.toarray()
        eigvals, eigvecs = np.linalg.eigh(H_matrix)
        
        # Get ground state (lowest energy)
        gs_idx = 0
        gs_energy = eigvals[gs_idx]
        gs_state = eigvecs[:, gs_idx]
        
        # Calculate expectation values
        # For spinful fermions, we need to calculate number operators
        super_cluster_size = int(np.prod(supercluster_k.shape))
        
        # Number operator (sum of n_up + n_down for each site)
        from quspin.operators import hamiltonian
        n_list_up = [[1.0, i] for i in range(super_cluster_size)]
        n_list_down = [[1.0, i] for i in range(super_cluster_size)]
        static_n = [["n|", n_list_up], ["|n", n_list_down]]
        N_op = hamiltonian(static_n, [], basis=basis, dtype=np.float64)
        
        # Calculate expectations
        n_expect = np.real(np.conj(gs_state) @ N_op.toarray() @ gs_state)
        
        energies.append(gs_energy)
        fillings.append(n_expect)
        
        print(f"  Ground state energy: {gs_energy:.6f}")
        print(f"  Filling: {n_expect:.6f}")
    
    # Average over all superclusters
    total_energy = np.sum(energies) / num_superclusters
    total_filling = np.sum(fillings) / num_superclusters
    
    # Convert to per-site quantities
    energy_per_site = total_energy / (Nc * all_superclusters.shape[2])  # Nc * num_blocks_per_supercluster
    filling_per_site = total_filling / (Nc * all_superclusters.shape[2])
    
    print(f"\nFinal results:")
    print(f"  Energy per site: {energy_per_site:.6f}")
    print(f"  Filling per site: {filling_per_site:.6f}")
    
    return total_energy, total_filling


def plot_parameter_scan(
    U_values: np.ndarray,
    V_values: np.ndarray,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    t: float = 1.0
):
    """
    Create a parameter scan plot for the general cluster method.
    
    Parameters
    ----------
    U_values : np.ndarray
        Array of U values to scan
    V_values : np.ndarray
        Array of V values to scan
    L : int
        System size
    Nc : int
        Cluster size
    int_sep_ratio : Tuple[int, int]
        Interaction cluster separation ratio
    v_sep_ratio : Tuple[int, int]
        V-term separation ratio
    t : float
        Hopping parameter
    """
    
    energy_grid = np.zeros((len(U_values), len(V_values)))
    filling_grid = np.zeros((len(U_values), len(V_values)))
    
    for i, U in enumerate(U_values):
        for j, V in enumerate(V_values):
            mu_0 = U / 2  # Half-filling condition
            
            print(f"\n{'='*50}")
            print(f"Computing U={U:.2f}, V={V:.2f}, mu_0={mu_0:.2f}")
            
            energy, filling = run_general_cluster_method(
                U=U, mu_0=mu_0, V=V, t=t,
                L=L, Nc=Nc,
                int_sep_ratio=int_sep_ratio,
                v_sep_ratio=v_sep_ratio
            )
            
            energy_grid[i, j] = energy
            filling_grid[i, j] = filling
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy plot
    im1 = axes[0].imshow(energy_grid, aspect='auto', origin='lower', 
                         extent=[V_values[0], V_values[-1], U_values[0], U_values[-1]])
    axes[0].set_xlabel('V')
    axes[0].set_ylabel('U')
    axes[0].set_title(f'Energy (L={L}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio})')
    plt.colorbar(im1, ax=axes[0])
    
    # Filling plot
    im2 = axes[1].imshow(filling_grid, aspect='auto', origin='lower',
                         extent=[V_values[0], V_values[-1], U_values[0], U_values[-1]])
    axes[1].set_xlabel('V')
    axes[1].set_ylabel('U')
    axes[1].set_title(f'Filling (L={L}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio})')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    return fig, energy_grid, filling_grid


if __name__ == "__main__":
    # Test with the requested parameters
    print("Testing general cluster method with L=8, v_sep=(1,2), int_sep=(1,4)")
    
    # Single point test
    U = 2.0
    V = 0.5
    mu_0 = U / 2
    t = 1.0
    L = 8
    Nc = 2
    int_sep_ratio = (1, 4)
    v_sep_ratio = (1, 2)
    
    energy, filling = run_general_cluster_method(
        U=U, mu_0=mu_0, V=V, t=t,
        L=L, Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio
    )
    
    print(f"\nSingle point result:")
    print(f"  U={U}, V={V}, mu_0={mu_0}")
    print(f"  Energy: {energy:.6f}")
    print(f"  Filling: {filling:.6f}")
    
    # Parameter scan
    print("\n" + "="*60)
    print("Running parameter scan...")
    
    U_values = np.linspace(0, 4, 5)
    V_values = np.linspace(0, 2, 5)
    
    fig, energy_grid, filling_grid = plot_parameter_scan(
        U_values=U_values,
        V_values=V_values,
        L=L, Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio
    )
    
    plt.savefig('cluster_method_scan.png', dpi=150)
    plt.show()
    
    print("\nPlot saved as 'cluster_method_scan.png'")