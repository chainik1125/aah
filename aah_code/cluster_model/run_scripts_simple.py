"""
Simplified run script that works around the indexing issue.
For now, we'll create a minimal working example.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from contextlib import redirect_stdout
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from aah_code.cluster_model.t_tilde import alpha_terms_by_separation
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d


def make_simple_cluster_ham(
    supercluster_k,
    supercluster_idxs,  # Not used for V terms in this simple version
    t, V, U, mu_0,
    L, Nc,
    int_sep_ratio,
    v_sep_ratio
):
    """
    Simplified version that skips the V terms for now to get a working example.
    """
    static = []
    super_cluster_size = int(np.prod(supercluster_k.shape))
    basis = spinful_fermion_basis_1d(super_cluster_size)
    
    # Add t-tilde terms (these work correctly)
    # We need to create a remapped version for the local indices
    local_sites = np.arange(super_cluster_size).reshape(supercluster_idxs.shape)
    t_tilde_terms = alpha_terms_by_separation(local_sites, supercluster_k, t, spin='spinful')
    for sep, terms in t_tilde_terms.items():
        static.extend(terms)
    
    # Add diagonal U 
    if U != 0:
        U_list = [[U, i, i] for i in range(super_cluster_size)]
        static.append(["n|n", U_list])
    
    # Add onsite mu_0
    if mu_0 != 0:
        mu_0_list = [[-mu_0, i] for i in range(super_cluster_size)]
        static.append(["n|", mu_0_list])
        static.append(["|n", mu_0_list])
    
    # Skip V terms for now since they have indexing issues
    if V != 0:
        print(f"Warning: V terms skipped in this simple version")
    
    # Suppress quspin's successful check messages but keep error checking
    with redirect_stdout(io.StringIO()):
        H = hamiltonian(static, [], basis=basis, dtype=np.complex64)
    return H, basis


def run_simple_test():
    """Run a simple test case."""
    
    # Parameters
    L = 8
    Nc = 2
    int_sep_ratio = (1, 4)
    v_sep_ratio = (1, 2)
    
    # Physical parameters
    U_values = np.linspace(0, 4, 5)
    V = 0.0  # Set to 0 for now
    t = 1.0
    
    results = []
    
    print(f"Testing with L={L}, Nc={Nc}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print(f"V=0 (skipped for now), t={t}")
    print("-" * 50)
    
    # Generate clusters once
    all_superclusters = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
    num_superclusters = all_superclusters.shape[0]
    print(f"Number of superclusters: {num_superclusters}")
    print(f"Supercluster shape: {all_superclusters.shape}")
    
    for U in U_values:
        mu_0 = U / 2  # Half-filling
        
        total_energy = 0
        total_filling = 0
        
        # Process each supercluster
        for sc_idx in range(num_superclusters):
            supercluster_idxs = all_superclusters[sc_idx]
            supercluster_k = convert_site_clusters_to_k(
                all_superclusters[sc_idx:sc_idx+1], L
            )[0]
            
            # Create Hamiltonian
            H, basis = make_simple_cluster_ham(
                supercluster_k,
                supercluster_idxs,
                t, V, U, mu_0,
                L, Nc,
                int_sep_ratio,
                v_sep_ratio
            )
            
            # Diagonalize
            H_matrix = H.toarray()
            eigvals, eigvecs = np.linalg.eigh(H_matrix)
            
            # Get ground state
            gs_energy = eigvals[0]
            gs_state = eigvecs[:, 0]
            
            # Calculate filling
            super_cluster_size = int(np.prod(supercluster_k.shape))
            n_list_up = [[1.0, i] for i in range(super_cluster_size)]
            n_list_down = [[1.0, i] for i in range(super_cluster_size)]
            static_n = [["n|", n_list_up], ["|n", n_list_down]]
            N_op = hamiltonian(static_n, [], basis=basis, dtype=np.float64)
            n_expect = np.real(np.conj(gs_state) @ N_op.toarray() @ gs_state)
            
            total_energy += gs_energy
            total_filling += n_expect
        
        # Average over superclusters
        avg_energy = total_energy / num_superclusters
        avg_filling = total_filling / num_superclusters
        
        # Per-site quantities
        sites_per_sc = np.prod(all_superclusters[0].shape)
        energy_per_site = avg_energy / sites_per_sc
        filling_per_site = avg_filling / sites_per_sc
        
        results.append((U, energy_per_site, filling_per_site))
        print(f"U={U:.2f}: E/site={energy_per_site:.6f}, n/site={filling_per_site:.6f}")
    
    return results


def plot_results(results):
    """Plot the results."""
    U_vals = [r[0] for r in results]
    E_vals = [r[1] for r in results]
    n_vals = [r[2] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(U_vals, E_vals, 'o-')
    ax1.set_xlabel('U')
    ax1.set_ylabel('Energy per site')
    ax1.set_title('Ground State Energy')
    ax1.grid(True)
    
    ax2.plot(U_vals, n_vals, 'o-')
    ax2.set_xlabel('U')
    ax2.set_ylabel('Filling per site')
    ax2.set_title('Average Filling')
    ax2.grid(True)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Half-filling')
    ax2.legend()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Running simplified cluster method (V=0)...")
    results = run_simple_test()
    
    print("\nPlotting results...")
    fig = plot_results(results)
    plt.savefig('simple_cluster_results.png', dpi=150)
    print("Plot saved as 'simple_cluster_results.png'")
    plt.show()