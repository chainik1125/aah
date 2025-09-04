"""
Simple test for the general cluster method.
"""

import numpy as np
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from aah_code.cluster_model.model_ham import make_cluster_ham

def test_single_supercluster():
    """Test with a single supercluster to debug the indexing issue."""
    
    # Parameters
    L = 8
    Nc = 2
    int_sep_ratio = (1, 4)
    v_sep_ratio = (1, 2)
    
    # Physical parameters
    U = 2.0
    V = 0.5
    mu_0 = U / 2
    t = 1.0
    
    # Generate clusters
    all_superclusters = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
    print(f"All superclusters shape: {all_superclusters.shape}")
    print(f"All superclusters:\n{all_superclusters}")
    
    # Take just the first supercluster
    first_sc_sites = all_superclusters[0]  # Shape: (num_blocks, Nc)
    print(f"\nFirst supercluster sites shape: {first_sc_sites.shape}")
    print(f"First supercluster sites:\n{first_sc_sites}")
    
    # Convert to k-values
    first_sc_k = convert_site_clusters_to_k(first_sc_sites[np.newaxis, :], L)[0]
    print(f"\nFirst supercluster k-values shape: {first_sc_k.shape}")
    print(f"First supercluster k-values/pi:\n{first_sc_k/np.pi}")
    
    # The key insight: make_cluster_ham expects the supercluster sites to be 
    # the actual k-point indices, not physical sites. The function internally
    # maps these to the flattened alpha basis indices.
    
    print("\nCreating Hamiltonian...")
    try:
        H, basis = make_cluster_ham(
            first_sc_k,
            first_sc_sites,  # This should be k-site indices
            t, V, U, mu_0,
            L, Nc,
            int_sep_ratio,
            v_sep_ratio,
            'quspin'
        )
        
        print("Hamiltonian created successfully!")
        print(f"Hamiltonian shape: {H.toarray().shape}")
        print(f"Basis size: {basis.Ns}")
        
        # Get ground state
        H_matrix = H.toarray()
        eigvals, eigvecs = np.linalg.eigh(H_matrix)
        print(f"\nLowest 5 eigenvalues: {eigvals[:5]}")
        
    except Exception as e:
        print(f"Error creating Hamiltonian: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_supercluster()