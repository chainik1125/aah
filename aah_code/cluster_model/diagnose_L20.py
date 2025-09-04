"""
Diagnose issues with L=20 clustering.
"""

import numpy as np
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k

def diagnose_clustering():
    """Check what happens with L=20 clustering."""
    
    L = 20
    Nc = 2
    
    print("=" * 60)
    print("DIAGNOSING L=20 CLUSTERING")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        ((1, 10), (1, 2), "L/10 separation, π V-term"),
        ((1, 20), (1, 10), "L/20 separation, L/10 V-term"),
        ((1, 5), (1, 2), "L/5 separation, π V-term"),
        ((1, 4), (1, 2), "L/4 separation, π V-term"),
        ((1, 2), (1, 2), "π separation, π V-term"),
    ]
    
    for int_sep, v_sep, desc in configs:
        print(f"\n{desc}")
        print(f"  int_sep={int_sep}, v_sep={v_sep}")
        
        try:
            # Generate clusters
            all_superclusters = generate_clusters(L, Nc, int_sep, v_sep)
            
            print(f"  ✓ Clusters generated successfully")
            print(f"    Shape: {all_superclusters.shape}")
            print(f"    Num superclusters: {all_superclusters.shape[0]}")
            print(f"    Sites per supercluster: {np.prod(all_superclusters[0].shape)}")
            
            # Check first supercluster
            if all_superclusters.shape[0] > 0:
                first_sc = all_superclusters[0]
                print(f"    First supercluster shape: {first_sc.shape}")
                print(f"    First supercluster sites: {first_sc}")
                
                # Convert to k-space
                k_vals = convert_site_clusters_to_k(all_superclusters[0:1], L)[0]
                print(f"    k-values shape: {k_vals.shape}")
                print(f"    k-values/π: {k_vals/np.pi}")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Check what the old method expects
    print("\n" + "=" * 60)
    print("OLD METHOD EXPECTATION")
    print("=" * 60)
    
    print(f"For L={L}, cluster_size={Nc}:")
    print(f"  cluster_k_generator = L // 2 = {L // 2} (π separation)")
    print(f"  This creates {L // Nc} = {L // Nc} clusters of size {Nc}")


def test_hamiltonian_size():
    """Test the Hamiltonian construction for different supercluster sizes."""
    
    print("\n" + "=" * 60)
    print("HAMILTONIAN SIZE TEST")
    print("=" * 60)
    
    from aah_code.cluster_model.run_scripts_simple import make_simple_cluster_ham
    
    L = 20
    Nc = 2
    int_sep_ratio = (1, 10)
    v_sep_ratio = (1, 2)
    
    # Simple test parameters
    t = 1.0
    V = 0.0
    U = 2.0
    mu_0 = U / 2
    
    print(f"Testing Hamiltonian construction for L={L}, Nc={Nc}")
    print(f"int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    
    all_superclusters = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
    print(f"\nGenerated {all_superclusters.shape[0]} superclusters")
    
    # Test first supercluster
    sc_idx = 0
    supercluster_idxs = all_superclusters[sc_idx]
    supercluster_k = convert_site_clusters_to_k(all_superclusters[sc_idx:sc_idx+1], L)[0]
    
    print(f"\nSupercluster {sc_idx}:")
    print(f"  Shape: {supercluster_idxs.shape}")
    print(f"  Total sites: {np.prod(supercluster_idxs.shape)}")
    print(f"  Sites: {supercluster_idxs}")
    
    print("\nBuilding Hamiltonian...")
    try:
        H, basis = make_simple_cluster_ham(
            supercluster_k,
            supercluster_idxs,
            t, V, U, mu_0,
            L, Nc,
            int_sep_ratio,
            v_sep_ratio
        )
        
        print(f"  ✓ Hamiltonian built successfully")
        print(f"    Basis dimension: {basis.Ns}")
        print(f"    Hamiltonian shape: {H.toarray().shape}")
        
        # Check sparsity
        H_matrix = H.toarray()
        nonzero = np.count_nonzero(H_matrix)
        total = H_matrix.size
        sparsity = 1 - (nonzero / total)
        print(f"    Sparsity: {sparsity:.2%}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_clustering()
    test_hamiltonian_size()