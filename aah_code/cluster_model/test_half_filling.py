"""
Test script to debug half-filling issue.
With V=0 and mu_0=U/2, we should always get half-filling (n=1.0 per site).
"""

import numpy as np
from aah_code.cluster_model.full_spectrum_custom import run_general_cluster_method
from aah_code.main import run_cluster_method


def test_half_filling_single_point(U=2.0, system_size=8):
    """Test a single point at half-filling."""
    
    V = 0.0
    mu_0 = U / 2  # This should guarantee half-filling
    t = 1.0
    
    print(f"\n{'='*60}")
    print(f"Testing half-filling with U={U}, V={V}, mu_0={mu_0}")
    print(f"System size: {system_size}")
    print("Expected: n/site = 1.0 (half-filling)")
    print('='*60)
    
    # Test general method with different clusterings
    clusterings = [
        ((1, 4), (1, 2), "L/4 separation, π modulation"),
        ((1, 2), (1, 2), "L/2 separation, π modulation"),
        ((1, 8), (1, 4), "L/8 separation, L/4 V-sep"),
    ]
    
    for int_sep, v_sep, desc in clusterings:
        print(f"\n--- General method: {desc} ---")
        print(f"    int_sep={int_sep}, v_sep={v_sep}")
        
        energy_gen, filling_gen = run_general_cluster_method(
            U=U,
            mu_0=mu_0,
            V=V,
            t=t,
            L=system_size,
            Nc=2,
            int_sep_ratio=int_sep,
            v_sep_ratio=v_sep,
            use_simple_ham=True
        )
        
        filling_per_site_gen = filling_gen / system_size
        energy_per_site_gen = energy_gen / system_size
        
        print(f"    Energy/site: {energy_per_site_gen:.6f}")
        print(f"    Filling/site: {filling_per_site_gen:.6f}")
        print(f"    Deviation from half-filling: {abs(filling_per_site_gen - 1.0):.6e}")
    
    # Test original method
    print(f"\n--- Original cluster method ---")
    energy_orig, filling_orig = run_cluster_method(
        U=U,
        mu_0=mu_0,
        V=V,
        t=t,
        system_size=system_size,
        ham_lib='quspin'
    )
    
    filling_per_site_orig = filling_orig / system_size
    energy_per_site_orig = energy_orig / system_size
    
    print(f"    Energy/site: {energy_per_site_orig:.6f}")
    print(f"    Filling/site: {filling_per_site_orig:.6f}")
    print(f"    Deviation from half-filling: {abs(filling_per_site_orig - 1.0):.6e}")


def test_half_filling_scan():
    """Scan over U values and check half-filling."""
    
    print("\n" + "="*70)
    print("HALF-FILLING TEST: V=0, mu_0=U/2")
    print("="*70)
    
    U_values = [0.5, 1.0, 2.0, 3.0, 4.0]
    system_size = 8
    V = 0.0
    t = 1.0
    
    # Use standard clustering for general method
    int_sep_ratio = (1, 4)
    v_sep_ratio = (1, 2)
    
    print(f"\nParameters:")
    print(f"  System size: {system_size}")
    print(f"  V = {V} (no staggered potential)")
    print(f"  t = {t}")
    print(f"  General method: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    
    print("\n" + "-"*70)
    print(f"{'U':>6} {'mu_0':>6} | {'Gen E/site':>12} {'Gen n/site':>12} | {'Orig E/site':>12} {'Orig n/site':>12}")
    print("-"*70)
    
    for U in U_values:
        mu_0 = U / 2
        
        # General method
        energy_gen, filling_gen = run_general_cluster_method(
            U=U,
            mu_0=mu_0,
            V=V,
            t=t,
            L=system_size,
            Nc=2,
            int_sep_ratio=int_sep_ratio,
            v_sep_ratio=v_sep_ratio,
            use_simple_ham=True
        )
        
        filling_per_site_gen = filling_gen / system_size
        energy_per_site_gen = energy_gen / system_size
        
        # Original method
        energy_orig, filling_orig = run_cluster_method(
            U=U,
            mu_0=mu_0,
            V=V,
            t=t,
            system_size=system_size,
            ham_lib='quspin'
        )
        
        filling_per_site_orig = filling_orig / system_size
        energy_per_site_orig = energy_orig / system_size
        
        print(f"{U:6.1f} {mu_0:6.1f} | {energy_per_site_gen:12.6f} {filling_per_site_gen:12.6f} | {energy_per_site_orig:12.6f} {filling_per_site_orig:12.6f}")
    
    print("-"*70)
    print("\nNote: All filling values should be 1.0 (half-filling) when V=0 and mu_0=U/2")


def debug_ground_state_selection():
    """Debug how ground states are being selected."""
    
    print("\n" + "="*70)
    print("DEBUGGING GROUND STATE SELECTION")
    print("="*70)
    
    from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
    from aah_code.cluster_model.run_scripts_simple import make_simple_cluster_ham
    from quspin.operators import hamiltonian
    
    # Simple test case
    U = 2.0
    V = 0.0
    mu_0 = U / 2
    t = 1.0
    L = 8
    Nc = 2
    int_sep_ratio = (1, 4)
    v_sep_ratio = (1, 2)
    
    print(f"\nTest case: U={U}, V={V}, mu_0={mu_0}, t={t}")
    print(f"L={L}, Nc={Nc}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    
    # Generate clusters
    all_superclusters = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
    
    # Look at first supercluster
    sc_idx = 0
    supercluster_idxs = all_superclusters[sc_idx]
    supercluster_k = convert_site_clusters_to_k(
        all_superclusters[sc_idx:sc_idx+1], L
    )[0]
    
    print(f"\nSupercluster {sc_idx}:")
    print(f"  Sites: {supercluster_idxs}")
    print(f"  k-values/pi: {supercluster_k/np.pi}")
    
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
    
    # Calculate filling for lowest energy states
    super_cluster_size = int(np.prod(supercluster_k.shape))
    n_list_up = [[1.0, i] for i in range(super_cluster_size)]
    n_list_down = [[1.0, i] for i in range(super_cluster_size)]
    static_n = [["n|", n_list_up], ["|n", n_list_down]]
    N_op = hamiltonian(static_n, [], basis=basis, dtype=np.float64)
    
    print(f"\nLowest 10 energy states:")
    print(f"{'Index':>6} {'Energy':>12} {'Filling':>12}")
    print("-"*30)
    
    for i in range(min(10, len(eigvals))):
        state = eigvecs[:, i]
        n_expect = np.real(np.conj(state) @ N_op.toarray() @ state)
        print(f"{i:6d} {eigvals[i]:12.6f} {n_expect:12.6f}")
    
    print(f"\nGround state (index 0):")
    print(f"  Energy: {eigvals[0]:.6f}")
    gs_state = eigvecs[:, 0]
    gs_filling = np.real(np.conj(gs_state) @ N_op.toarray() @ gs_state)
    print(f"  Total filling: {gs_filling:.6f}")
    print(f"  Filling per site: {gs_filling/super_cluster_size:.6f}")
    
    # Check if there are degenerate states
    energy_tol = 1e-10
    degenerate_indices = np.where(np.abs(eigvals - eigvals[0]) < energy_tol)[0]
    if len(degenerate_indices) > 1:
        print(f"\nWARNING: Found {len(degenerate_indices)} degenerate ground states!")
        print("Degenerate state fillings:")
        for idx in degenerate_indices:
            state = eigvecs[:, idx]
            n_expect = np.real(np.conj(state) @ N_op.toarray() @ state)
            print(f"  State {idx}: n={n_expect:.6f}")


if __name__ == "__main__":
    # Run tests
    test_half_filling_scan()
    test_half_filling_single_point(U=2.0, system_size=8)
    debug_ground_state_selection()