"""
Debug script to understand the spectra mismatch issues
"""

import numpy as np
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d
from quspin.operators import hamiltonian
from aah_code.cluster_model.t_tilde_test import alpha_terms_by_separation
from aah_code.hamiltonian import sparse_diagonalize

def debug_sparse_diag(H, name="Hamiltonian"):
    """Debug sparse diagonalization with different k values"""
    H_array = H.toarray()
    n_dim = H_array.shape[0]
    
    print(f"\n{name} dimension: {n_dim}")
    
    # Try different k values
    for k in [6, 10, n_dim-1, n_dim]:
        try:
            if k < n_dim:
                evals_sparse, _ = sparse_diagonalize(H, k=k)
                print(f"  k={k}: {len(evals_sparse)} eigenvalues returned")
                print(f"    First few: {evals_sparse[:min(3, len(evals_sparse))]}")
            else:
                print(f"  k={k}: Would exceed matrix dimension")
        except Exception as e:
            print(f"  k={k}: Failed with {e}")
    
    # Full diagonalization for comparison
    evals_full = np.linalg.eigvalsh(H_array)
    print(f"  Full spectrum: {n_dim} eigenvalues")
    print(f"    First few: {evals_full[:3]}")
    
    return evals_full

def analyze_spectra_mismatch():
    L = 6
    Nc = 3
    t = 1.0
    
    v_sep_ratio = (1, 6)
    matched_int_sep_ratio = (1, 3)
    mismatched_int_sep_ratio = (1, 6)
    
    # Generate clusters
    supercluster_matched = generate_clusters(L, Nc, matched_int_sep_ratio, v_sep_ratio)[0]
    supercluster_matched_k = convert_site_clusters_to_k(supercluster_matched, L)
    
    supercluster_unmatched = generate_clusters(L, Nc, mismatched_int_sep_ratio, v_sep_ratio)[0]
    supercluster_unmatched_k = convert_site_clusters_to_k(supercluster_unmatched, L)
    
    print("="*60)
    print("CLUSTER CONFIGURATIONS")
    print("="*60)
    print(f"Matched supercluster shape: {supercluster_matched.shape}")
    print(f"Matched supercluster: {supercluster_matched}")
    print(f"Unmatched supercluster shape: {supercluster_unmatched.shape}")
    print(f"Unmatched supercluster: {supercluster_unmatched}")
    
    # Get alpha terms for both spinless and spinful
    alpha_spinless_matched = alpha_terms_by_separation(supercluster_matched, supercluster_matched_k, t, spin="spinless")
    alpha_spinless_unmatched = alpha_terms_by_separation(supercluster_unmatched, supercluster_unmatched_k, t, spin="spinless")
    
    alpha_spinful_matched = alpha_terms_by_separation(supercluster_matched, supercluster_matched_k, t, spin="spinful")
    alpha_spinful_unmatched = alpha_terms_by_separation(supercluster_unmatched, supercluster_unmatched_k, t, spin="spinful")
    
    # Combine terms
    static_matched_spinless = []
    for s in sorted(alpha_spinless_matched.keys()):
        static_matched_spinless.extend(alpha_spinless_matched[s])
    
    static_unmatched_spinless = []
    for s in sorted(alpha_spinless_unmatched.keys()):
        static_unmatched_spinless.extend(alpha_spinless_unmatched[s])
    
    static_matched_spinful = []
    for s in sorted(alpha_spinful_matched.keys()):
        static_matched_spinful.extend(alpha_spinful_matched[s])
    
    static_unmatched_spinful = []
    for s in sorted(alpha_spinful_unmatched.keys()):
        static_unmatched_spinful.extend(alpha_spinful_unmatched[s])
    
    print("\n" + "="*60)
    print("1-PARTICLE SECTOR ANALYSIS")
    print("="*60)
    
    # 1-particle basis
    basis_1p = spinless_fermion_basis_1d(L=6, Nf=1)
    print(f"1-particle basis dimension: {basis_1p.Ns}")
    
    H_matched_1p = hamiltonian(static_matched_spinless, [], basis=basis_1p, dtype=np.complex128)
    H_unmatched_1p = hamiltonian(static_unmatched_spinless, [], basis=basis_1p, dtype=np.complex128)
    
    evals_matched_1p = np.linalg.eigvalsh(H_matched_1p.toarray())
    evals_unmatched_1p = np.linalg.eigvalsh(H_unmatched_1p.toarray())
    
    print(f"Matched 1p eigenvalues: {evals_matched_1p}")
    print(f"Unmatched 1p eigenvalues: {evals_unmatched_1p}")
    print(f"1-particle spectra match: {np.allclose(evals_matched_1p, evals_unmatched_1p)}")
    
    print("\n" + "="*60)
    print("SPINLESS MANY-BODY ANALYSIS")
    print("="*60)
    
    basis_spinless_full = spinless_fermion_basis_1d(L=6)
    print(f"Spinless full basis dimension: {basis_spinless_full.Ns}")
    
    H_matched_spinless = hamiltonian(static_matched_spinless, [], basis=basis_spinless_full, dtype=np.complex128)
    H_unmatched_spinless = hamiltonian(static_unmatched_spinless, [], basis=basis_spinless_full, dtype=np.complex128)
    
    print("\nMatched spinless:")
    evals_matched_spinless = debug_sparse_diag(H_matched_spinless, "Matched spinless")
    
    print("\nUnmatched spinless:")
    evals_unmatched_spinless = debug_sparse_diag(H_unmatched_spinless, "Unmatched spinless")
    
    print(f"\nSpinless spectra match: {np.allclose(evals_matched_spinless, evals_unmatched_spinless)}")
    
    print("\n" + "="*60)
    print("SPINFUL MANY-BODY ANALYSIS")
    print("="*60)
    
    basis_spinful_full = spinful_fermion_basis_1d(L=6)
    print(f"Spinful full basis dimension: {basis_spinful_full.Ns}")
    
    H_matched_spinful = hamiltonian(static_matched_spinful, [], basis=basis_spinful_full, dtype=np.complex128)
    H_unmatched_spinful = hamiltonian(static_unmatched_spinful, [], basis=basis_spinful_full, dtype=np.complex128)
    
    print("\nMatched spinful:")
    evals_matched_spinful = debug_sparse_diag(H_matched_spinful, "Matched spinful")
    
    print("\nUnmatched spinful:")
    evals_unmatched_spinful = debug_sparse_diag(H_unmatched_spinful, "Unmatched spinful")
    
    print(f"\nSpinful spectra match: {np.allclose(evals_matched_spinful, evals_unmatched_spinful)}")
    
    # Compare spinless vs spinful for same configuration
    print("\n" + "="*60)
    print("SPIN DEGENERACY ANALYSIS")
    print("="*60)
    
    # Compare ground states
    print(f"Matched - Spinless GS: {evals_matched_spinless[0]:.6f}, Spinful GS: {evals_matched_spinful[0]:.6f}")
    print(f"Unmatched - Spinless GS: {evals_unmatched_spinless[0]:.6f}, Spinful GS: {evals_unmatched_spinful[0]:.6f}")
    
    # Check for degeneracies in spinful case
    print("\nSpinful degeneracies (first 10 levels):")
    print(f"Matched: {evals_matched_spinful[:10]}")
    print(f"Unmatched: {evals_unmatched_spinful[:10]}")
    
    return {
        'matched_spinless': evals_matched_spinless,
        'unmatched_spinless': evals_unmatched_spinless,
        'matched_spinful': evals_matched_spinful,
        'unmatched_spinful': evals_unmatched_spinful
    }

if __name__ == "__main__":
    results = analyze_spectra_mismatch()