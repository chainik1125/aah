"""
Debug script to understand why spinful spectrum appears doubled
"""

import numpy as np
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d
from quspin.operators import hamiltonian
from aah_code.cluster_model.t_tilde_test import alpha_terms_by_separation

def analyze_spin_doubling():
    L = 6
    Nc = 3
    t = 1.0
    
    v_sep_ratio = (1, 6)
    int_sep_ratio = (1, 3)
    
    # Generate cluster
    supercluster = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)[0]
    supercluster_k = convert_site_clusters_to_k(supercluster, L)
    
    print("="*60)
    print("ANALYZING SPIN DOUBLING ISSUE")
    print("="*60)
    print(f"Supercluster shape: {supercluster.shape}")
    print(f"Supercluster: {supercluster}")
    print(f"Supercluster k/π: {supercluster_k/np.pi}")
    
    # Get terms
    alpha_spinless = alpha_terms_by_separation(supercluster, supercluster_k, t, spin="spinless")
    alpha_spinful = alpha_terms_by_separation(supercluster, supercluster_k, t, spin="spinful")
    
    print("\n" + "="*60)
    print("TERM ANALYSIS")
    print("="*60)
    
    # Analyze spinless terms
    print("\nSpinless terms by separation:")
    for s, terms in alpha_spinless.items():
        print(f"  s={s}:")
        for op, coupling_list in terms:
            print(f"    Operator '{op}': {len(coupling_list)} couplings")
            if len(coupling_list) > 0 and len(coupling_list) <= 3:
                print(f"      First few: {coupling_list[:3]}")
    
    # Analyze spinful terms
    print("\nSpinful terms by separation:")
    for s, terms in alpha_spinful.items():
        print(f"  s={s}:")
        for op, coupling_list in terms:
            print(f"    Operator '{op}': {len(coupling_list)} couplings")
            if len(coupling_list) > 0 and len(coupling_list) <= 3:
                print(f"      First few: {coupling_list[:3]}")
    
    # Check if spinful has exactly twice the terms
    static_spinless = []
    for s in sorted(alpha_spinless.keys()):
        static_spinless.extend(alpha_spinless[s])
    
    static_spinful = []
    for s in sorted(alpha_spinful.keys()):
        static_spinful.extend(alpha_spinful[s])
    
    print("\n" + "="*60)
    print("COUPLING COUNT COMPARISON")
    print("="*60)
    
    # Count total couplings
    n_couplings_spinless = sum(len(coupling_list) for _, coupling_list in static_spinless)
    n_couplings_spinful = sum(len(coupling_list) for _, coupling_list in static_spinful)
    
    print(f"Total spinless couplings: {n_couplings_spinless}")
    print(f"Total spinful couplings: {n_couplings_spinful}")
    print(f"Ratio spinful/spinless: {n_couplings_spinful/n_couplings_spinless if n_couplings_spinless > 0 else 'inf'}")
    
    # Build Hamiltonians
    print("\n" + "="*60)
    print("HAMILTONIAN COMPARISON")
    print("="*60)
    
    # Spinless
    basis_spinless = spinless_fermion_basis_1d(L=6)
    H_spinless = hamiltonian(static_spinless, [], basis=basis_spinless, dtype=np.complex128)
    H_spinless_array = H_spinless.toarray()
    
    # Spinful
    basis_spinful = spinful_fermion_basis_1d(L=6)
    H_spinful = hamiltonian(static_spinful, [], basis=basis_spinful, dtype=np.complex128)
    H_spinful_array = H_spinful.toarray()
    
    print(f"Spinless Hamiltonian shape: {H_spinless_array.shape}")
    print(f"Spinful Hamiltonian shape: {H_spinful_array.shape}")
    print(f"Dimension ratio: {H_spinful_array.shape[0]/H_spinless_array.shape[0]}")
    
    # Get spectra
    evals_spinless = np.linalg.eigvalsh(H_spinless_array)
    evals_spinful = np.linalg.eigvalsh(H_spinful_array)
    
    print("\n" + "="*60)
    print("SPECTRUM COMPARISON")
    print("="*60)
    
    print(f"Spinless spectrum (first 10): {evals_spinless[:10]}")
    print(f"Spinful spectrum (first 20): {evals_spinful[:20]}")
    
    # Check for doubling pattern
    print("\n" + "="*60)
    print("CHECKING FOR DOUBLING PATTERN")
    print("="*60)
    
    # For spinful with no magnetic field, we expect each spinless level
    # to appear with degeneracy based on particle number sectors
    
    # Compare unique values
    spinless_unique = np.unique(np.round(evals_spinless, 8))
    spinful_unique = np.unique(np.round(evals_spinful, 8))
    
    print(f"Unique spinless values: {len(spinless_unique)}")
    print(f"Unique spinful values: {len(spinful_unique)}")
    
    # Check if spinful contains doubled spinless values
    print("\nChecking if each spinless value appears doubled in spinful:")
    for val in spinless_unique[:5]:  # Check first 5 unique values
        count_in_spinful = np.sum(np.abs(evals_spinful - 2*val) < 1e-8)
        print(f"  Spinless value {val:.4f} -> 2*{val:.4f} appears {count_in_spinful} times in spinful")
    
    # Also check if spinful values are exactly double
    print("\nDirect comparison of values (spinful vs 2*spinless):")
    for i in range(min(10, len(evals_spinless))):
        print(f"  spinless[{i}] = {evals_spinless[i]:.6f}, 2*spinless[{i}] = {2*evals_spinless[i]:.6f}")
        if i < len(evals_spinful):
            print(f"  spinful[{i}] = {evals_spinful[i]:.6f}")
    
    return evals_spinless, evals_spinful

if __name__ == "__main__":
    evals_spinless, evals_spinful = analyze_spin_doubling()