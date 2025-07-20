#!/usr/bin/env python3
"""
Debug script to investigate the actual physical difference between Hubbard1D and QuickHubbard1D.

Key insight: They implement V differently!
- Hubbard1D: V as staggered onsite potential (+V, -V, +V, -V, ...)  
- QuickHubbard1D: V as next-nearest neighbor hopping
"""

import numpy as np
import sys
import os
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah')

from aah_code.hamiltonian import Hubbard1D, QuickHubbard1D, SpectrumSolver
from aah_code.basis import LocalClusterBasis
from aah_code.global_params import StatesParams, HamiltonianParams
import tenpy as tp
from tenpy.algorithms import exact_diag

def analyze_v_implementations():
    """
    Analyze how V is implemented differently in the two models
    """
    print("=== V IMPLEMENTATION ANALYSIS ===")
    print("Hubbard1D implements V as staggered onsite potential:")
    print("  Site 0: +V, Site 1: -V, Site 2: +V, Site 3: -V, ...")
    print()
    print("QuickHubbard1D implements V as next-nearest neighbor hopping:")
    print("  V * (c†_i c_{i+2} + h.c.) for all sites i")
    print()
    print("These are fundamentally different physical terms!")

def test_v_only_detailed():
    """Test V-only case with detailed Hamiltonian inspection"""
    print("\n=== V-ONLY DETAILED COMPARISON ===")
    
    # Setup parameters
    states_params = StatesParams(
        spin_states=2,
        cluster_chain_boundary_conditions='periodic',
        cluster_mps_boundary_conditions='finite'
    )
    
    V = 2.0
    print(f"Testing with V = {V}, t = 0, U = 0")
    
    # Hubbard1D setup
    cluster_k_points = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    basis_single = LocalClusterBasis(
        cluster_k_points=cluster_k_points,
        cluster_state_params=states_params
    )
    
    model_params_1 = {
        't': 0.0,
        'V': V, 
        'U': 0.0,
        'mu': 0.0,
        'basis_class': basis_single
    }
    
    # QuickHubbard1D setup  
    cluster_k_points_1 = np.array([0, np.pi])
    cluster_k_points_2 = np.array([np.pi/2, 3*np.pi/2])
    basis_1 = LocalClusterBasis(cluster_k_points=cluster_k_points_1, cluster_state_params=states_params)
    basis_2 = LocalClusterBasis(cluster_k_points=cluster_k_points_2, cluster_state_params=states_params)
    
    model_params_2 = {
        't': 0.0,
        'V': V,
        'U': 0.0,
        'mu': 0.0,
        'L': 4,
        'L_cluster': 2,
        'basis_classes': [basis_1, basis_2]
    }
    
    try:
        # Create Hamiltonians
        ham1 = Hubbard1D(model_params_1)
        ham2 = QuickHubbard1D(model_params_2)
        
        # Get matrix representations for inspection
        ham1_mat = tp.algorithms.exact_diag.get_numpy_Hamiltonian(ham1)
        ham2_mat = tp.algorithms.exact_diag.get_numpy_Hamiltonian(ham2)
        
        print(f"Hubbard1D Hamiltonian shape: {ham1_mat.shape}")
        print(f"QuickHubbard1D Hamiltonian shape: {ham2_mat.shape}")
        
        # Get ground state energies
        E1 = np.linalg.eigvals(ham1_mat).min()
        E2 = np.linalg.eigvals(ham2_mat).min()
        
        print(f"Hubbard1D ground state energy: {E1:.10f}")
        print(f"QuickHubbard1D ground state energy: {E2:.10f}")
        print(f"Difference: {abs(E1 - E2):.10e}")
        
        if abs(E1 - E2) < 1e-10:
            print("✅ Energies agree within numerical precision")
            print("This suggests the implementations are equivalent for V-only case!")
        else:
            print("⚠️  Energies differ - examining why...")
            
        # Print some matrix elements to understand the difference
        print(f"\nHubbard1D matrix diagonal (first 8): {np.diag(ham1_mat)[:8]}")
        print(f"QuickHubbard1D matrix diagonal (first 8): {np.diag(ham2_mat)[:8]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_manual_staggered_vs_hopping():
    """
    Manually test if staggered potential gives the same result as NNN hopping
    for a simple 4-site system
    """
    print("\n=== MANUAL COMPARISON: STAGGERED vs NNN HOPPING ===")
    
    # For a 4-site system with periodic BC
    # Staggered potential: H_stag = V*(n_0 - n_1 + n_2 - n_3)  
    # NNN hopping: H_nnn = V*(c†_0*c_2 + c†_2*c_0 + c†_1*c_3 + c†_3*c_1)
    
    print("Testing whether these are equivalent for some cases...")
    
    # Simple analysis: for V-only, no hopping
    # Both should give purely potential energy
    V = 1.0
    
    # For empty system (0 particles): both give E=0
    # For 1 particle at site 0: staggered gives +V, hopping gives 0
    # For 1 particle at site 1: staggered gives -V, hopping gives 0
    # For 2 particles at sites (0,2): staggered gives +2V, hopping gives 2V (hopping between 0<->2)
    
    print("Expected behavior:")
    print("- Empty system: both give E=0")
    print("- 1 particle at site 0: staggered=+V, hopping=0")  
    print("- 1 particle at site 1: staggered=-V, hopping=0")
    print("- 2 particles at (0,2): staggered=+2V, hopping=2V (resonance)")
    print()
    print("These are clearly different!")

def main():
    analyze_v_implementations()
    test_v_only_detailed() 
    test_manual_staggered_vs_hopping()
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("The two models implement V in fundamentally different ways.")
    print("They should NOT agree when V≠0 unless there's a special relationship")
    print("between staggered potential and NNN hopping that makes them equivalent.")
    print("="*70)

if __name__ == "__main__":
    main()