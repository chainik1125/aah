#!/usr/bin/env python3
"""
Debug script to investigate discrepancies between Hubbard1D and QuickHubbard1D
when t≠0, V≠0, U=0.
"""

import numpy as np
import sys
import os
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah')

from aah_code.hamiltonian import Hubbard1D, QuickHubbard1D, SpectrumSolver
from aah_code.basis import LocalClusterBasis
from aah_code.global_params import StatesParams, HamiltonianParams

def test_case(t, V, U, description):
    """Test a specific parameter combination"""
    print(f"\n=== {description} ===")
    print(f"Parameters: t={t}, V={V}, U={U}")
    
    # Create StatesParams for the basis classes
    states_params = StatesParams(
        spin_states=2,
        cluster_chain_boundary_conditions='periodic',
        cluster_mps_boundary_conditions='finite'
    )
    
    # Setup basis classes for both models
    # For Hubbard1D: single cluster with 4 k-points
    cluster_k_points = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    basis_single = LocalClusterBasis(
        cluster_k_points=cluster_k_points,
        cluster_state_params=states_params
    )
    
    # For QuickHubbard1D: 2 clusters with 2 k-points each (blocking pi/2 apart)
    cluster_k_points_1 = np.array([0, np.pi])
    cluster_k_points_2 = np.array([np.pi/2, 3*np.pi/2])
    basis_1 = LocalClusterBasis(
        cluster_k_points=cluster_k_points_1,
        cluster_state_params=states_params
    )
    basis_2 = LocalClusterBasis(
        cluster_k_points=cluster_k_points_2,
        cluster_state_params=states_params
    )
    basis_classes = [basis_1, basis_2]
    
    # Model parameters for Hubbard1D
    model_params_1 = {
        't': t,
        'V': V, 
        'U': U,
        'mu': 0.0,
        'basis_class': basis_single
    }
    
    # Model parameters for QuickHubbard1D  
    model_params_2 = {
        't': t,
        'V': V,
        'U': U,
        'mu': 0.0,
        'L': 4,  # Total system size
        'L_cluster': 2,  # Individual cluster size
        'basis_classes': basis_classes
    }
    
    try:
        # Create and solve Hubbard1D
        ham1 = Hubbard1D(model_params_1)
        solver1 = SpectrumSolver(ham1, basis_single, solver='tenpy_ED')
        eigvals1, eigvecs1, n_ups1, n_downs1, n_tot1 = solver1.solve_spectrum()
        E1_ground = eigvals1[0]  # Ground state energy
        
        # Create and solve QuickHubbard1D  
        ham2 = QuickHubbard1D(model_params_2)
        solver2 = SpectrumSolver(ham2, basis_1, solver='tenpy_ED')  # Use first basis for solver
        eigvals2, eigvecs2, n_ups2, n_downs2, n_tot2 = solver2.solve_spectrum()
        E2_ground = eigvals2[0]  # Ground state energy
        
        print(f"Hubbard1D ground state energy: {E1_ground:.10f}")
        print(f"QuickHubbard1D ground state energy: {E2_ground:.10f}")
        print(f"Difference: {abs(E1_ground - E2_ground):.10e}")
        
        if abs(E1_ground - E2_ground) > 1e-10:
            print("⚠️  SIGNIFICANT DISCREPANCY DETECTED")
        else:
            print("✅ Energies agree within numerical precision")
            
    except Exception as e:
        print(f"❌ Error during calculation: {e}")
        
def main():
    print("Debugging Hamiltonian discrepancies between Hubbard1D and QuickHubbard1D")
    print("="*70)
    
    # Test cases as described by the user
    test_case(0.0, 0.0, 0.0, "Edge case: all parameters zero")
    test_case(1.0, 0.0, 0.0, "Edge case: only t non-zero") 
    test_case(0.0, 1.0, 0.0, "Edge case: only V non-zero")
    test_case(1.0, 1.0, 0.0, "Problematic case: t and V both non-zero")
    test_case(0.5, 2.0, 0.0, "Problematic case: different t,V values")

if __name__ == "__main__":
    main()