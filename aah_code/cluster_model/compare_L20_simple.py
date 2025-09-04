"""
Simple comparison of old and new cluster methods for L=20.
"""

import numpy as np
from aah_code.cluster_model.full_spectrum_custom import run_general_cluster_method
from aah_code.hamiltonian import test_quick_mismatched, HamiltonianParams

# Silence the logger to reduce output
import logging
logging.getLogger('aah').setLevel(logging.WARNING)

def quick_comparison():
    """Quick comparison for a few U values."""
    
    L = 20
    t = 1.0
    V = 0.0
    Nc = 2
    
    # Just test a few U values
    U_values = [0.5, 1.0, 2.0, 3.0]
    
    # New method parameters - try L/10 to match old method's clustering
    int_sep_ratio = (1, 10) 
    v_sep_ratio = (1, 2)
    
    print("=" * 70)
    print(f"QUICK COMPARISON: L={L}, V={V}")
    print("=" * 70)
    print(f"New method: Nc={Nc}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print()
    print("-" * 70)
    print(f"{'U':^6} | {'Old E/site':^12} {'Old n/site':^12} | {'New E/site':^12} {'New n/site':^12} | {'ΔE':^10}")
    print("-" * 70)
    
    for U in U_values:
        mu_0 = U / 2
        
        # Old method
        print(f"Running U={U:.1f}...", end='', flush=True)
        
        physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
        system_exp, _ = test_quick_mismatched(L, Nc, physical_params)
        E_old = system_exp[0] / L
        n_old = system_exp[1] / L
        
        print(" old done, ", end='', flush=True)
        
        # New method  
        E_new_total, n_new_total = run_general_cluster_method(
            U=U, mu_0=mu_0, V=V, t=t, L=L, Nc=Nc,
            int_sep_ratio=int_sep_ratio,
            v_sep_ratio=v_sep_ratio,
            use_simple_ham=True
        )
        E_new = E_new_total / L
        n_new = n_new_total / L
        
        print("new done")
        
        delta_E = abs(E_new - E_old)
        
        print(f"{U:6.1f} | {E_old:12.6f} {n_old:12.6f} | {E_new:12.6f} {n_new:12.6f} | {delta_E:10.6f}")
    
    print("-" * 70)

if __name__ == "__main__":
    quick_comparison()