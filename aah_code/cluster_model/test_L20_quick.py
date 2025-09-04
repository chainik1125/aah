"""
Quick test for L=20 comparison - single point only.
"""

import numpy as np
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Reduce parallel overhead

from aah_code.cluster_model.full_spectrum_custom import run_general_cluster_method
from aah_code.hamiltonian import test_quick_mismatched, HamiltonianParams

# Silence most output
import sys
import logging
logging.getLogger().setLevel(logging.ERROR)

# Redirect QuSpin output
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def test_single_point():
    """Test just one U value for L=20."""
    
    L = 20
    U = 2.0
    mu_0 = U / 2
    V = 0.0
    t = 1.0
    Nc = 2
    
    print(f"Testing L={L}, U={U}, V={V}, mu_0={mu_0}")
    print("-" * 50)
    
    # Old method
    print("Running old method...", flush=True)
    physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
    
    with SuppressOutput():
        system_exp, _ = test_quick_mismatched(L, Nc, physical_params)
    
    E_old = system_exp[0] / L
    n_old = system_exp[1] / L
    print(f"  Old: E/site={E_old:.6f}, n/site={n_old:.6f}")
    
    # New method with L/10 separation
    print("Running new method (int_sep=1/10)...", flush=True)
    
    with SuppressOutput():
        E_new_total, n_new_total = run_general_cluster_method(
            U=U, mu_0=mu_0, V=V, t=t, L=L, Nc=Nc,
            int_sep_ratio=(1, 10),
            v_sep_ratio=(1, 2),
            use_simple_ham=True
        )
    
    E_new = E_new_total / L
    n_new = n_new_total / L
    print(f"  New: E/site={E_new:.6f}, n/site={n_new:.6f}")
    
    print(f"\nDifference: ΔE/site={abs(E_new - E_old):.6f}")

if __name__ == "__main__":
    test_single_point()