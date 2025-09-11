#!/usr/bin/env python3
"""
Simple test script to verify the main code works
with minimal parameters before running full profiling.
"""

import numpy as np
from aah_code.cluster_model.plots import compare_int_seps_with_dmrg

# Minimal test parameters
L = 20  # Small system
Nc = 2  # Small cluster
U_values = np.array([0.0, 2.0])  # Just 2 values
V_values = np.array([0.5])  # Just 1 value
t = 1.0
v_sep_ratio = (1, 2)
int_sep_list = [(1, 2), (1, 4)]  # Just 2 configurations

print("=" * 60)
print("Running Simple Test")
print("=" * 60)
print(f"L={L}, Nc={Nc}")
print(f"U values: {U_values}")
print(f"V values: {V_values}")
print(f"v_sep={v_sep_ratio}")
print(f"int_sep_list={int_sep_list}")
print()

try:
    figures, results = compare_int_seps_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_list=int_sep_list,
        U_values=U_values,
        V_values=V_values,
        t=t,
        L=L,
        Nc=Nc,
        chi=32,
        solver_method='dense_ED',
        states_retained=4,
        show_plots=False,
        save_pickle=False,
        include_idmrg=True,
        include_finite_dmrg=False
    )
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"\nError during test: {e}")
    import traceback
    traceback.print_exc()