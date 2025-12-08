import numpy as np
import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from aah_code.cluster_model.plots import compare_U_values_with_dmrg

def test_plot():
    # Define parameters
    cluster_sizes = [2, 4]
    U_values = np.linspace(0, 4, 5)
    V_values = [0.5, 1.0]
    v_sep_ratio = (1, 2)
    int_sep_ratios = {2: (1, 2), 4: (1, 2)}
    
    # Create dummy data
    # DMRG energies: random but consistent
    dmrg_energies = np.random.rand(len(U_values), len(V_values)) * -2.0
    
    # Cluster energies: DMRG + some error that decreases with Nc
    cluster_energies = {}
    for Nc in cluster_sizes:
        cluster_energies[Nc] = {}
        for v_idx, V in enumerate(V_values):
            # Error decreases with Nc
            error = np.random.rand(len(U_values)) * 0.1 / Nc
            cluster_energies[Nc][V] = dmrg_energies[:, v_idx] + error

    results_payload = {
        'cluster_sizes': cluster_sizes,
        'U_values': U_values.tolist(),
        'V_values': V_values,
        'cluster_energies': {Nc: {V: cluster_energies[Nc][V].tolist() for V in V_values} for Nc in cluster_sizes},
        'dmrg_energies': dmrg_energies.tolist(),
        'int_sep_ratios': int_sep_ratios,
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': 1.0,
            'L': 10,
            'chi': 16,
            'solver_method': 'dense_ED',
            'states_retained': 4,
            'reference_scheme': 'idmrg',
        }
    }
    
    print("Running compare_U_values_with_dmrg with dummy data...")
    try:
        fig, _ = compare_U_values_with_dmrg(
            v_sep_ratio=v_sep_ratio,
            int_sep_ratios=int_sep_ratios,
            cluster_sizes=cluster_sizes,
            U_values=U_values,
            V_values=V_values,
            results=results_payload,
            show_plots=False, # Don't try to open browser
            save_html=True,
            save_data=True,
            include_idmrg=True,
            include_finite_dmrg=True,
            output_dir='test_output'
        )
        print("Success! Plot generated.")
    except Exception as e:
        print(f"Failed: {e}")
        raise

if __name__ == "__main__":
    test_plot()
