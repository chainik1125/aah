"""
Three-way comparison: Old QSpin (mismatched) vs New General Setup vs DMRG
"""

import numpy as np
from typing import Tuple, Optional
from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations, test_quick_mismatched
from aah_code.hamiltonian import HamiltonianParams, StatesParams
from aah_code.main import run_dmrg_method


def compare_three_methods(
    U: float,
    V: float,
    t: float = 1.0,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    chi: int = 32,
    verbose: bool = True
) -> dict:
    """
    Compare three methods: old QSpin (mismatched), new general setup, and DMRG.
    
    Parameters
    ----------
    U : float
        Hubbard U interaction
    V : float
        V interaction strength
    t : float
        Hopping parameter (default: 1.0)
    L : int
        System size (default: 8)
    Nc : int
        Cluster size (default: 2)
    int_sep_ratio : tuple
        Interaction cluster separation ratio for new method
    v_sep_ratio : tuple
        V-term separation ratio for new method
    chi : int
        Bond dimension for DMRG (default: 32)
    verbose : bool
        Print results (default: True)
        
    Returns
    -------
    dict
        Dictionary containing results from all three methods
    """
    
    mu_0 = U / 2  # Half-filling
    
    results = {}
    
    # Method 1: Old QSpin (mismatched) implementation
    if verbose:
        print(f"\n=== Running Old QSpin (Mismatched) Method ===")
        print(f"Parameters: U={U}, V={V}, t={t}, mu_0={mu_0}, L={L}")
    
    # For mismatched, we need specific parameters
    physical_params_old = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
    system_expectations_old, cluster_expectations_old = test_quick_mismatched(
        lattice_points=L,
        cluster_size=Nc,
        physical_params=physical_params_old,
        ham_lib='quspin'  # Old method uses quspin
    )
    
    energy_old, filling_old, spin_old = system_expectations_old
    energy_old_per_site = energy_old / L
    filling_old_per_site = filling_old / L
    
    if verbose:
        print(f"Old QSpin: E/site = {energy_old_per_site:.6f}, n/site = {filling_old_per_site:.6f}")
    
    results['old_quspin'] = {
        'energy_total': energy_old,
        'energy_per_site': energy_old_per_site,
        'filling_total': filling_old,
        'filling_per_site': filling_old_per_site,
        'spin': spin_old
    }
    
    # Method 2: New General Setup
    if verbose:
        print(f"\n=== Running New General Setup Method ===")
        print(f"int_sep_ratio={int_sep_ratio}, v_sep_ratio={v_sep_ratio}")
    
    physical_params_new = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
    
    run_config = ClusterModelConfig(
        L=L,
        int_cluster_size=Nc,
        cluster_separation_ratio=int_sep_ratio,
        V_separation_ratio=v_sep_ratio,
        ham_lib='quspin',
        physical_params=physical_params_new,
        model_bc='periodic',
        int_cluster_bc='periodic',
        super_cluster_bc='periodic'
    )
    
    system_expectations_new, cluster_expectations_new = get_general_expectations(run_config)
    
    energy_new, filling_new, spin_new = system_expectations_new
    energy_new_per_site = energy_new / L
    filling_new_per_site = filling_new / L
    
    if verbose:
        print(f"New General: E/site = {energy_new_per_site:.6f}, n/site = {filling_new_per_site:.6f}")
    
    results['new_general'] = {
        'energy_total': energy_new,
        'energy_per_site': energy_new_per_site,
        'filling_total': filling_new,
        'filling_per_site': filling_new_per_site,
        'spin': spin_new
    }
    
    # Method 3: DMRG
    if verbose:
        print(f"\n=== Running DMRG Method ===")
        print(f"Bond dimension chi={chi}")
    
    energy_dmrg, filling_dmrg, psi_dmrg = run_dmrg_method(U, mu_0, V, t, L, chi)
    
    # DMRG returns per-site quantities for infinite systems
    energy_dmrg_per_site = energy_dmrg
    filling_dmrg_per_site = filling_dmrg
    
    if verbose:
        print(f"DMRG: E/site = {energy_dmrg_per_site:.6f}, n/site = {filling_dmrg_per_site:.6f}")
    
    results['dmrg'] = {
        'energy_per_site': energy_dmrg_per_site,
        'filling_per_site': filling_dmrg_per_site,
        'psi': psi_dmrg
    }
    
    # Calculate differences
    # Old vs New
    energy_diff_old_new = abs(energy_old_per_site - energy_new_per_site)
    filling_diff_old_new = abs(filling_old_per_site - filling_new_per_site)
    
    # Old vs DMRG
    energy_diff_old_dmrg = abs(energy_old_per_site - energy_dmrg_per_site)
    filling_diff_old_dmrg = abs(filling_old_per_site - filling_dmrg_per_site)
    
    # New vs DMRG
    energy_diff_new_dmrg = abs(energy_new_per_site - energy_dmrg_per_site)
    filling_diff_new_dmrg = abs(filling_new_per_site - filling_dmrg_per_site)
    
    if verbose:
        print(f"\n=== Comparison Results ===")
        print(f"Old vs New: |ΔE/site| = {energy_diff_old_new:.6f}, |Δn/site| = {filling_diff_old_new:.6f}")
        print(f"Old vs DMRG: |ΔE/site| = {energy_diff_old_dmrg:.6f}, |Δn/site| = {filling_diff_old_dmrg:.6f}")
        print(f"New vs DMRG: |ΔE/site| = {energy_diff_new_dmrg:.6f}, |Δn/site| = {filling_diff_new_dmrg:.6f}")
        
        # Percentage differences (relative to DMRG)
        if abs(energy_dmrg_per_site) > 1e-12:
            energy_pct_old = 100 * energy_diff_old_dmrg / abs(energy_dmrg_per_site)
            energy_pct_new = 100 * energy_diff_new_dmrg / abs(energy_dmrg_per_site)
            print(f"\nRelative to DMRG:")
            print(f"Old energy error: {energy_pct_old:.2f}%")
            print(f"New energy error: {energy_pct_new:.2f}%")
    
    results['comparisons'] = {
        'old_vs_new': {
            'energy_diff': energy_diff_old_new,
            'filling_diff': filling_diff_old_new
        },
        'old_vs_dmrg': {
            'energy_diff': energy_diff_old_dmrg,
            'filling_diff': filling_diff_old_dmrg
        },
        'new_vs_dmrg': {
            'energy_diff': energy_diff_new_dmrg,
            'filling_diff': filling_diff_new_dmrg
        }
    }
    
    return results


def scan_parameters_three_way(
    U_values: np.ndarray,
    V_values: np.ndarray,
    t: float = 1.0,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    chi: int = 32
) -> dict:
    """
    Scan over U and V values and compare all three methods.
    
    Returns
    -------
    dict
        Dictionary with grids of results for all parameter combinations
    """
    
    n_U = len(U_values)
    n_V = len(V_values)
    
    # Initialize storage grids
    results = {
        'old_quspin': {
            'energy': np.zeros((n_U, n_V)),
            'filling': np.zeros((n_U, n_V))
        },
        'new_general': {
            'energy': np.zeros((n_U, n_V)),
            'filling': np.zeros((n_U, n_V))
        },
        'dmrg': {
            'energy': np.zeros((n_U, n_V)),
            'filling': np.zeros((n_U, n_V))
        }
    }
    
    print(f"Scanning {n_U} U values × {n_V} V values")
    print(f"System: L={L}, Nc={Nc}, t={t}")
    print(f"New method: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print(f"DMRG: chi={chi}")
    
    for i, U in enumerate(U_values):
        for j, V in enumerate(V_values):
            print(f"\n--- U={U:.2f}, V={V:.2f} ---")
            
            comparison = compare_three_methods(
                U=U, V=V, t=t, L=L, Nc=Nc,
                int_sep_ratio=int_sep_ratio,
                v_sep_ratio=v_sep_ratio,
                chi=chi,
                verbose=False
            )
            
            # Store results
            results['old_quspin']['energy'][i, j] = comparison['old_quspin']['energy_per_site']
            results['old_quspin']['filling'][i, j] = comparison['old_quspin']['filling_per_site']
            
            results['new_general']['energy'][i, j] = comparison['new_general']['energy_per_site']
            results['new_general']['filling'][i, j] = comparison['new_general']['filling_per_site']
            
            results['dmrg']['energy'][i, j] = comparison['dmrg']['energy_per_site']
            results['dmrg']['filling'][i, j] = comparison['dmrg']['filling_per_site']
            
            # Print summary
            e_old = comparison['old_quspin']['energy_per_site']
            e_new = comparison['new_general']['energy_per_site']
            e_dmrg = comparison['dmrg']['energy_per_site']
            
            print(f"  E/site: Old={e_old:.4f}, New={e_new:.4f}, DMRG={e_dmrg:.4f}")
            print(f"  |ΔE|: Old-New={abs(e_old-e_new):.4f}, Old-DMRG={abs(e_old-e_dmrg):.4f}, New-DMRG={abs(e_new-e_dmrg):.4f}")
    
    # Calculate error grids
    results['errors'] = {
        'old_vs_new': {
            'energy': np.abs(results['old_quspin']['energy'] - results['new_general']['energy']),
            'filling': np.abs(results['old_quspin']['filling'] - results['new_general']['filling'])
        },
        'old_vs_dmrg': {
            'energy': np.abs(results['old_quspin']['energy'] - results['dmrg']['energy']),
            'filling': np.abs(results['old_quspin']['filling'] - results['dmrg']['filling'])
        },
        'new_vs_dmrg': {
            'energy': np.abs(results['new_general']['energy'] - results['dmrg']['energy']),
            'filling': np.abs(results['new_general']['filling'] - results['dmrg']['filling'])
        }
    }
    
    results['U_values'] = U_values
    results['V_values'] = V_values
    
    return results


if __name__ == "__main__":
    # Test with parameters from run_scripts_me.py
    U = 0.0
    V = 2.0
    t = 1.0
    L = 8
    Nc = 2
    v_sep_ratio = (1, 2)
    int_sep_ratio = (1, 4)
    
    print("=" * 60)
    print("Three-Way Comparison: Old QSpin vs New General vs DMRG")
    print("=" * 60)
    
    results = compare_three_methods(
        U=U, V=V, t=t, L=L, Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        chi=32,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Parameter scan example")
    print("=" * 60)
    
    # Small parameter scan
    U_values = np.array([0.0, 2.0, 4.0])
    V_values = np.array([0.0, 1.0, 2.0])
    
    scan_results = scan_parameters_three_way(
        U_values=U_values,
        V_values=V_values,
        t=t, L=L, Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        chi=32
    )
    
    print("\n" + "=" * 60)
    print("Summary of maximum errors:")
    print("=" * 60)
    
    for comparison in ['old_vs_new', 'old_vs_dmrg', 'new_vs_dmrg']:
        max_e_err = np.max(scan_results['errors'][comparison]['energy'])
        max_n_err = np.max(scan_results['errors'][comparison]['filling'])
        print(f"{comparison}: max |ΔE/site| = {max_e_err:.6f}, max |Δn/site| = {max_n_err:.6f}")