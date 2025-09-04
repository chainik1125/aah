"""
Main script for running the general cluster method with arbitrary tilings.
This uses the FullSpectrumCustom class and can be directly compared with the original run_cluster_method.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from aah_code.cluster_model.full_spectrum_custom import (
    FullSpectrumCustom, 
    GeneralClusterParams,
    PhysicalParams,
    run_general_cluster_method
)
from aah_code.main import run_cluster_method


def compare_cluster_methods(
    U: float,
    V: float = 0.0,
    t: float = 1.0,
    system_size: int = 8,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    Nc: int = 2,
    use_simple_ham: bool = True
):
    """
    Compare the general cluster method with the original cluster method.
    
    For the original method, we'll use system_size and let it determine the clustering.
    For the general method, we'll use the specified ratios.
    """
    mu_0 = U / 2  # Half-filling
    
    print(f"\n{'='*60}")
    print(f"Comparing methods for U={U:.2f}, V={V:.2f}, t={t:.2f}")
    print(f"System size: {system_size}")
    print(f"General method: int_sep={int_sep_ratio}, v_sep={v_sep_ratio}, Nc={Nc}")
    
    # Run general cluster method
    print("\n1. Running general cluster method...")
    energy_general, filling_general = run_general_cluster_method(
        U=U,
        mu_0=mu_0,
        V=V,
        t=t,
        L=system_size,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        use_simple_ham=use_simple_ham
    )
    energy_per_site_general = energy_general / system_size
    filling_per_site_general = filling_general / system_size
    
    print(f"   Energy/site: {energy_per_site_general:.6f}")
    print(f"   Filling/site: {filling_per_site_general:.6f}")
    
    # Run original cluster method (if compatible system size)
    if system_size % 2 == 0:  # Original method needs even system size
        print("\n2. Running original cluster method...")
        try:
            energy_original, filling_original = run_cluster_method(
                U=U,
                mu_0=mu_0,
                V=V,
                t=t,
                system_size=system_size,
                ham_lib='quspin'
            )
            energy_per_site_original = energy_original / system_size
            filling_per_site_original = filling_original / system_size
            
            print(f"   Energy/site: {energy_per_site_original:.6f}")
            print(f"   Filling/site: {filling_per_site_original:.6f}")
            
            # Calculate differences
            energy_diff = abs(energy_per_site_general - energy_per_site_original)
            filling_diff = abs(filling_per_site_general - filling_per_site_original)
            
            print(f"\n3. Differences:")
            print(f"   |ΔE/site|: {energy_diff:.6e}")
            print(f"   |Δn/site|: {filling_diff:.6e}")
            
            return {
                'general': (energy_per_site_general, filling_per_site_general),
                'original': (energy_per_site_original, filling_per_site_original),
                'diff': (energy_diff, filling_diff)
            }
        except Exception as e:
            print(f"   Could not run original method: {e}")
            return {
                'general': (energy_per_site_general, filling_per_site_general),
                'original': None,
                'diff': None
            }
    else:
        print("\n2. Skipping original method (needs even system size)")
        return {
            'general': (energy_per_site_general, filling_per_site_general),
            'original': None,
            'diff': None
        }


def parameter_scan_comparison(
    U_values: np.ndarray,
    V_values: np.ndarray = np.array([0.0]),
    system_size: int = 8,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    Nc: int = 2
):
    """
    Scan parameters and compare both methods.
    """
    results = []
    
    for V in V_values:
        for U in U_values:
            result = compare_cluster_methods(
                U=U,
                V=V,
                system_size=system_size,
                int_sep_ratio=int_sep_ratio,
                v_sep_ratio=v_sep_ratio,
                Nc=Nc,
                use_simple_ham=(V == 0)  # Use simple if V=0
            )
            results.append((U, V, result))
    
    return results


def plot_comparison_results(results, save_path: Optional[str] = None):
    """
    Plot the comparison results.
    """
    # Extract data
    U_vals = []
    E_general = []
    n_general = []
    E_original = []
    n_original = []
    
    for U, V, result in results:
        if V == 0:  # Just plot V=0 for now
            U_vals.append(U)
            E_general.append(result['general'][0])
            n_general.append(result['general'][1])
            if result['original'] is not None:
                E_original.append(result['original'][0])
                n_original.append(result['original'][1])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy comparison
    axes[0, 0].plot(U_vals, E_general, 'o-', label='General method', markersize=8)
    if E_original:
        axes[0, 0].plot(U_vals[:len(E_original)], E_original, 's--', label='Original method', markersize=6)
    axes[0, 0].set_xlabel('U')
    axes[0, 0].set_ylabel('Energy per site')
    axes[0, 0].set_title('Ground State Energy Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Filling comparison
    axes[0, 1].plot(U_vals, n_general, 'o-', label='General method', markersize=8)
    if n_original:
        axes[0, 1].plot(U_vals[:len(n_original)], n_original, 's--', label='Original method', markersize=6)
    axes[0, 1].axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label='Half-filling')
    axes[0, 1].set_xlabel('U')
    axes[0, 1].set_ylabel('Filling per site')
    axes[0, 1].set_title('Average Filling Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy difference
    if E_original:
        E_diff = [abs(eg - eo) for eg, eo in zip(E_general[:len(E_original)], E_original)]
        axes[1, 0].semilogy(U_vals[:len(E_diff)], E_diff, 'ro-', markersize=6)
        axes[1, 0].set_xlabel('U')
        axes[1, 0].set_ylabel('|ΔE/site|')
        axes[1, 0].set_title('Energy Difference (General - Original)')
        axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Filling difference
    if n_original:
        n_diff = [abs(ng - no) for ng, no in zip(n_general[:len(n_original)], n_original)]
        axes[1, 1].semilogy(U_vals[:len(n_diff)], n_diff, 'bo-', markersize=6)
        axes[1, 1].set_xlabel('U')
        axes[1, 1].set_ylabel('|Δn/site|')
        axes[1, 1].set_title('Filling Difference (General - Original)')
        axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.suptitle('General vs Original Cluster Method Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    return fig


def main():
    """Main function to run comparisons."""
    
    print("=" * 70)
    print("GENERAL CLUSTER METHOD COMPARISON")
    print("=" * 70)
    
    # Test parameters
    system_size = 8  # Must be compatible with both methods
    U_values = np.linspace(0, 4, 9)  # More points for smoother curves
    V_values = np.array([0.0])  # Just V=0 for now
    
    # Cluster configuration for general method
    int_sep_ratio = (1, 4)  # L/4 separation 
    v_sep_ratio = (1, 2)     # L/2 separation (π modulation)
    Nc = 2                   # 2-site clusters
    
    print(f"\nParameters:")
    print(f"  System size: {system_size}")
    print(f"  U values: {U_values}")
    print(f"  V values: {V_values}")
    print(f"  General method clustering:")
    print(f"    - Cluster size (Nc): {Nc}")
    print(f"    - Interaction separation: {int_sep_ratio[0]}/{int_sep_ratio[1]} * L")
    print(f"    - V-term separation: {v_sep_ratio[0]}/{v_sep_ratio[1]} * L")
    
    # Run parameter scan
    print("\n" + "=" * 70)
    print("Running parameter scan...")
    print("=" * 70)
    
    results = parameter_scan_comparison(
        U_values=U_values,
        V_values=V_values,
        system_size=system_size,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        Nc=Nc
    )
    
    # Plot results
    print("\n" + "=" * 70)
    print("Plotting results...")
    fig = plot_comparison_results(results, save_path='cluster_method_comparison.png')
    plt.show()
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    max_energy_diff = 0
    max_filling_diff = 0
    
    for U, V, result in results:
        if result['diff'] is not None:
            max_energy_diff = max(max_energy_diff, result['diff'][0])
            max_filling_diff = max(max_filling_diff, result['diff'][1])
    
    if max_energy_diff > 0:
        print(f"Maximum energy difference: {max_energy_diff:.6e}")
        print(f"Maximum filling difference: {max_filling_diff:.6e}")
    else:
        print("No comparison with original method was possible.")
    
    print("\nComparison complete!")
    
    return results


if __name__ == "__main__":
    results = main()