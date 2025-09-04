"""
Compare old and new cluster methods for L=20.
Uses test_quick_mismatched() for the old method.
"""

import numpy as np
import matplotlib.pyplot as plt
from aah_code.cluster_model.full_spectrum_custom import run_general_cluster_method
from aah_code.hamiltonian import test_quick_mismatched, HamiltonianParams


def compare_methods_L20():
    """
    Compare old and new methods for L=20 system.
    """
    
    # System parameters
    L = 20
    t = 1.0
    V = 0.0  # No staggered potential for clean comparison
    U_values = np.linspace(0, 4, 9)  # Test range of U values
    
    # New method parameters
    Nc = 2  # Cluster size
    int_sep_ratio = (1, 10)  # L/10 separation (trying to match old method)
    v_sep_ratio = (1, 2)     # π modulation
    
    print("=" * 70)
    print(f"COMPARISON: OLD vs NEW METHOD (L={L})")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  System size: {L}")
    print(f"  Hopping t: {t}")
    print(f"  Staggered potential V: {V}")
    print(f"  New method clustering: Nc={Nc}, int_sep={int_sep_ratio}, v_sep={v_sep_ratio}")
    print()
    
    # Storage for results
    energies_old = []
    fillings_old = []
    energies_new = []
    fillings_new = []
    
    print("-" * 70)
    print(f"{'U':^6} {'mu_0':^6} | {'Old E/site':^12} {'Old n/site':^12} | {'New E/site':^12} {'New n/site':^12} | {'ΔE':^10} {'Δn':^10}")
    print("-" * 70)
    
    for U in U_values:
        mu_0 = U / 2  # Half-filling
        
        # Old method using test_quick_mismatched
        physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
        system_expectations, cluster_expectations = test_quick_mismatched(
            lattice_points=L,
            cluster_size=Nc,
            physical_params=physical_params
        )
        total_energy_old, total_filling_old, total_spin_old = system_expectations
        
        # Per-site quantities for old method
        energy_per_site_old = total_energy_old / L
        filling_per_site_old = total_filling_old / L
        
        energies_old.append(energy_per_site_old)
        fillings_old.append(filling_per_site_old)
        
        # New method
        energy_new, filling_new = run_general_cluster_method(
            U=U,
            mu_0=mu_0,
            V=V,
            t=t,
            L=L,
            Nc=Nc,
            int_sep_ratio=int_sep_ratio,
            v_sep_ratio=v_sep_ratio,
            use_simple_ham=True  # V=0 so we can use simple version
        )
        
        # Per-site quantities for new method
        energy_per_site_new = energy_new / L
        filling_per_site_new = filling_new / L
        
        energies_new.append(energy_per_site_new)
        fillings_new.append(filling_per_site_new)
        
        # Calculate differences
        delta_E = abs(energy_per_site_new - energy_per_site_old)
        delta_n = abs(filling_per_site_new - filling_per_site_old)
        
        print(f"{U:6.2f} {mu_0:6.2f} | {energy_per_site_old:12.6f} {filling_per_site_old:12.6f} | "
              f"{energy_per_site_new:12.6f} {filling_per_site_new:12.6f} | {delta_E:10.6f} {delta_n:10.6f}")
    
    print("-" * 70)
    
    # Calculate average differences
    avg_energy_diff = np.mean([abs(e_new - e_old) for e_new, e_old in zip(energies_new, energies_old)])
    avg_filling_diff = np.mean([abs(n_new - n_old) for n_new, n_old in zip(fillings_new, fillings_old)])
    max_energy_diff = np.max([abs(e_new - e_old) for e_new, e_old in zip(energies_new, energies_old)])
    max_filling_diff = np.max([abs(n_new - n_old) for n_new, n_old in zip(fillings_new, fillings_old)])
    
    print(f"\nSummary Statistics:")
    print(f"  Average |ΔE/site|: {avg_energy_diff:.6f}")
    print(f"  Maximum |ΔE/site|: {max_energy_diff:.6f}")
    print(f"  Average |Δn/site|: {avg_filling_diff:.6f}")
    print(f"  Maximum |Δn/site|: {max_filling_diff:.6f}")
    
    return U_values, energies_old, fillings_old, energies_new, fillings_new


def plot_comparison(U_values, energies_old, fillings_old, energies_new, fillings_new):
    """
    Create comparison plots.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy comparison
    axes[0, 0].plot(U_values, energies_old, 'o-', label='Old method (test_quick_mismatched)', 
                    markersize=8, linewidth=2)
    axes[0, 0].plot(U_values, energies_new, 's--', label='New method (general cluster)', 
                    markersize=6, linewidth=2)
    axes[0, 0].set_xlabel('U')
    axes[0, 0].set_ylabel('Energy per site')
    axes[0, 0].set_title('Ground State Energy (L=20)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Filling comparison
    axes[0, 1].plot(U_values, fillings_old, 'o-', label='Old method', markersize=8, linewidth=2)
    axes[0, 1].plot(U_values, fillings_new, 's--', label='New method', markersize=6, linewidth=2)
    axes[0, 1].axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label='Half-filling')
    axes[0, 1].set_xlabel('U')
    axes[0, 1].set_ylabel('Filling per site')
    axes[0, 1].set_title('Average Filling (L=20)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy difference
    energy_diff = [abs(e_new - e_old) for e_new, e_old in zip(energies_new, energies_old)]
    axes[1, 0].semilogy(U_values, energy_diff, 'ro-', markersize=6)
    axes[1, 0].set_xlabel('U')
    axes[1, 0].set_ylabel('|ΔE/site|')
    axes[1, 0].set_title('Energy Difference (New - Old)')
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Filling difference
    filling_diff = [abs(n_new - n_old) for n_new, n_old in zip(fillings_new, fillings_old)]
    axes[1, 1].semilogy(U_values, filling_diff, 'bo-', markersize=6)
    axes[1, 1].set_xlabel('U')
    axes[1, 1].set_ylabel('|Δn/site|')
    axes[1, 1].set_title('Filling Difference (New - Old)')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Old vs New Cluster Method Comparison (L=20)', fontsize=14)
    plt.tight_layout()
    
    return fig


def test_different_clusterings():
    """
    Test different clustering configurations for the new method.
    """
    
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT CLUSTERINGS (L=20, U=2.0)")
    print("=" * 70)
    
    L = 20
    U = 2.0
    mu_0 = U / 2
    V = 0.0
    t = 1.0
    Nc = 2
    
    # Get reference from old method
    physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
    system_expectations, _ = test_quick_mismatched(L, Nc, physical_params)
    energy_old = system_expectations[0] / L
    filling_old = system_expectations[1] / L
    
    print(f"\nReference (old method): E/site={energy_old:.6f}, n/site={filling_old:.6f}")
    
    # Try different clusterings
    clusterings = [
        ((1, 10), (1, 2), "L/10 separation, π V-term"),
        ((1, 5), (1, 2), "L/5 separation, π V-term"),
        ((1, 4), (1, 2), "L/4 separation, π V-term"),
        ((1, 2), (1, 2), "π separation, π V-term"),
        ((1, 20), (1, 10), "L/20 separation, L/10 V-term"),
        ((1, 20), (1, 4), "L/20 separation, L/4 V-term"),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Configuration':^40} | {'E/site':^12} {'n/site':^12} | {'ΔE':^10} {'Δn':^10}")
    print("-" * 70)
    
    for int_sep, v_sep, desc in clusterings:
        try:
            energy_new, filling_new = run_general_cluster_method(
                U=U, mu_0=mu_0, V=V, t=t, L=L, Nc=Nc,
                int_sep_ratio=int_sep, v_sep_ratio=v_sep,
                use_simple_ham=True
            )
            energy_per_site = energy_new / L
            filling_per_site = filling_new / L
            delta_E = abs(energy_per_site - energy_old)
            delta_n = abs(filling_per_site - filling_old)
            
            print(f"{desc:40} | {energy_per_site:12.6f} {filling_per_site:12.6f} | "
                  f"{delta_E:10.6f} {delta_n:10.6f}")
        except Exception as e:
            print(f"{desc:40} | Failed: {str(e)[:30]}")
    
    print("-" * 70)


if __name__ == "__main__":
    # Main comparison
    print("\nRunning main comparison for L=20...")
    U_values, E_old, n_old, E_new, n_new = compare_methods_L20()
    
    # Create plots
    print("\nGenerating plots...")
    fig = plot_comparison(U_values, E_old, n_old, E_new, n_new)
    plt.savefig('comparison_L20.png', dpi=150, bbox_inches='tight')
    print("Plots saved to 'comparison_L20.png'")
    
    # Test different clusterings
    test_different_clusterings()
    
    plt.show()