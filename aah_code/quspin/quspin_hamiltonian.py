"""
Test for 1D spinful chain with NN and NNN hoppings using QuSpin.
Verifies that the many-body spectrum reduces to a sum of single-particle energies.
"""

import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis import spinful_fermion_basis_1d
from itertools import combinations


def create_single_particle_hamiltonian(L, t1, t2):
    """
    Create single-particle Hamiltonian matrix for 1D chain with NN and NNN hoppings.
    
    Parameters:
    -----------
    L : int
        Number of sites
    t1 : float
        Nearest neighbor hopping amplitude
    t2 : float
        Next-nearest neighbor hopping amplitude
        
    Returns:
    --------
    H_sp : ndarray
        Single-particle Hamiltonian matrix (L x L)
    """
    H_sp = np.zeros((L, L), dtype=complex)
    
    # Nearest neighbor hopping
    for i in range(L-1):
        H_sp[i, i+1] = -t1
        H_sp[i+1, i] = -t1
    
    # Next-nearest neighbor hopping
    for i in range(L-2):
        H_sp[i, i+2] = -t2
        H_sp[i+2, i] = -t2
    
    # Periodic boundary conditions
    H_sp[0, L-1] = -t1
    H_sp[L-1, 0] = -t1
    if L > 2:
        H_sp[0, L-2] = -t2
        H_sp[L-2, 0] = -t2
        H_sp[1, L-1] = -t2
        H_sp[L-1, 1] = -t2
    
    return H_sp


def create_quspin_hamiltonian(L, t1, t2):
    """
    Create many-body Hamiltonian using QuSpin for 1D spinful chain with NN and NNN hoppings.
    
    Parameters:
    -----------
    L : int
        Number of sites
    t1 : float
        Nearest neighbor hopping amplitude
    t2 : float
        Next-nearest neighbor hopping amplitude
        
    Returns:
    --------
    H : hamiltonian object
        QuSpin Hamiltonian
    basis : basis object
        QuSpin spinful fermion basis
    """
    # Create basis
    basis = spinful_fermion_basis_1d(L)
    
    # Create hopping terms - include both directions for hermiticity
    # For spinful fermions, hopping preserves spin
    hop_list = []
    
    # Nearest neighbor hopping (both spins)
    for i in range(L):
        j = (i+1) % L
        # Spin up hopping: site indices are 2*position + spin
        i_up, j_up = 2*i + 0, 2*j + 0
        hop_list.append([-t1, i_up, j_up])  # c_i,up^dag c_j,up
        hop_list.append([-t1, j_up, i_up])  # c_j,up^dag c_i,up
        # Spin down hopping
        i_down, j_down = 2*i + 1, 2*j + 1
        hop_list.append([-t1, i_down, j_down])  # c_i,down^dag c_j,down
        hop_list.append([-t1, j_down, i_down])  # c_j,down^dag c_i,down
    
    # Next-nearest neighbor hopping (both spins)
    for i in range(L):
        j = (i+2) % L
        # Spin up hopping
        i_up, j_up = 2*i + 0, 2*j + 0
        hop_list.append([-t2, i_up, j_up])  # c_i,up^dag c_j,up
        hop_list.append([-t2, j_up, i_up])  # c_j,up^dag c_i,up
        # Spin down hopping
        i_down, j_down = 2*i + 1, 2*j + 1
        hop_list.append([-t2, i_down, j_down])  # c_i,down^dag c_j,down
        hop_list.append([-t2, j_down, i_down])  # c_j,down^dag c_i,down
    
    # Create static Hamiltonian
    static = [["+-|", hop_list]]
    dynamic = []
    
    try:
        # Try to create Hamiltonian, and automatically answer 'y' to hermiticity check
        import sys
        from io import StringIO
        old_stdin = sys.stdin
        sys.stdin = StringIO('y\n')
        
        H = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128)
        
        sys.stdin = old_stdin
        
    except:
        # If that fails, create it without the check
        H = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128)
    
    return H, basis


def get_many_body_energies_for_n_particles(L, t1, t2, n_particles, n_states=None):
    """
    Get many-body energies for a specific number of particles in spinful system.
    
    Parameters:
    -----------
    L : int
        Number of sites
    t1 : float
        Nearest neighbor hopping amplitude  
    t2 : float
        Next-nearest neighbor hopping amplitude
    n_particles : int
        Number of particles (spin-up + spin-down)
    n_states : int or None
        Number of states to compute (None for all)
        
    Returns:
    --------
    energies : ndarray
        Many-body energies
    """
    # Get basis for specific particle number
    # For spinful fermions, we need to specify (Nf_up, Nf_down)
    # For simplicity, distribute particles evenly between spins when possible
    n_up = n_particles // 2
    n_down = n_particles - n_up
    basis_n = spinful_fermion_basis_1d(L, Nf=(n_up, n_down))
    
    # Create hopping list for spinful fermions
    hop_list = []
    
    # Nearest neighbor hopping (both spins)
    for i in range(L):
        j = (i+1) % L
        # Spin up hopping: site indices are 2*position + spin
        i_up, j_up = 2*i + 0, 2*j + 0
        hop_list.append([-t1, i_up, j_up])
        hop_list.append([-t1, j_up, i_up])
        # Spin down hopping
        i_down, j_down = 2*i + 1, 2*j + 1
        hop_list.append([-t1, i_down, j_down])
        hop_list.append([-t1, j_down, i_down])
    
    # Next-nearest neighbor hopping (both spins)
    for i in range(L):
        j = (i+2) % L
        # Spin up hopping
        i_up, j_up = 2*i + 0, 2*j + 0
        hop_list.append([-t2, i_up, j_up])
        hop_list.append([-t2, j_up, i_up])
        # Spin down hopping
        i_down, j_down = 2*i + 1, 2*j + 1
        hop_list.append([-t2, i_down, j_down])
        hop_list.append([-t2, j_down, i_down])
    
    static = [["+-|", hop_list]]
    
    try:
        import sys
        from io import StringIO
        old_stdin = sys.stdin
        sys.stdin = StringIO('y\n')
        
        H_n = hamiltonian(static, [], basis=basis_n, dtype=np.complex128)
        
        sys.stdin = old_stdin
        
    except:
        H_n = hamiltonian(static, [], basis=basis_n, dtype=np.complex128)
    
    # Diagonalize
    if n_states is None:
        energies = H_n.eigvalsh()
    else:
        energies = H_n.eigsh(k=n_states, which='SA', return_eigenvectors=False)
    
    return np.sort(energies)


def get_single_particle_combinations(single_particle_energies, n_particles):
    """
    Get all possible combinations of single-particle energies for n particles.
    
    Parameters:
    -----------
    single_particle_energies : ndarray
        Single-particle energy eigenvalues
    n_particles : int
        Number of particles
        
    Returns:
    --------
    combination_energies : ndarray
        Sum of energies for all possible combinations
    """
    L = len(single_particle_energies)
    combination_energies = []
    
    # Generate all combinations of n_particles indices
    for indices in combinations(range(L), n_particles):
        energy_sum = sum(single_particle_energies[i] for i in indices)
        combination_energies.append(energy_sum)
    
    return np.sort(combination_energies)


def test_spectrum_comparison(L=6, t1=1.0, t2=0.5, n_particles=2):
    """
    Test that many-body spectrum matches sum of single-particle energies for spinful system.
    
    Parameters:
    -----------
    L : int
        Number of sites
    t1 : float
        Nearest neighbor hopping amplitude
    t2 : float
        Next-nearest neighbor hopping amplitude
    n_particles : int
        Number of particles to test (total spin-up + spin-down)
    """
    print(f"Testing 1D chain with L={L}, t1={t1}, t2={t2}, N={n_particles} particles")
    print("=" * 60)
    
    # 1. Single-particle analysis
    H_sp = create_single_particle_hamiltonian(L, t1, t2)
    sp_energies = np.linalg.eigvals(H_sp)
    sp_energies = np.sort(sp_energies.real)  # Take real part and sort
    
    print("Single-particle energies:")
    for i, E in enumerate(sp_energies):
        print(f"  E_{i} = {E:.6f}")
    
    # 2. Many-body analysis
    H_mb, basis = create_quspin_hamiltonian(L, t1, t2)
    mb_energies = get_many_body_energies_for_n_particles(L, t1, t2, n_particles)
    
    print(f"\nMany-body energies for {n_particles} particles:")
    for i, E in enumerate(mb_energies):
        print(f"  E_{i} = {E:.6f}")
    
    # 3. Expected energies from single-particle combinations
    expected_energies = get_single_particle_combinations(sp_energies, n_particles)
    
    print(f"\nExpected energies (single-particle combinations):")
    for i, E in enumerate(expected_energies):
        print(f"  E_{i} = {E:.6f}")
    
    # 4. Comparison
    print(f"\nComparison (tolerance = 1e-10):")
    print(f"Number of many-body states: {len(mb_energies)}")
    print(f"Number of expected states: {len(expected_energies)}")
    
    if len(mb_energies) == len(expected_energies):
        max_diff = np.max(np.abs(mb_energies - expected_energies))
        print(f"Maximum difference: {max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("✓ TEST PASSED: Many-body spectrum matches single-particle sum!")
        else:
            print("✗ TEST FAILED: Spectra do not match within tolerance")
            
        # Show differences
        print("\nState-by-state comparison:")
        for i, (mb_E, exp_E) in enumerate(zip(mb_energies, expected_energies)):
            diff = abs(mb_E - exp_E)
            status = "✓" if diff < 1e-10 else "✗"
            print(f"  {status} State {i}: MB={mb_E:.8f}, Expected={exp_E:.8f}, Diff={diff:.2e}")
    else:
        print("✗ TEST FAILED: Different number of states")
    
    return mb_energies, expected_energies, sp_energies


def plot_spectrum_comparison(mb_energies, expected_energies, sp_energies, n_particles):
    """
    Plot comparison of many-body and expected energies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy levels
    ax1.plot(range(len(mb_energies)), mb_energies, 'bo-', label='Many-body', markersize=8)
    ax1.plot(range(len(expected_energies)), expected_energies, 'rx--', 
             label='Single-particle sum', markersize=8, markeredgewidth=2)
    ax1.set_xlabel('State index')
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Energy Spectrum Comparison ({n_particles} particles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Differences
    if len(mb_energies) == len(expected_energies):
        differences = np.abs(mb_energies - expected_energies)
        ax2.semilogy(range(len(differences)), differences, 'go-', markersize=6)
        ax2.set_xlabel('State index')
        ax2.set_ylabel('|Difference|')
        ax2.set_title('Absolute Differences')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1e-10, color='r', linestyle='--', label='Tolerance (1e-10)')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()


import numpy as np
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian

def hubbard_1d(
    L,
    t=1.0,
    V=0.0,
    U=0.0,
    mu=0.0,
    bc="open",                 # "open" or "periodic"
    Nf=None,                   # e.g. (L//2, L//2) for half-filling; or None for all sectors
    double_occupancy=True,
    dtype=np.float64,
):
    """
    Construct H = -t * Σ_{⟨i,j⟩,σ} (c†_{iσ} c_{jσ} + h.c.)
                   + U * Σ_i n_{i↑} n_{i↓}
                   - μ * Σ_i (n_{i↑} + n_{i↓})

    Returns
    -------
    H : quspin.operators.hamiltonian
    basis : quspin.basis.spinful_fermion_basis_1d
    """
    # basis over spin-↑ and spin-↓ fermions; Nf fixes (N_up, N_down) sector if given
    basis = spinful_fermion_basis_1d(
        L,
        Nf=Nf,
        double_occupancy=double_occupancy,
    )

    # nearest-neighbor bonds
    if bc == "periodic":
        bonds_NN = [(i, (i + 1) % L) for i in range(L)]
        bonds_NNN = [(i, (i + 2) % L) for i in range(L)]
    elif bc == "open":
        bonds_NN = [(i, i + 1) for i in range(L - 1)]
        bonds_NNN = [(i, i + 2) for i in range(L - 2)]
    else:
        raise ValueError("bc must be 'open' or 'periodic'")

    # Site-coupling lists: [coef, i, j] for two-site ops; [coef, i] for one-site ops
    hop_NN_pm = [[t, i, j] for (i, j) in bonds_NN]  # "+-" terms (c†_i c_j)
    hop_NN_mp = [[-t, i, j] for (i, j) in bonds_NN]  # "-+" terms (c_i c†_j)
    hop_NNN_pm = [[V, i, j] for (i, j) in bonds_NNN]  # "+-" terms (c†_i c_j)
    hop_NNN_mp = [[-V, i, j] for (i, j) in bonds_NNN]  # "-+" terms (c_i c†_j)
    U_list = [[U, i, i] for i in range(L)]     # "n|n" acts on same site i
    mu_list = [[-mu, i] for i in range(L)]     # -μ(n_up + n_down)

    # spinful opstrings use a pipe: "op_up|op_down"
    static = [
        ["+-|", hop_NN_pm],  # ↑ hopping
        ["-+|", hop_NN_mp],
        ["+-|", hop_NNN_pm],
        ["-+|", hop_NNN_mp],
        ["|+-", hop_NN_pm],  # ↓ hopping
        ["|-+", hop_NN_mp],
        ["|+-", hop_NNN_pm],
        ["|-+", hop_NNN_mp],
        ["n|n", U_list],  # on-site Hubbard U
        ["n|",  mu_list], # chemical potential ↑
        ["|n",  mu_list], # chemical potential ↓
    ]

    H = hamiltonian(static, [], basis=basis, dtype=dtype)
    return H, basis

def analytic_nonint_gs_energy(L,t,V,mu):
    k_grid=np.arange(-np.pi,np.pi,2*np.pi/(L))
    print(f'k_grid shape: {k_grid.shape}')
    E_k=2*t*np.cos(k_grid)+2*V*np.cos(2*k_grid)
    E_occ=np.heaviside((mu-E_k),0)*(E_k-mu)
    E_tot=np.sum(E_occ)
    return E_tot

if __name__ == "__main__":
    L=4
    t=1.0
    V=2.0
    mu=0
    E_analytic_gs=2*analytic_nonint_gs_energy(L=L,t=t,V=V,mu=mu)
    H,basis=hubbard_1d(L=L,t=t,V=V,U=0.0,mu=mu,bc="periodic",Nf=None,double_occupancy=True,dtype=np.float64)
    E_tot_quspin=H.eigvalsh()
    print(E_analytic_gs)
    print(E_tot_quspin[0])
    print(E_analytic_gs-E_tot_quspin[0])


    
    raise Exception("stop")
    # Test parameters
    L = 6  # Number of sites
    
    print("1D Spinful Chain with NN and NNN Hopping Test")
    print("=" * 40)
    
    # Test cases to check
    test_cases = [
        {"t1": 1.0, "t2": 0.5, "name": "General case: t1=1.0, t2=0.5"},
        {"t1": 0.0, "t2": 0.5, "name": "NNN only: t1=0.0, t2=0.5"},
        {"t1": 1.0, "t2": 0.0, "name": "NN only: t1=1.0, t2=0.0"},
    ]
    
    # Test each case
    for case in test_cases:
        t1, t2 = case["t1"], case["t2"]
        print(f"\n{'='*80}")
        print(f"TESTING {case['name']}")
        print(f"{'='*80}")
        
        # Test for different particle numbers
        for n_particles in [1, 2]:  # Reduce to 2 particles for faster testing
            print(f"\n{'='*60}")
            mb_energies, expected_energies, sp_energies = test_spectrum_comparison(
                L=L, t1=t1, t2=t2, n_particles=n_particles
            )
            
            # Plot results (comment out to avoid too many plots)
            # if len(mb_energies) == len(expected_energies):
            #     plot_spectrum_comparison(mb_energies, expected_energies, sp_energies, n_particles)