"""
Helper functions to extract the spectrum in QuSpin.
"""

import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis import spinful_fermion_basis_1d
from itertools import combinations
from aah_code.hamiltonian import ClusterExperiment, StatesParams, HamiltonianParams

def run_cluster_method(U, mu_0, V=0, t=1, system_size=10):
    """
    Run cluster method calculation using corrected FullSpectrum class
    
    Args:
        U: Hubbard interaction strength
        mu_0: Chemical potential
        V: Staggered potential (default 0)
        t: Hopping parameter (default 1)
        system_size: Number of lattice sites
    
    Returns:
        (energy, filling): Total energy and filling
    """
    # Create grid and partition into clusters (proper cluster method)
    cluster_size = 2  # 2-site clusters
    cluster_k_generator = system_size // 2  # π separation case
    
    cluster_experiment = ClusterExperiment(
        cluster_size=cluster_size,
        lattice_points=system_size,
        cluster_k_generator=cluster_k_generator,
    )
    
    # Generate k-points grid and clusters
    k_points = cluster_experiment.generate_clusters()
    
    # Set up parameters
    state_params = StatesParams(spin_states=2)
    physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
    
    # Now use the corrected FullSpectrum class
    full_spectrum_object = FullSpectrum(k_points, state_params, physical_params)
    cluster_spectra = full_spectrum_object.get_full_spectrum()
    
    # Get ground state expectations (zero temperature)
    system_expectations, cluster_expectations = full_spectrum_object.get_cluster_thermodynamic_expectations(
        cluster_spectra, temperature=None
    )
    
    energy, filling, spin = system_expectations
    
    return energy, filling