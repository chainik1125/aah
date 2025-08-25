"""
Helper functions to extract the spectrum in QuSpin.
"""

import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis import spinful_fermion_basis_1d
from itertools import combinations
#from aah_code.hamiltonian import ClusterExperiment, StatesParams, HamiltonianParams
#from aah_code.hamiltonian import FullSpectrum
from typing import Literal, Tuple, List

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




SpinChoice = Literal["up", "down", "both"]

def _check_spinful_basis(basis):
    if not isinstance(basis, spinful_fermion_basis_1d):
        raise TypeError("Expected a spinful_fermion_basis_1d basis.")

def _one_particle_index(basis: spinful_fermion_basis_1d, site: int, spin: Literal["up", "down"]) -> int:
    """Index of |site,spin> = c†_{site,spin} |vac> in the many-body basis."""
    up_bits   = (1 << site) if spin == "up" else 0
    down_bits = (1 << site) if spin == "down" else 0
    idx = basis.index(up_bits, down_bits)
    if idx is None or idx < 0:
        raise ValueError(
            f"State |{site},{spin}> not present in this basis. "
            "Rebuild with Nf=None or a compatible sector."
        )
    return idx

def _one_particle_indices(basis: spinful_fermion_basis_1d, spin: SpinChoice) -> Tuple[List[int], List[Tuple[int, str]]]:
    """Return (indices, labels) for the chosen 1-particle sector, ordered as requested."""
    _check_spinful_basis(basis)
    L = basis.L
    if spin == "up":
        labels = [(i, "up") for i in range(L)]
    elif spin == "down":
        labels = [(i, "down") for i in range(L)]
    elif spin == "both":
        labels = [(i, "up") for i in range(L)] + [(i, "down") for i in range(L)]
    else:
        raise ValueError("spin must be 'up', 'down', or 'both'.")
    indices = [_one_particle_index(basis, i, s) for i, s in labels]
    return indices, labels

def extract_single_particle_hamiltonian_dense(
    H: hamiltonian,
    basis: spinful_fermion_basis_1d,
    spin: SpinChoice = "both",
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Dense extraction of the single-particle Hamiltonian from a QuSpin many-body H.

    Parameters
    ----------
    H : quspin.operators.hamiltonian
        Many-body Hamiltonian defined on `basis`.
    basis : quspin.basis.spinful_fermion_basis_1d
        Basis used to build H. Must include the desired 1-particle sector.
    spin : {'up','down','both'}, default 'both'
        Which single-particle sector to extract.
        - 'up'   -> L x L
        - 'down' -> L x L
        - 'both' -> (2L) x (2L) with up (0..L-1) then down (L..2L-1)

    Returns
    -------
    H_sp : np.ndarray (dense)
        Single-particle Hamiltonian in the chosen sector, ordered by `labels`.
    labels : list[(site, 'up'/'down')]
        Row/column labels matching H_sp.
    """
    _check_spinful_basis(basis)
    idxs, labels = _one_particle_indices(basis, spin)

    # Convert to dense once, then take the principal submatrix
    H_dense = H.tocsc().toarray()  # QuSpin -> SciPy CSC -> dense numpy
    H_sp = H_dense[np.ix_(idxs, idxs)]
    return H_sp, labels