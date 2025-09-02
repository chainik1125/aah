


def extract_single_particle_hamiltonian_quspin(hamiltonian,basis):
    """
    Extract the single-particle Hamiltonian from the given hamiltonian and basis.
    Uses the quspin utility function.
    """
    return hamiltonian.tocsc().toarray()[basis.index_list,basis.index_list]