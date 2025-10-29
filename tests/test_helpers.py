import numpy as np
from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.hamiltonian import Hubbard1D, FullSpectrum, inspect_hamiltonian_terms
from aah_code.hamiltonian import QuickHubbard1D, get_spectra, MismatchedQuick
from aah_code.global_params import StatesParams, HamiltonianParams

import logging

#log = logging.getLogger(__name__)


import numpy as np
from numpy.linalg import eigvalsh
from tenpy.algorithms import exact_diag as ed



def check_number_conservation(model):
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)  # doc: basis order
    L = model.lat.N_sites
    # Build N operator in the same Kronecker basis: local states 0, up, down, full → occupation 0,1,1,2
    occ = np.array([0,1,1,2], dtype=np.uint8)
    N_diag = np.zeros(4**L, dtype=float)
    for s in range(4**L):
        tmp, n = s, 0
        for _ in range(L):
            st = tmp & 3; n += occ[st]; tmp >>= 2
        N_diag[s] = n
    N = np.diag(N_diag)

    comm = H @ N - N @ H
    print("||[H, N]||_F =", np.linalg.norm(comm))

def assert_free_fermion_consistency(model, atol=1e-9, verbose=False, max_orbitals=22):
    """
    Assert that a quadratic, number-conserving fermion model's many-body spectrum equals
    the sums of its single-particle eigenvalues.

    Parameters
    ----------
    model : tenpy.models.model.CouplingMPOModel (already initialized)
        Your spinful-fermion model built with add_coupling/add_onsite.
    atol : float
        Tolerance for equality checks.
    verbose : bool
        If True, print small diagnostics.
    max_orbitals : int
        Safety cap: we enumerate 2^(#orbitals) many-body energies. With spin, #orbitals=2L.
        If 2L > max_orbitals, we only perform the N=1 check.

    Raises
    ------
    AssertionError
        If any of the consistency checks fail.
    """
    # ---- (0) Build full many-body Hamiltonian (dense) in documented Kronecker local basis ----
    # Using undo_sort_charge=True gives the plain kronecker-order basis, as the docs specify. :contentReference[oaicite:0]{index=0}
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)

    L = model.lat.N_sites
    if verbose:
        print(f"L={L}, H.shape={H.shape}")

    # Map local basis states |0>,|↑>,|↓>,|↑↓> to (n, Sz).
    # This matches SpinHalfFermionSite’s local state order and operators (Nu/Nd, Cu/Cdu, Cd/Cdd). :contentReference[oaicite:1]{index=1}
    occ_per_st  = np.array([0, 1, 1, 2], dtype=np.uint8)
    spin_per_st = np.array([0, 1,-1, 0], dtype=np.int8)

    def projector_indices(target_n, target_sz=None):
        keep = []
        fourL = 4 ** L
        for idx in range(fourL):
            tmp, n, sz = idx, 0, 0
            for _ in range(L):
                st = tmp & 3
                n  += occ_per_st[st]
                sz += spin_per_st[st]
                tmp >>= 2
                if n > target_n:  # cheap early exit
                    break
            if n == target_n and (target_sz is None or sz == target_sz):
                keep.append(idx)
        return np.asarray(keep, dtype=np.int64)

    # ---- (1) Single-particle spectrum from the N=1 block (spin-up and spin-down) ----
    # For a quadratic number-conserving H, the N=1 spectrum IS the one-body spectrum.
    keep_up   = projector_indices(1, +1)
    keep_down = projector_indices(1, -1)

    H1_up   = H[np.ix_(keep_up,   keep_up)]
    H1_down = H[np.ix_(keep_down, keep_down)]

    eps_up   = np.sort(eigvalsh(H1_up))
    eps_down = np.sort(eigvalsh(H1_down))

    # Assert the two spin sectors are equal unless you intentionally broke SU(2) with Zeeman, etc.
    # (If you have spin-dependent onsite/hopping, this may legitimately differ; in that case, remove this assert.)
    np.testing.assert_allclose(eps_up, eps_down, atol=atol)

    if verbose:
        print("N=1 (↑) eigenvalues:", eps_up)

    # ---- (2) Build the FULL many-body spectrum from the GLOBAL one-body spectrum ----
    # The global spinful one-body set is just the union of the two spin sectors’ N=1 spectra.
    vals = np.concatenate([eps_up, eps_down])  # length 2L

    if 2 * L <= max_orbitals:
        # Enumerate all subset sums (Minkowski sums over orbitals). O(2^(2L)) states.
        E = np.array([0.0])
        for e in vals:
            E = np.concatenate([E, E + e])

        E_sp = np.sort(E)

        # Exact diagonalization of the full many-body Hamiltonian:
        E_ed = np.sort(eigvalsh(H))

        # They MUST match for quadratic, number-conserving fermions (up to sorting). :contentReference[oaicite:2]{index=2}
        np.testing.assert_allclose(E_ed, E_sp, atol=atol)

        if verbose:
            print("MB == sums of SP eigenvalues (global) ✔")
    else:
        if verbose:
            print(f"Skipping full MB enumeration: 2L={2*L} > max_orbitals={max_orbitals}")
            print("N=1 check passed; consider lowering L or raising max_orbitals for full MB test.")

    return {"eps_up": eps_up, "eps_down": eps_down}
