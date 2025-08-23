import numpy as np
import itertools
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian


def quspin_t_tprime_chain(L, t, tp, mu=0.0, U=0.0, bc="open"):
    basis = spinful_fermion_basis_1d(L)  # <-- pass L, not two bases!

    nn  = [(i, i+1) for i in range(L-1)]
    nnn = [(i, i+2) for i in range(L-2)]
    if bc == "periodic":
        nn  += [(L-1, 0)]
        nnn += [(L-2, 0), (L-1, 1)]

    static = []
    # -mu (n_up + n_dn)
    static += [["n|", [[-mu, i] for i in range(L)]]]
    static += [["|n", [[-mu, i] for i in range(L)]]]
    # optional on-site U
    if U != 0.0:
        static += [["n|n", [[U, i, i] for i in range(L)]]]

    # NN hopping -t (c†_i c_j + h.c.) for ↑ and ↓
    static += [["+-|", [[-t, i, j] for (i, j) in nn]]]
    static += [["-+|", [[-t, j, i] for (i, j) in nn]]]
    static += [["|+-", [[-t, i, j] for (i, j) in nn]]]
    static += [["|-+", [[-t, j, i] for (i, j) in nn]]]

    # NNN hopping -t'
    if tp != 0.0:
        static += [["+-|", [[-tp, i, j] for (i, j) in nnn]]]
        static += [["-+|", [[-tp, j, i] for (i, j) in nnn]]]
        static += [["|+-", [[-tp, i, j] for (i, j) in nnn]]]
        static += [["|-+", [[-tp, j, i] for (i, j) in nnn]]]

    H = hamiltonian(static, [], basis=basis, dtype=np.float64)
    return H, basis


def single_particle_tb_from_params(L, t, tp, mu=0.0, bc="open"):
    """Spin-degenerate one-body tight-binding matrix for NN t and NNN t'."""
    H = -mu * np.eye(L, dtype=float)
    # NN
    for i in range(L - 1):
        H[i, i+1] += -t; H[i+1, i] += -t
    # NNN
    for i in range(L - 2):
        H[i, i+2] += -tp; H[i+2, i] += -tp
    if bc == "periodic":
        # wrap NN
        H[0, L-1] += -t;   H[L-1, 0] += -t
        # wrap NNN
        H[0, L-2] += -tp;  H[L-2, 0] += -tp
        H[1, L-1] += -tp;  H[L-1, 1] += -tp
    return H


if __name__ == "__main__":
    # ----- params -----
    L  = 4
    t  = 1.0
    tp = 0.25
    mu = 0.0
    U  = 0.0          # keep zero to test free-fermion reconstruction
    bc = "open"       # or "periodic"

    # ----- build many-body H in QuSpin -----
    H, basis = quspin_t_tprime_chain(L, t, tp, mu, U, bc)
    H_dense = H.toarray()
    mb_evals = np.linalg.eigvalsh(H_dense)

    # ----- build single-particle TB and reconstruct -----
    H1 = single_particle_tb_from_params(L, t, tp, mu, bc)
    sp_evals = np.linalg.eigvalsh(H1)
    sp_evals = np.repeat(sp_evals, 2)  # spin degeneracy

    # sum over all particle numbers (0..2L)
    recon = []
    for k in range(0, 2*L + 1):
        for comb in itertools.combinations(sp_evals, k):
            recon.append(sum(comb))
    recon = np.sort(np.array(recon))
    mb_evals = np.sort(mb_evals)

    print("First few MB eigvals   :", mb_evals[:8])
    print("First few recon eigvals:", recon[:8])
    print("sizes:", mb_evals.size, recon.size)  # both should be 4**L

    # exact match for quadratic Hamiltonian
    np.testing.assert_allclose(mb_evals, recon, atol=1e-12)
    print("✓ Many-body spectrum equals sum of 1-particle eigenvalues.")
