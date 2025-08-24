import numpy as np
import itertools
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian
from aah_code.hamiltonian import Hubbard1D
from aah_code.hamiltonian import FullSpectrum
from aah_code.basis import LocalClusterBasis
from aah_code.global_params import StatesParams,HamiltonianParams
from aah_code.quspin.quspin_hamiltonian import QuSpinHamiltonian



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


def check_quspin_vs_tenpy_pi_pi():
    cluster_k=np.array([[-np.pi],[0]])
    state_params=StatesParams(spin_states=2)
    cluster_object=LocalClusterBasis(cluster_k,state_params)
    L=2
    t=1.0
    V=2.0
    U=0
    mu=0
    mu_0=0
    physical_params=HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
    ham_dict={'L':L,'t':t,'V':V,'U':U,'mu':mu,'basis_class':cluster_object}

    ham_obj=QuSpinHamiltonian(ham_dict)
    test_ham,test_basis=ham_obj.create_pi_V_pi_int_ham()
    print(f'test_ham shape: {test_ham.shape}')
    eigvals=np.linalg.eigvalsh(test_ham.toarray())
    

    #ten_ham=Hubbard1D(ham_dict)
    
    full_spectrum_object = FullSpectrum(np.array([cluster_k]), state_params, physical_params,ham_lib='tenpy')
    k_points,energy_spectrum,number_spectrum,spin_spectrum=full_spectrum_object.get_full_spectrum()
    
    print(f'energy_spectrum shape: {energy_spectrum.shape}')

    try:
        np.testing.assert_allclose(eigvals,energy_spectrum[0],atol=1e-12,err_msg='eigvals and energy_spectrum do not match')
        print("✅ Tenpy gives the same eigvals as QuSpin for pi modulation, pi cluster")
    except AssertionError as e:
        print("❌ Test failed:", e)

    return np.testing.assert_allclose(eigvals,energy_spectrum[0],atol=1e-12,err_msg='eigvals and energy_spectrum do not match')
    
    

if __name__ == "__main__":


    check_quspin_vs_tenpy_pi_pi()

    exit('testing complete')

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
