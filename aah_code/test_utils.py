import numpy as np
from numpy.linalg import eigvalsh
from tenpy.algorithms import exact_diag as ed

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

def assert_free_fermion_consistency(model, atol=1e-9, verbose=True, check_terms=False):
    """
    For a quadratic, number-conserving fermion model:
      • N=1 spectrum == single-particle spectrum
      • Full many-body spectrum == subset sums of global single-particle spectrum
    """
    # --- Dense many-body H in documented Kronecker local-basis order ---
    # (The basis is the tensor product of the per-site basis |empty>,|↑>,|↓>,|full|.)
    # TeNPy doc: exact_diag.get_numpy_Hamiltonian(..., undo_sort_charge=True)
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)  # doc
    L = model.lat.N_sites

    # --- Build index lists for N=1, Sz=±1 sectors in that basis ---
    occ  = np.array([0,1,1,2], dtype=np.uint8)  # |0>,|↑>,|↓>,|↑↓>
    s2   = np.array([0,1,-1,0], dtype=np.int8)  # Sz*2 (avoid halves) ; doc: SpinHalfFermionSite

    def select(N, Sz_twice):
        keep = []
        for s in range(4**L):
            tmp, n, sz = s, 0, 0
            for _ in range(L):
                st = tmp & 3; n += occ[st]; sz += s2[st]; tmp >>= 2
                if n > N: break
            if n == N and sz == Sz_twice:
                keep.append(s)
        return np.asarray(keep, dtype=np.int64)

    up_idx   = select(1, +1)
    dn_idx   = select(1, -1)

    H1_up = H[np.ix_(up_idx, up_idx)]     # this IS the single-particle Hamiltonian (↑)
    H1_dn = H[np.ix_(dn_idx, dn_idx)]     # same for (↓)

    # 1) N=1 spectra must match between spins (you didn’t add spin-dependent terms)
    eps_up = np.sort(eigvalsh(H1_up))
    eps_dn = np.sort(eigvalsh(H1_dn))
    np.testing.assert_allclose(eps_up, eps_dn, atol=atol)
    if verbose:
        print("N=1 eigenvalues:", np.round(eps_up, 8))

    # 2) Reconstruct *global* spinful one-body set and sum
    orbitals = np.concatenate([eps_up, eps_dn])  # length 2L
    E = np.array([0.0])
    for e in orbitals:
        E = np.concatenate([E, E + e])           # subset sums (Minkowski sum)
    E_sp = np.sort(E)

    # Compare to exact diagonalization of the full H:
    E_ed = np.sort(eigvalsh(H))
    np.testing.assert_allclose(E_ed, E_sp, atol=atol)
    if verbose:
        print("MB == sums of SP eigenvalues ✔")

    if check_terms:
        # Optional: build A from terms with a *lenient* matcher and compare to H1_up.
        # This avoids missing bonds that sit under a different (op, op_string) bin.
        A = np.zeros((L, L), complex)
        def is_cre_up(x):   return x in ("Cdu",)   # creation ↑
        def is_ann_up(x):   return x in ("Cu",)    # annihilation ↑
        def is_cre_dn(x):   return x in ("Cdd",)   # creation ↓
        def is_ann_dn(x):   return x in ("Cd",)    # annihilation ↓

        # iterate all categories of coupling_terms (doc: CouplingModel.coupling_terms)
        for _, cts in (model.coupling_terms or {}).items():
            for i, left in cts.coupling_terms.items():                 # i < j
                for (op_i, _opstr), right in left.items():             # op_string may be 'JW'
                    for j, ops_j in right.items():
                        for op_j, val in ops_j.items():
                            # spin-↑ channel (either direction)
                            if is_cre_up(op_i) and is_ann_up(op_j): A[i, j] += val
                            if is_ann_up(op_i) and is_cre_up(op_j): A[j, i] += val
                            # spin-↓ channel (ignored; we compare only to ↑)
                            # if you want spinless A, add the ↓ channel into A as well.

        A = 0.5*(A + A.T.conj())
        lam = np.sort(eigvalsh(A))
        try:
            np.testing.assert_allclose(lam, eps_up, atol=atol)
            if verbose:
                print("A(↑) from terms matches N=1(↑) ✔")
        except AssertionError:
            # Helpful dump to see what's missing
            print("❌ A(↑) from terms != N=1(↑). Max |Δe|:", np.max(np.abs(lam - eps_up)))
            print("Δe:", np.round(np.sort(lam) - np.sort(eps_up), 9))
            # Show the stored terms so you can eyeball them
            for _, cts in (model.coupling_terms or {}).items():
                for i, left in cts.coupling_terms.items():
                    for (op_i, opstr), right in left.items():
                        for j, ops_j in right.items():
                            for op_j, v in ops_j.items():
                                print(f"({i},{j}) ({op_i}, op_string={opstr}) -> {op_j}: {v}")
            raise




def debug_free_fermion(model, atol=1e-10, verbose=True, show_terms=True):
    """
    Diagnose why MB != sums-of-SP by checking:
      (A) N=1(↑), N=1(↓) spectra from ED are equal
      (B) N=1(↑) spectrum == eigenvalues of A_up built from model.{onsite,coupling}_terms
      (C) full ED spectrum == all subset sums over the global (spinful) one-body spectrum

    Prints diffs when something fails and returns a dict of arrays/matrices.
    """

    # ---------- (0) ED Hamiltonian in documented basis ----------
    # Basis is Kronecker of local site bases when undo_sort_charge=True.  (doc)
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)  # :contentReference[oaicite:1]{index=1}
    L = model.lat.N_sites

    # Local basis of SpinHalfFermionSite is ['empty','up','down','full'] in this order.  (doc)
    occ_per  = np.array([0,1,1,2], dtype=np.uint8)
    sz_per   = np.array([0,1,-1,0], dtype=np.int8)  # Sz*2 to avoid 1/2s  :contentReference[oaicite:2]{index=2}

    def select_indices(N, Sz_twice=None):
        keep = []
        for s in range(4**L):
            tmp, n, sz = s, 0, 0
            for _ in range(L):
                st = tmp & 3; n += occ_per[st]; sz += sz_per[st]; tmp >>= 2
                if n > N: break
            if n == N and (Sz_twice is None or sz == Sz_twice):
                keep.append(s)
        return np.asarray(keep, dtype=np.int64)

    # ---------- (1) N=1 spectra (these MUST be the one-body spectra) ----------
    idx_up   = select_indices(1, +1)
    idx_down = select_indices(1, -1)
    H1_up, H1_down = H[np.ix_(idx_up, idx_up)], H[np.ix_(idx_down, idx_down)]

    eps_up, eps_down = np.sort(eigvalsh(H1_up)), np.sort(eigvalsh(H1_down))
    if verbose:
        print("eps_up (N=1,↑):", np.round(eps_up, 8))
        print("eps_dn (N=1,↓):", np.round(eps_down, 8))

    try:
        np.testing.assert_allclose(eps_up, eps_down, atol=atol)
    except AssertionError as e:
        print("❌ Spin sectors differ in N=1. If you didn’t add spin-dependent terms, this is a bug.")
        raise

    # ---------- (2) Build the one-body A from the model's term containers ----------
    # This uses the *same* data structures TenPy uses to build the MPO,
    # hence it respects masks and boundary conditions.  (docs)
    def one_body_from_terms(spin):
        want = ("Cdu","Cu") if spin=="up" else ("Cdd","Cd")
        A = np.zeros((L, L), complex)

        # Onsite terms add to the diagonal. OnsiteTerms = list[dict] per site.  (doc)
        for _, ons in (model.onsite_terms or {}).items():                               # :contentReference[oaicite:3]{index=3}
            for i, terms in enumerate(ons.onsite_terms):
                if spin=="up" and "Nu" in terms:  A[i,i] += terms["Nu"]
                if spin=="down" and "Nd" in terms:A[i,i] += terms["Nd"]
                if "Ntot" in terms:               A[i,i] += terms["Ntot"]

        # Two-site hoppings. CouplingTerms is a nested dict with i<j.  (doc)
        for _, cts in (model.coupling_terms or {}).items():                            # :contentReference[oaicite:4]{index=4}
            for i, left in cts.coupling_terms.items():
                for (op_i, _opstr), right in left.items():
                    for j, ops_j in right.items():
                        if op_i == want[0] and want[1] in ops_j:
                            A[i, j] += ops_j[want[1]]
                        if op_i == want[1] and want[0] in ops_j:
                            A[j, i] += ops_j[want[0]]

        # If only one direction was stored somewhere, enforce hermiticity:
        return 0.5*(A + A.T.conj())

    A_up = one_body_from_terms("up")
    lam_up = np.sort(eigvalsh(A_up))

    # Compare A_up to H1_up spectrum:
    try:
        np.testing.assert_allclose(lam_up, eps_up, atol=atol)
    except AssertionError:
        print("❌ N=1(↑) spectrum from ED does NOT match eigenvalues of A_up built from terms.")
        # Show the largest discrepancies and the actual matrices for inspection.
        print("diff one-body eigenvalues:", np.round(np.sort(lam_up) - np.sort(eps_up), 12))
        print("||A_up - H1_up||_F:", np.linalg.norm(A_up - H1_up))
        if show_terms:
            print("\n--- Onsite terms as stored ---")
            for _, ons in (model.onsite_terms or {}).items():
                print(ons.onsite_terms)
            print("\n--- Coupling terms (i,j, op_i-op_j : val) ---")
            for _, cts in (model.coupling_terms or {}).items():
                for i, left in cts.coupling_terms.items():
                    for (op_i, _), right in left.items():
                        for j, ops_j in right.items():
                            for op_j, val in ops_j.items():
                                print(f"({i},{j}) {op_i}-{op_j}: {val}")
        raise

    # ---------- (3) Full many-body = subset sums over global spinful one-body set ----------
    vals = np.concatenate([eps_up, eps_down])  # global 1-body set (length 2L)
    E = np.array([0.0])
    for e in vals:     # Minkowski (subset) sums over orbitals
        E = np.concatenate([E, E + e])

    E_sp = np.sort(E)
    E_ed = np.sort(eigvalsh(H))

    try:
        np.testing.assert_allclose(E_ed, E_sp, atol=atol)
    except AssertionError:
        print("❌ Full ED spectrum != subset sums of global 1-body spectrum.")
        # Pinpoint where/why:
        print("ground energies:  ED =", E_ed[0], "  sums =", E_sp[0], "  Δ =", E_ed[0]-E_sp[0])
        # show a handful of lowest mismatches
        k = 16
        print("\nlowest ED:", np.round(E_ed[:k], 8))
        print("lowest SP:", np.round(E_sp[:k], 8))
        raise

    if verbose:
        print("✅ All checks passed: N=1 matches A_up, and MB == sums-of-SP.")
    return dict(A_up=A_up, eps_up=eps_up, eps_down=eps_down, H1_up=H1_up)




def per_N_compare(model, eps_up, eps_dn, atol=1e-10):
    from tenpy.algorithms import exact_diag as ed
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)
    L = model.lat.N_sites

    # build subsets of orbital sums for each N=0..2L
    orbitals = np.concatenate([eps_up, eps_dn])
    sums_by_N = {0: np.array([0.0])}
    for e in orbitals:
        for N in sorted(list(sums_by_N.keys()), reverse=True):
            sums_by_N.setdefault(N+1, np.array([]))
            sums_by_N[N+1] = np.concatenate([sums_by_N[N+1], sums_by_N[N] + e])
    for N in sums_by_N:
        sums_by_N[N] = np.sort(sums_by_N[N])

    # ED spectra per N
    occ  = np.array([0,1,1,2], dtype=np.uint8)
    def idx_N(N):
        keep=[]
        for s in range(4**L):
            tmp, n = s, 0
            for _ in range(L):
                st = tmp & 3; n += occ[st]; tmp >>= 2
                if n > N: break
            if n == N: keep.append(s)
        return np.asarray(keep, dtype=np.int64)

    for N in range(0, 2*L+1):
        idx = idx_N(N)
        if idx.size == 0: continue
        E_N = np.sort(np.linalg.eigvalsh(H[np.ix_(idx, idx)]))
        try:
            np.testing.assert_allclose(E_N, sums_by_N[N], atol=atol)
        except AssertionError:
            print(f"Mismatch first occurs at particle number N={N}.")
            print("ED  (lowest 16):", np.round(E_N[:16], 8))
            print("SPs (lowest 16):", np.round(sums_by_N[N][:16], 8))
            break

def build_free_H_from_A(model):
    """Reconstruct the full many-body Hamiltonian from the one-body matrix A (↑) measured in N=1.
       Then compare it to ED. If they differ, your MPO has hidden 2-body/pairing terms."""
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)
    L = model.lat.N_sites

    # 1) Extract A_up from N=1, Sz=+1 block (this *is* the one-body matrix for ↑ in site basis).
    occ  = np.array([0,1,1,2], dtype=np.uint8)
    s2   = np.array([0,1,-1,0], dtype=np.int8)  # Sz*2
    def select(N, Sz_twice):
        keep = []
        for s in range(4**L):
            tmp, n, sz = s, 0, 0
            for _ in range(L):
                st = tmp & 3; n += occ[st]; sz += s2[st]; tmp >>= 2
                if n > N: break
            if n == N and sz == Sz_twice: keep.append(s)
        return np.asarray(keep, dtype=np.int64)
    idx_up = select(1, +1)
    H1_up  = H[np.ix_(idx_up, idx_up)]             # basis {|i,↑⟩} with i=0..L-1
    A_up   = H1_up                                  # already the matrix in site basis for ↑
    A_dn   = A_up.copy()

    # 2) Build the *free* second-quantized many-body Hamiltonian from A_up/A_dn
    #    on the same 4^L-dimensional Fock basis (Kronecker) and compare to H.
    # Fast bit-ops builder: act c_i^† c_j (for both spins) on basis states.
    # Local encodings: 0, up, down, full → use 2 bits per site.
    def apply_cdag_c(state, site, spin):  # spin: 0 for up, 1 for down
        # returns (sign, new_state) or (0, state) if annihilates
        # ... implement standard fermionic sign using parity count before the mode ...
        # (Omitted here for brevity; in your code, use the same bit-ops you use elsewhere.)
        raise NotImplementedError

    dim = 4**L
    H_free = np.zeros_like(H, dtype=complex)
    # ↑ channel
    for i in range(L):
        for j in range(L):
            hij = A_up[i, j]
            if abs(hij) < 1e-15: continue
            for s in range(dim):
                sig, t = apply_cdag_c(s, i, 0)  # up spin
                if sig == 0: continue
                sig2, u = apply_cdag_c(t, j, 0) # up spin (annihilation at j)
                if sig2 == 0: continue
                H_free[u, s] += hij * sig * sig2
    # ↓ channel
    for i in range(L):
        for j in range(L):
            hij = A_dn[i, j]
            if abs(hij) < 1e-15: continue
            for s in range(dim):
                sig, t = apply_cdag_c(s, i, 1)  # down spin
                if sig == 0: continue
                sig2, u = apply_cdag_c(t, j, 1)
                if sig2 == 0: continue
                H_free[u, s] += hij * sig * sig2

    print("||H_ED - H_free(A)||_F =", np.linalg.norm(H - H_free))
    return H_free



import numpy as np
from numpy.linalg import eigvalsh
from tenpy.algorithms import exact_diag as ed

def per_N_mismatch_report(model, atol=1e-10, show=12):
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)
    L = model.lat.N_sites

    # selectors in Kronecker local basis (|0>,|↑>,|↓>,|↑↓>)
    occ  = np.array([0,1,1,2], np.uint8)
    sz2  = np.array([0,1,-1,0], np.int8)

    def select(N, Sz_twice=None):
        keep=[]
        for s in range(4**L):
            t,n,sz = s,0,0
            for _ in range(L):
                st = t & 3; n += occ[st]; sz += sz2[st]; t >>= 2
                if n > N: break
            if n == N and (Sz_twice is None or sz == Sz_twice):
                keep.append(s)
        return np.asarray(keep, np.int64)

    # N=1 spectra (these ARE the one-body spectra)
    up = select(1, +1); dn = select(1, -1)
    H1_up = H[np.ix_(up, up)]
    H1_dn = H[np.ix_(dn, dn)]
    E1_up = np.sort(eigvalsh(H1_up))
    E1_dn = np.sort(eigvalsh(H1_dn))
    np.testing.assert_allclose(E1_up, E1_dn, atol=atol)

    # subset sums per particle number from the global spinful set
    orbitals = np.concatenate([E1_up, E1_dn])
    sums_by_N = {0: np.array([0.0])}
    for e in orbitals:
        for N in sorted(list(sums_by_N.keys()), reverse=True):
            sums_by_N.setdefault(N+1, np.array([]))
            sums_by_N[N+1] = np.concatenate([sums_by_N[N+1], sums_by_N[N] + e])
    for N in sums_by_N:
        sums_by_N[N] = np.sort(sums_by_N[N])

    # ED spectra per fixed N
    def idx_N(N):
        keep=[]
        for s in range(4**L):
            t,n = s,0
            for _ in range(L):
                st = t & 3; n += occ[st]; t >>= 2
                if n > N: break
            if n == N: keep.append(s)
        return np.asarray(keep, np.int64)

    for N in range(0, 2*L+1):
        idx = idx_N(N)
        if idx.size == 0: continue
        E_ED = np.sort(eigvalsh(H[np.ix_(idx, idx)]))
        E_SP = sums_by_N[N]
        try:
            np.testing.assert_allclose(E_ED, E_SP, atol=atol)
        except AssertionError:
            print(f"❌ mismatch starts at N = {N}")
            print("ED  (lowest):", np.round(E_ED[:show], 8))
            print("SPs (lowest):", np.round(E_SP[:show], 8))
            print("mean(ED - SP) at this N:", float(np.mean(E_ED - E_SP)))
            break


def _select_indices_N_Sz2(L, N, Sz2=None):
    occ  = np.array([0,1,1,2], np.uint8)   # |0>,|↑>,|↓>,|↑↓>
    sz2  = np.array([0,1,-1,0], np.int8)
    keep=[]
    for s in range(4**L):
        t,n,sz = s,0,0
        for _ in range(L):
            st=t & 3; n+=occ[st]; sz+=sz2[st]; t >>= 2
            if n>N: break
        if n==N and (Sz2 is None or sz==Sz2):
            keep.append(s)
    return np.asarray(keep, np.int64)

def _one_body_from_N1(H, L):
    up = _select_indices_N_Sz2(L, 1, +1)
    # N=1 block *is* the 1-body matrix in site basis for spin-↑
    return H[np.ix_(up, up)]

def _n_op_diag(L, which):  # which: ('up',i), ('down',i) or ('tot',i)
    occ = np.array([0,1,1,2], np.uint8)
    if which[0]=='up':   mask = np.array([0,1,0,1], np.uint8)
    elif which[0]=='down': mask = np.array([0,0,1,1], np.uint8)
    else:                 mask = np.array([0,1,1,2], np.uint8)
    i = which[1]
    diag = np.zeros(4**L, float)
    for s in range(4**L):
        # read local state at site i (2 bits per site)
        st = (s >> (2*i)) & 3
        diag[s] = mask[st]
    return np.diag(diag)

def residual_in_N2(model):
    H = ed.get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)  # documented Kronecker basis
    L = model.lat.N_sites

    # 1) one-body (↑) from N=1
    A = _one_body_from_N1(H, L)

    # 2) build a pure-free H from A for both spins, on full 4^L basis
    #    action of c^†_{iσ} c_{jσ} in this basis is standard; we implement via sparse application
    dim = 4**L
    H_free = np.zeros_like(H, dtype=float)
    # precompute number-parity ("JW") between two sites for signs:
    def jw_sign(state, left, right):
        # counts total fermions (↑+↓) strictly between sites [min+1, max-1]
        if left > right: left, right = right, left
        count = 0
        for k in range(left+1, right):
            st = (state >> (2*k)) & 3
            count += (st==1) + (st==2) + (st==3)  # any occupied adds 1
        return -1.0 if (count % 2) else +1.0

    def apply_cdag_c(state, i, j, spin):  # spin: 0→↑, 1→↓
        # annihilate at j
        stj = (state >> (2*j)) & 3
        want = 1 if spin==0 else 2
        full = 3
        if stj == 0 or (stj==1 and spin==1) or (stj==2 and spin==0):
            return None  # annihilates to zero
        # remove particle at j
        new = state
        if stj == full:
            new ^= (1 << (2*j)) | (1 << (2*j+1))  # from full -> empty, then add back at i below
        elif stj == want:
            new ^= (1 << (2*j+spin))              # toggle that spin bit off

        # create at i
        sti = (new >> (2*i)) & 3
        if sti == 0:
            pass
        elif (sti==1 and spin==1) or (sti==2 and spin==0):
            pass  # other spin present → becomes 'full'
        else:
            return None  # Pauli block

        sign = jw_sign(state, i, j)
        # set the spin bit at i
        if ((new >> (2*i+spin)) & 1) == 1:  # already occupied -> invalid
            return None
        new ^= (1 << (2*i+spin))
        return sign, new

    for i in range(L):
        for j in range(L):
            hij = A[i, j]
            if abs(hij) < 1e-14: continue
            for s in range(dim):
                for spin in (0, 1):  # ↑ and ↓ channels add
                    out = apply_cdag_c(s, i, j, spin)
                    if out is None: continue
                    sign, t = out
                    H_free[t, s] += hij * sign

    # 3) project both to N=2 and compute residual
    N2 = _select_indices_N_Sz2(L, 2, None)
    HN2     = H[np.ix_(N2, N2)]
    HfreeN2 = H_free[np.ix_(N2, N2)]
    R = HN2 - HfreeN2
    print("||R||_F in N=2 =", float(np.linalg.norm(R)))
    return R, N2, HN2, HfreeN2

def fit_R_as_density_like(model, R, N2):
    L = model.lat.N_sites
    # Build N2-sector matrices of n_up(i) n_down(i) and n_tot(i) n_tot(j)
    def project_to_N2(M_full):  # M_full is full 4^L x 4^L diag or simple op
        return M_full[np.ix_(N2, N2)]

    # onsite U_eff
    U_ops = []
    for i in range(L):
        Ni_up   = _n_op_diag(L, ('up', i))
        Ni_down = _n_op_diag(L, ('down', i))
        U_ops.append(project_to_N2(Ni_up @ Ni_down))

    # bond-density V_ij on your active bonds: NN (0-1, 2-3) and NNN (0-2, 1-3)
    bonds = [(0,1), (2,3), (0,2), (1,3)]
    V_ops = []
    for (i,j) in bonds:
        Ni_tot = _n_op_diag(L, ('tot', i))
        Nj_tot = _n_op_diag(L, ('tot', j))
        V_ops.append(project_to_N2(Ni_tot @ Nj_tot))

    # least squares: vec(R) ≈ Σ a_i vec(U_i) + Σ b_k vec(V_k)
    Acols = [op.reshape(-1) for op in (U_ops + V_ops)]
    A = np.stack(Acols, axis=1)
    b = R.reshape(-1)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    U_eff = coeffs[:L]
    V_eff = coeffs[L:]
    # report magnitudes
    print("||R||_F:", float(np.linalg.norm(R)))
    recon = (A @ coeffs).reshape(R.shape)
    print("||R - R_fit||_F:", float(np.linalg.norm(R - recon)))
    for i, val in enumerate(U_eff):
        if abs(val) > 1e-8:
            print(f"U_eff on site {i}: {val}")
    for (bond,val) in zip(bonds, V_eff):
        if abs(val) > 1e-8:
            print(f"V_eff density-density on bond {bond}: {val}")

def scan_for_correlated_hopping(model):
    hits = []
    cts = getattr(model, "coupling_terms", {}) or {}
    for _, bucket in cts.items():
        for i, left in bucket.coupling_terms.items():
            for (op_i, _opstr), right in left.items():
                for j, ops_j in right.items():
                    for op_j, val in ops_j.items():
                        # number ⨂ hop  (either side)
                        if (op_i in ("Nu","Nd","Ntot") and op_j in ("Cu","Cdu","Cd","Cdd")) \
                           or (op_j in ("Nu","Nd","Ntot") and op_i in ("Cu","Cdu","Cd","Cdd")):
                            hits.append((i, j, op_i, op_j, float(val)))
    return hits
