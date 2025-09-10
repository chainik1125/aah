import numpy as np
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d
from quspin.operators import hamiltonian
from typing import Iterable, List, Tuple, Union, Literal, Optional
from aah_code.cluster_model.visualization_helpers import visualize_quspin_couplings_1d_chain
from aah_code.hamiltonian import sparse_diagonalize

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal


SpinMode = Literal["spinless", "spinful"]

# def dft_U_block(Nc: int,
#                 k0: Optional[np.ndarray] = None,
#                 R0: Optional[np.ndarray] = None) -> np.ndarray:
#     if k0 is None:
#         k0 = 2*np.pi*np.arange(Nc)/Nc
#     if R0 is None:
#         R0 = np.arange(Nc, dtype=float)
#     return np.exp(1j*np.outer(np.asarray(k0,float), np.asarray(R0,float))) / np.sqrt(Nc)

# def alpha_dispersion_block(k_phys: np.ndarray,
#                            t: float,
#                            k0: Optional[np.ndarray] = None,
#                            R0: Optional[np.ndarray] = None) -> np.ndarray:
#     k_phys = np.asarray(k_phys, dtype=float)
#     eps = 2.0 * t * np.cos(k_phys)           # ε(k)=2t cos k
#     U = dft_U_block(k_phys.size, k0=k0, R0=R0)
#     return U.conj().T @ np.diag(eps.astype(complex)) @ U  # Nc×Nc

# def alpha_terms_by_separation(
#     supercluster: np.ndarray,
#     k_phys: Union[np.ndarray, List[np.ndarray]],
#     t: float,
#     spin: SpinMode = "spinful",
#     k0: Optional[np.ndarray] = None,
#     R0: Optional[np.ndarray] = None,
#     tol: float = 1e-10
# ) -> Dict[int, List[List[Union[str, list]]]]:
#     """
#     Build *within-cluster* α-basis terms grouped by separation s = (β-α) mod Nc.
#     IMPORTANT: indices are in the flattened α ordering: i = μ*Nc + α.
#     """
#     supercluster = np.asarray(supercluster, dtype=int)
#     B, Nc = supercluster.shape

#     k_phys = np.asarray(k_phys, dtype=float)
#     if k_phys.ndim == 1:
#         k_per_block = np.tile(k_phys, (B, 1))
#     else:
#         assert k_phys.shape == (B, Nc)
#         k_per_block = k_phys

#     sep_to_pm: Dict[int, List[List[Union[complex, int]]]] = {s: [] for s in range(Nc)}
#     sep_to_n:  Dict[int, List[List[Union[float,   int]]]] = {0: []}

#     for mu in range(B):
#         # Nc×Nc α-block for this cluster
#         Hα = alpha_dispersion_block(k_per_block[mu], t, k0=k0, R0=R0)

#         # s=0: on-sites
#         for alpha in range(Nc):
#             val = float(Hα[alpha, alpha].real)
#             if abs(val) > tol:
#                 i = mu*Nc + alpha
#                 if spin == "spinful":
#                     sep_to_n[0].append([val, i])
#                 else:
#                     sep_to_n[0].append([val, i])

#         # s>0: each unordered pair once (α<β), put both directions in the SAME bucket s
#         for alpha in range(Nc):
#             for s in range(1, Nc):
#                 beta = (alpha + s) % Nc
#                 if not (alpha < beta):  # process unordered {alpha,beta} once
#                     continue
#                 amp = Hα[beta, alpha]
#                 if abs(amp) <= tol:
#                     continue
#                 i = mu*Nc + beta  # dest β
#                 j = mu*Nc + alpha # src  α
#                 sep_to_pm[s].append([complex(amp), i, j])
#                 sep_to_pm[s].append([complex(np.conjugate(amp)), j, i])

#     # Assemble QuSpin statics by separation
#     terms_by_sep: Dict[int, List[List[Union[str, list]]]] = {}
#     for s in range(Nc):
#         static: List[List[Union[str, list]]] = []
#         if s == 0:
#             if sep_to_n[0]:
#                 if spin == "spinful":
#                     static.append(["n|", sep_to_n[0]])
#                     static.append(["|n", sep_to_n[0]])
#                 else:
#                     static.append(["n", sep_to_n[0]])
#         else:
#             if sep_to_pm[s]:
#                 if spin == "spinful":
#                     static += [["+-|", sep_to_pm[s]], ["|+-", sep_to_pm[s]]]
#                 else:
#                     static += [["+-", sep_to_pm[s]]]
#         if static:
#             terms_by_sep[s] = static
#     return terms_by_sep




#import numpy as np
#from typing import Dict, List, Optional, Tuple, Union, Literal

#SpinMode = Literal["spinless","spinful"]

# --- helpers ---

def k_phys_from_supercluster(supercluster: np.ndarray, L: int,
							 row_rolls: Optional[List[int]] = None) -> np.ndarray:
	"""
	Build k_phys per cluster directly from the supercluster's global k indices:
	  k_phys[mu,a] = 2π * supercluster[mu,a] / L
	If row_rolls is given, roll each row by row_rolls[mu] (same gauge used for V).
	"""
	supercluster = np.asarray(supercluster, dtype=int)
	B, Nc = supercluster.shape
	k_phys = (2.0*np.pi/L) * supercluster.astype(float)
	if row_rolls is not None:
		assert len(row_rolls) == B
		for mu, r in enumerate(row_rolls):
			if r % Nc != 0:
				k_phys[mu] = np.roll(k_phys[mu], - (r % Nc))
	return k_phys

def dft_U_block(Nc: int,
				k0: Optional[np.ndarray] = None,
				R0: Optional[np.ndarray] = None) -> np.ndarray:
	if k0 is None:
		k0 = 2*np.pi*np.arange(Nc)/Nc
	if R0 is None:
		R0 = np.arange(Nc, dtype=float)
	return np.exp(1j*np.outer(np.asarray(k0,float), np.asarray(R0,float))) / np.sqrt(Nc)

def alpha_dispersion_block(k_phys: np.ndarray,
						   t: float,
						   k0: Optional[np.ndarray] = None,
						   R0: Optional[np.ndarray] = None) -> np.ndarray:
	k_phys = np.asarray(k_phys, dtype=float)
	eps = 2.0 * t * np.cos(k_phys)           # ε(k)=2t cos k
	U = dft_U_block(k_phys.size, k0=k0, R0=R0)
	return U.conj().T @ np.diag(eps.astype(np.complex128)) @ U  # Nc×Nc, hermitian

# --- patched t~ builder (indices are flattened α: i = mu*Nc + alpha) ---

def alpha_terms_by_separation(supercluster: np.ndarray,
							  k_phys: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
							  t: float = 1.0,
							  spin: SpinMode = "spinful",
							  k0: Optional[np.ndarray] = None,
							  R0: Optional[np.ndarray] = None,
							  tol: float = 1e-12,
							  *,
							  L: Optional[int] = None,
							  row_rolls: Optional[List[int]] = None
							  ) -> Dict[int, List[List[Union[str, list]]]]:
	"""
	Within-cluster α terms grouped by separation s=(β-α) mod Nc.
	IMPORTANT:
	  - If k_phys is None, it is constructed from supercluster & L.
	  - If row_rolls is provided, the same rolls are applied to k_phys rows (to match V).
	  - Indices are flattened α: i = μ*Nc + α.
	"""
	supercluster = np.asarray(supercluster, dtype=int)
	B, Nc = supercluster.shape

	# k_phys per row
	if k_phys is None:
		assert L is not None, "Pass L if you want k_phys computed internally."
		k_per_block = k_phys_from_supercluster(supercluster, L=L, row_rolls=row_rolls)
	else:
		k_per_block = np.asarray(k_phys, dtype=float)
		if k_per_block.ndim == 1:
			# DANGEROUS in mismatched-period cases; keep only if you *intend* to tile:
			k_per_block = np.tile(k_per_block, (B, 1))
		else:
			assert k_per_block.shape == (B, Nc)
			# if you rolled rows for V, you probably also want to roll k_phys here:
			if row_rolls is not None:
				for mu, r in enumerate(row_rolls):
					if r % Nc != 0:
						k_per_block[mu] = np.roll(k_per_block[mu], - (r % Nc))

	sep_to_pm: Dict[int, List[List[Union[complex, int]]]] = {s: [] for s in range(Nc)}
	sep_to_n:  Dict[int, List[List[Union[float,   int]]]] = {0: []}

	for mu in range(B):
		Hα = alpha_dispersion_block(k_per_block[mu], t, k0=k0, R0=R0)  # Nc×Nc, Hermitian

		# s=0 on-sites
		for alpha in range(Nc):
			val = float(Hα[alpha, alpha].real)
			if abs(val) > tol:
				i = mu*Nc + alpha
				if spin == "spinful":
					sep_to_n[0].append([val, i])
					# Note: 'n|' and '|n' will both use same list
				else:
					sep_to_n[0].append([val, i])

		# s>0: each unordered pair once (α<β), add both directions
		for alpha in range(Nc):
			for s in range(1, Nc):
				beta = (alpha + s) % Nc
				if not (alpha < beta):
					continue
				amp = complex(Hα[beta, alpha])
				if abs(amp) <= tol:
					continue
				i = mu*Nc + beta  # dest β
				j = mu*Nc + alpha # src  α
				sep_to_pm[s].append([amp, i, j])
				sep_to_pm[s].append([amp.conjugate(), j, i])

	# Assemble QuSpin statics by separation
	terms_by_sep: Dict[int, List[List[Union[str, list]]]] = {}
	for s in range(Nc):
		static: List[List[Union[str, list]]] = []
		if s == 0:
			if sep_to_n[0]:
				if spin == "spinful":
					static.append(["n|", sep_to_n[0]])
					static.append(["|n", sep_to_n[0]])
				else:
					static.append(["n", sep_to_n[0]])
		else:
			if sep_to_pm[s]:
				if spin == "spinful":
					static += [["+-|", sep_to_pm[s]], ["|+-", sep_to_pm[s]]]
				else:
					static += [["+-", sep_to_pm[s]]]
		if static:
			terms_by_sep[s] = static
	return terms_by_sep


def dedupe_s0_inplace(terms_by_sep):
    """
    In-place: coalesce duplicate s=0 (number-operator) entries.
    Works for spinless ['n'] and spinful ['n|', '|n'] statics.
    """
    if 0 not in terms_by_sep:
        return terms_by_sep

    def _coalesce(lst):
        acc = {}
        for val, i in lst:
            acc[int(i)] = acc.get(int(i), 0.0) + float(val)
        # sort for determinism
        return [[float(v), int(i)] for i, v in sorted(acc.items(), key=lambda x: x[0])]

    cleaned = []
    s0_statics = terms_by_sep[0]
    # coalesce each s=0 opstring independently
    for opstr, lst in s0_statics:
        if opstr in ("n", "n|", "|n"):
            cleaned.append([opstr, _coalesce(lst)])
        else:
            cleaned.append([opstr, lst])

    # (optional) ensure 'n|' and '|n' carry the SAME coalesced list
    ops = {op for op, _ in cleaned}
    if "n|" in ops and "|n" in ops:
        # union & coalesce once, then share
        union = _coalesce([item for op, lst in cleaned if op in ("n|", "|n") for item in lst])
        cleaned = [[op, union if op in ("n|", "|n") else lst] for op, lst in cleaned]

    terms_by_sep[0] = cleaned
    return terms_by_sep


def dedupe_s0_static(static):
    """
    Return a new static list with s=0 duplicates coalesced.
    """
    def _coalesce(lst):
        acc = {}
        for val, i in lst:
            acc[int(i)] = acc.get(int(i), 0.0) + float(val)
        return [[float(v), int(i)] for i, v in sorted(acc.items(), key=lambda x: x[0])]

    out = []
    # first pass: coalesce per opstring
    for op, lst in static:
        if op in ("n", "n|", "|n"):
            out.append([op, _coalesce(lst)])
        else:
            out.append([op, lst])

    # optional: enforce identical lists for 'n|' and '|n'
    have_up = any(op == "n|" for op, _ in out)
    have_dn = any(op == "|n" for op, _ in out)
    if have_up and have_dn:
        union = _coalesce([item for op, lst in out if op in ("n|", "|n") for item in lst])
        out = [[op, union if op in ("n|", "|n") else lst] for op, lst in out]
    return out

def minimal_n3_failure():
	L=6
	Nc=3
	t=1.0

	v_sep_ratio=(1,6)
	matched_int_sep_ratio=(1,3)
	mismatched_int_sep_ratio=(1,6)

	supercluster_matched=generate_clusters(L,Nc,matched_int_sep_ratio,v_sep_ratio)[0]
	supercluster_matched_k=convert_site_clusters_to_k(supercluster_matched,L)

	supercluster_unmatched=generate_clusters(L,Nc,mismatched_int_sep_ratio,v_sep_ratio)[0]
	supercluster_unmatched_k=convert_site_clusters_to_k(supercluster_unmatched,L)

	print(f"Supercluster matched shape: {supercluster_matched.shape}")
	print(f"Supercluster matched: {supercluster_matched}")
	print(f"Supercluster matched k: {supercluster_matched_k/np.pi}")

	print(f"Supercluster unmatched shape: {supercluster_unmatched.shape}")
	print(f"Supercluster unmatched: {supercluster_unmatched}")
	print(f"Supercluster unmatched k: {supercluster_unmatched_k/np.pi}")

	alpha_terms_spinless_matched=alpha_terms_by_separation(supercluster_matched,supercluster_matched_k,t,spin="spinless")
	alpha_terms_spinless_unmatched=alpha_terms_by_separation(supercluster_unmatched,supercluster_unmatched_k,t,spin="spinless")

	alpha_terms_spinful_matched=alpha_terms_by_separation(supercluster_matched,supercluster_matched_k,t,spin="spinful")
	alpha_terms_spinful_unmatched=alpha_terms_by_separation(supercluster_unmatched,supercluster_unmatched_k,t,spin="spinful")

	static_matched_spinless=[]
	static_matched_spinful=[]
	for s in sorted(alpha_terms_spinless_matched.keys()):
		static_matched_spinless.extend(alpha_terms_spinless_matched[s])
	for s in sorted(alpha_terms_spinful_matched.keys()):
		static_matched_spinful.extend(alpha_terms_spinful_matched[s])
	
	static_unmatched_spinless=[]
	static_unmatched_spinful=[]
	
	for s in sorted(alpha_terms_spinless_unmatched.keys()):
		static_unmatched_spinless.extend(alpha_terms_spinless_unmatched[s])
	for s in sorted(alpha_terms_spinful_unmatched.keys()):
		static_unmatched_spinful.extend(alpha_terms_spinful_unmatched[s])
	
	print(f"Alpha terms spinless matched: {static_matched_spinless}")
	print(f"Alpha terms spinless unmatched: {static_unmatched_spinless}")
	print(f"Alpha terms spinful matched: {static_matched_spinful}")
	print(f"Alpha terms spinful unmatched: {static_unmatched_spinful}")
	
	spinless_basis_1p=spinless_fermion_basis_1d(L=6,Nf=1)

	H_matched_1p = hamiltonian(static_matched_spinless, [], basis=spinless_basis_1p, dtype=np.complex128)
	H_unmatched_1p = hamiltonian(static_unmatched_spinless, [], basis=spinless_basis_1p, dtype=np.complex128)

	H_matched_1p_array=H_matched_1p.toarray()
	H_unmatched_1p_array=H_unmatched_1p.toarray()

	print(f"H_matched_1p_array shape: {H_matched_1p_array.shape}")
	print(f"H_unmatched_1p_array shape: {H_unmatched_1p_array.shape}")

	matched_evals,matched_evecs=np.linalg.eigh(H_matched_1p_array)
	unmatched_evals,unmatched_evecs=np.linalg.eigh(H_unmatched_1p_array)

	print(f"Matched evals: {matched_evals}")
	print(f"Unmatched evals: {unmatched_evals}")
	
	spinful_basis_full=spinful_fermion_basis_1d(L=6)

	H_matched_full = hamiltonian(static_matched_spinful, [], basis=spinful_basis_full, dtype=np.complex128)
	H_unmatched_full = hamiltonian(static_unmatched_spinful, [], basis=spinful_basis_full, dtype=np.complex128)

	H_matched_full_array=H_matched_full.toarray()
	H_unmatched_full_array=H_unmatched_full.toarray()

	print(f"H_matched_full_array shape: {H_matched_full_array.shape}")
	print(f"H_unmatched_full_array shape: {H_unmatched_full_array.shape}")

	matched_evals_full,matched_evecs_full=sparse_diagonalize(H_matched_full,k=6)
	unmatched_evals_full,unmatched_evecs_full=sparse_diagonalize(H_unmatched_full,k=6)

	print(f"Matched evals full: {matched_evals_full}")
	print(f"Unmatched evals full: {unmatched_evals_full}")
	print(f'full spinful spectra same: {np.allclose(matched_evals_full,unmatched_evals_full)}')

	#do full spectra match?

	# matched_evals_full_dense,matched_evecs_full_dense=np.linalg.eigh(H_matched_full_array)
	# unmatched_evals_full_dense,unmatched_evecs_full_dense=np.linalg.eigh(H_unmatched_full_array)

	# print(f"Matched evals full dense: {matched_evals_full_dense[:6]}")
	# print(f"Unmatched evals full dense: {unmatched_evals_full_dense[:6]}")
	# print(f'full (dense solved) spinful spectra same: {np.allclose(matched_evals_full_dense,unmatched_evals_full_dense)}')

	#Do the spinless full case as well
	spinless_basis_full=spinless_fermion_basis_1d(L=6)

	H_matched_full_spinless = hamiltonian(static_matched_spinless, [], basis=spinless_basis_full, dtype=np.complex128)
	H_unmatched_full_spinless = hamiltonian(static_unmatched_spinless, [], basis=spinless_basis_full, dtype=np.complex128)

	H_matched_full_spinless_array=H_matched_full_spinless.toarray()
	H_unmatched_full_spinless_array=H_unmatched_full_spinless.toarray()

	print(f"H_matched_full_spinless_array shape: {H_matched_full_spinless_array.shape}")
	print(f"H_unmatched_full_spinless_array shape: {H_unmatched_full_spinless_array.shape}")

	

	matched_evals_full_spinless,matched_evecs_full_spinless=np.linalg.eigh(H_matched_full_spinless_array)#sparse_diagonalize(H_matched_full_spinless,k=10)
	unmatched_evals_full_spinless,unmatched_evecs_full_spinless=np.linalg.eigh(H_unmatched_full_spinless_array)#sparse_diagonalize(H_unmatched_full_spinless,k=10)

	print(f"Matched evals full spinless: {matched_evals_full_spinless[:6]}")
	print(f"Unmatched evals full spinless: {unmatched_evals_full_spinless[:6]}")
	
	print(f'spinless spectra same: {np.allclose(matched_evals_full_spinless,unmatched_evals_full_spinless)}')


	#Now try and see if they'll match after we remove duplicates
	
	static_matched_spinful_deduped=dedupe_s0_static(static_matched_spinful)
	static_unmatched_spinful_deduped=dedupe_s0_static(static_unmatched_spinful)

	print(f'deduped spinful static: \n {static_matched_spinful_deduped}')
	print(f'deduped spinful static unmatched:\n {static_unmatched_spinful_deduped}')
	
	H_matched_full_spinful_deduped=hamiltonian(static_matched_spinful_deduped, [], basis=spinful_basis_full, dtype=np.complex128)
	H_unmatched_full_spinful_deduped=hamiltonian(static_unmatched_spinful_deduped, [], basis=spinful_basis_full, dtype=np.complex128)

	H_matched_full_spinful_deduped_array=H_matched_full_spinful_deduped.toarray()
	H_unmatched_full_spinful_deduped_array=H_unmatched_full_spinful_deduped.toarray()

	print(f"H_matched_full_spinful_deduped_array shape: {H_matched_full_spinful_deduped_array.shape}")
	print(f"H_unmatched_full_spinful_deduped_array shape: {H_unmatched_full_spinful_deduped_array.shape}")

	#Use sparse solver
	matched_evals_full_spinful_sparse,matched_evecs_full_spinful_sparse=sparse_diagonalize(H_matched_full_spinful_deduped,k=6)
	unmatched_evals_full_spinful_sparse,unmatched_evecs_full_spinful_sparse=sparse_diagonalize(H_unmatched_full_spinful_deduped,k=6)

	print(f"Matched evals full spinful (sparse): {matched_evals_full_spinful_sparse[:6]}")
	print(f"Unmatched evals full spinful (sparse): {unmatched_evals_full_spinful_sparse[:6]}")

	print(f'spinful spectra same: {np.allclose(matched_evals_full_spinful_sparse,unmatched_evals_full_spinful_sparse)}')
	
	#Solve densely to be sure 
	matched_evals_full_spinful,matched_evecs_full_spinful=np.linalg.eigh(H_matched_full_spinful_deduped_array)
	unmatched_evals_full_spinful,unmatched_evecs_full_spinful=np.linalg.eigh(H_unmatched_full_spinful_deduped_array)

	print(f"Matched evals full spinful (dense): {matched_evals_full_spinful[:6]}")
	print(f"Unmatched evals full spinful (dense): {unmatched_evals_full_spinful[:6]}")

	print(f'spinful spectra same: {np.allclose(matched_evals_full_spinful,unmatched_evals_full_spinful)}')

if __name__ == "__main__":

	minimal_n3_failure()
	exit()

	L=6
	Nc=3
	t=1.0

	int_sep_ratio=(1,6)  # Test with 1x2 supercluster
	v_sep_ratio=(1,6)
	unmatched_int_sep=(1,3)

	supercluster=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)[0]
	supercluster_k=convert_site_clusters_to_k(supercluster,L)

	supercluster_unmatched=generate_clusters(L,Nc,unmatched_int_sep,v_sep_ratio)[0]
	supercluster_unmatched_k=convert_site_clusters_to_k(supercluster_unmatched,L)

	print(f"Supercluster shape: {supercluster.shape}")
	print(f"Supercluster: {supercluster}")
	print(f"Supercluster k: {supercluster_k/np.pi}")

	print(f"Supercluster unmatched shape: {supercluster_unmatched.shape}")
	print(f"Supercluster unmatched: {supercluster_unmatched}")
	print(f"Supercluster unmatched k: {supercluster_unmatched_k/np.pi}")

	#Test spinless case for varying separation

	alpha_terms_spinless_matched=alpha_terms_by_separation(supercluster,supercluster_k,t,spin="spinless")
	alpha_terms_spinless_unmatched=alpha_terms_by_separation(supercluster_unmatched,supercluster_unmatched_k,t,spin="spinless")

	static_matched=[]
	for s in sorted(alpha_terms_spinless_matched.keys()):
		static_matched.extend(alpha_terms_spinless_matched[s])
	static_unmatched=[]
	for s in sorted(alpha_terms_spinless_unmatched.keys()):
		static_unmatched.extend(alpha_terms_spinless_unmatched[s])

	print(f"Alpha terms spinless matched: {static_matched}")
	print(f"Alpha terms spinless unmatched: {static_unmatched}")
	
	
	spinless_basis_1p=spinless_fermion_basis_1d(L=6,Nf=1)

	H_matched = hamiltonian(static_matched, [], basis=spinless_basis_1p, dtype=np.complex128)
	H_unmatched = hamiltonian(static_unmatched, [], basis=spinless_basis_1p, dtype=np.complex128)
	# H_matched=hamiltonian(static_matched,[],spinless_basis,dtype=np.complex128)
	# H_unmatched=hamiltonian(static_unmatched,[],spinless_basis,dtype=np.complex128)

	H_matched_array=H_matched.toarray()
	H_unmatched_array=H_unmatched.toarray()

	print(f"H_matched_array shape: {H_matched_array.shape}")
	print(f"H_unmatched_array shape: {H_unmatched_array.shape}")
	
	matched_evals,matched_evecs=np.linalg.eigh(H_matched_array)
	unmatched_evals,unmatched_evecs=np.linalg.eigh(H_unmatched_array)

	print(f"Matched evals: {matched_evals}")
	print(f"Unmatched evals: {unmatched_evals}")

	#try to use the sparse solver routine to see if its a many-body issue

	spinless_basis_full=spinless_fermion_basis_1d(L=6)

	H_matched_full=hamiltonian(static_matched, [], basis=spinless_basis_full, dtype=np.complex128)
	H_unmatched_full=hamiltonian(static_unmatched, [], basis=spinless_basis_full, dtype=np.complex128)

	H_matched_full_array=H_matched_full.toarray()
	H_unmatched_full_array=H_unmatched_full.toarray()

	matched_evals_full,_=sparse_diagonalize(H_matched_full,k=6)
	unmatched_evals_full,_=sparse_diagonalize(H_unmatched_full,k=6)

	print(f"Matched evals full: {matched_evals_full}")
	print(f"Unmatched evals full: {unmatched_evals_full}")
	
	
	

	exit()

	# Get both spinful and spinless terms
	alpha_terms_spinful = alpha_terms_by_separation(supercluster, supercluster_k, t, spin="spinful")
	alpha_terms_spinless = alpha_terms_by_separation(supercluster, supercluster_k, t, spin="spinless")
	
	alpha_terms_list=[alpha_terms_spinful[s] for s in alpha_terms_spinful]

	
	print(f"Alpha terms keys: {alpha_terms_spinful.keys()}")
	print(f"Alpha terms: {alpha_terms_spinful}")

	# Create visualization
	import os
	os.makedirs('large_files/viz', exist_ok=True)
	
	# Combine all terms for visualization
	all_static_spinful = []
	all_static_spinless = []
	for s in sorted(alpha_terms_spinful.keys()):
		all_static_spinful.extend(alpha_terms_spinful[s])
	for s in sorted(alpha_terms_spinless.keys()):
		all_static_spinless.extend(alpha_terms_spinless[s])
	
	visualize_quspin_couplings_1d_chain(
		all_static_spinful,
		k_sites_supercluster=supercluster,
		output_file='large_files/viz/t_tilde_visualization.html',
		title="t-tilde Terms Visualization",
		to_quspin_spinless=all_static_spinless
	)
	#print("Visualization saved to large_files/viz/t_tilde_visualization.html")

	# Then:
	# from quspin.operators import hamiltonian
	# H = hamiltonian(static, [], basis=your_spinful_basis)
