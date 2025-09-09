import numpy as np
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from typing import Iterable, List, Tuple, Union, Literal, Optional
from aah_code.cluster_model.visualization_helpers import visualize_quspin_couplings_1d_chain

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal


SpinMode = Literal["spinless", "spinful"]

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
    return U.conj().T @ np.diag(eps.astype(complex)) @ U  # Nc×Nc

def alpha_terms_by_separation(
    supercluster: np.ndarray,
    k_phys: Union[np.ndarray, List[np.ndarray]],
    t: float,
    spin: SpinMode = "spinful",
    k0: Optional[np.ndarray] = None,
    R0: Optional[np.ndarray] = None,
    tol: float = 1e-10
) -> Dict[int, List[List[Union[str, list]]]]:
    """
    Build *within-cluster* α-basis terms grouped by separation s = (β-α) mod Nc.
    IMPORTANT: indices are in the flattened α ordering: i = μ*Nc + α.
    """
    supercluster = np.asarray(supercluster, dtype=int)
    B, Nc = supercluster.shape

    k_phys = np.asarray(k_phys, dtype=float)
    if k_phys.ndim == 1:
        k_per_block = np.tile(k_phys, (B, 1))
    else:
        assert k_phys.shape == (B, Nc)
        k_per_block = k_phys

    sep_to_pm: Dict[int, List[List[Union[complex, int]]]] = {s: [] for s in range(Nc)}
    sep_to_n:  Dict[int, List[List[Union[float,   int]]]] = {0: []}

    for mu in range(B):
        # Nc×Nc α-block for this cluster
        Hα = alpha_dispersion_block(k_per_block[mu], t, k0=k0, R0=R0)

        # s=0: on-sites
        for alpha in range(Nc):
            val = float(Hα[alpha, alpha].real)
            if abs(val) > tol:
                i = mu*Nc + alpha
                if spin == "spinful":
                    sep_to_n[0].append([val, i])
                else:
                    sep_to_n[0].append([val, i])

        # s>0: each unordered pair once (α<β), put both directions in the SAME bucket s
        for alpha in range(Nc):
            for s in range(1, Nc):
                beta = (alpha + s) % Nc
                if not (alpha < beta):  # process unordered {alpha,beta} once
                    continue
                amp = Hα[beta, alpha]
                if abs(amp) <= tol:
                    continue
                i = mu*Nc + beta  # dest β
                j = mu*Nc + alpha # src  α
                sep_to_pm[s].append([complex(amp), i, j])
                sep_to_pm[s].append([complex(np.conjugate(amp)), j, i])

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







if __name__ == "__main__":
	L=3
	Nc=3
	t=1.0

	int_sep_ratio=(1,3)  # Test with 1x2 supercluster
	v_sep_ratio=(1,3)

	supercluster=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)[0]
	supercluster_k=convert_site_clusters_to_k(supercluster,L)

	print(f"Supercluster shape: {supercluster.shape}")
	print(f"Supercluster: {supercluster}")
	print(f"Supercluster k: {supercluster_k/np.pi}")

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
