"""
Code for generating the Hamiltonian on a given supercluster entry.

"""

import numpy as np
import quspin
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from typing import Any, Union,Tuple, Dict, List, Optional
from collections import defaultdict,deque

from aah_code.cluster_model.clustering import step_from_ratio
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from fractions import Fraction
from aah_code.cluster_model.visualization_helpers import visualize_quspin_couplings_1d_chain
import os



def _step_from_ratio(L: int, ratio: Tuple[int,int]) -> int:
    p, q = ratio
    if q == 0:
        raise ValueError("denominator must be nonzero")
    val = Fraction(p, q) * L
    if val.denominator != 1:
        raise ValueError(f"(p/q)*L = {val} not integer; need (p*L) % q == 0")
    return int(val.numerator % L)


import numpy as np
from typing import Dict, Tuple, List, Optional

def build_V_matrix_k_only(L:int, n:int, V0:float=1.0):
    """
    Steps 1–2: H_k for +n only (NO hermitian conjugate).
    H_k[k', k] = V0 when k' = (k+n) mod L.
    """
    Hk = np.zeros((L, L), dtype=np.complex128)
    for k in range(L):
        kp = (k + n) % L
        Hk[kp, k] += V0
    return Hk

def cluster_permutation(k_sites_supercluster: np.ndarray) -> np.ndarray:
    """Step 3: permutation p s.t. k-basis -> clustered order [C0..., C1..., ...]."""
    k_sites_supercluster = np.asarray(k_sites_supercluster, dtype=int)
    p = []
    for row in k_sites_supercluster:
        p.extend(list(row))
    return np.array(p, dtype=int)

def reorder_by_clusters(Hk: np.ndarray, k_sites_supercluster: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the permutation that groups rows/cols by clusters."""
    p = cluster_permutation(k_sites_supercluster)
    Hc = Hk[np.ix_(p, p)]
    return Hc, p

def block_alpha_transform(k_sites_supercluster: np.ndarray, R_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Step 4: block-diagonal DFT within each cluster (canonical α-basis):
      F_mu[a,α] = (1/sqrt(Nc)) * exp(-i κ_a R_α),  κ_a = 2π a / Nc,  R_α = α (default).
    Returns F_block of size (C*Nc) x (C*Nc).
    """
    rows = np.asarray(k_sites_supercluster, dtype=int)
    C, Nc = rows.shape
    if R_alpha is None:
        R_alpha = np.arange(Nc, dtype=float)
    else:
        R_alpha = np.asarray(R_alpha, dtype=float)
        assert R_alpha.shape == (Nc,)
    kappa = 2.0 * np.pi * np.arange(Nc, dtype=float) / float(Nc)

    F_block = np.zeros((C*Nc, C*Nc), dtype=np.complex128)
    F = (1.0 / np.sqrt(Nc)) * np.exp(-1j * np.outer(kappa, R_alpha))  # (a, α)
    for mu in range(C):
        i0 = mu * Nc
        F_block[i0:i0+Nc, i0:i0+Nc] = F
    return F_block

def to_alpha_basis(H_clustered: np.ndarray, F_block: np.ndarray) -> np.ndarray:
    """H_α = F† H_clustered F.  This is still +n only (non-Hermitian)."""
    return F_block.conj().T @ H_clustered @ F_block

def extract_pairs_from_H_alpha(H_alpha_plus: np.ndarray, tol: float = 1e-12) -> List[Tuple[int,int,complex]]:
    """
    Step 5: Return all nonzero entries (i <- j, coeff) from H_α (+n only).
    We add h.c. later in the QuSpin opstrings.
    """
    Lalpha = H_alpha_plus.shape[0]
    out = []
    for i in range(Lalpha):
        for j in range(Lalpha):
            c = H_alpha_plus[i, j]
            if abs(c) > tol:
                out.append((i, j, complex(c)))
    return out

def v_term_quspin_from_pairs(pairs: List[Tuple[int,int,complex]], spinful: bool = False):
    """
    Step 6: Produce QuSpin static lists.
    For fermions, use ONLY '+-' and add h.c. by reversing indices into the SAME opstring.
    """
    hop_ij = [[complex(c), int(i), int(j)] for (i, j, c) in pairs]
    hop_ji = [[complex(np.conjugate(c)), int(j), int(i)] for (i, j, c) in pairs]
    plus_minus = [["+-", hop_ij + hop_ji]]
    if not spinful:
        return plus_minus
    else:
        # Up and down sectors identical
        return [["+-|", hop_ij + hop_ji], ["|+-", hop_ij + hop_ji]]

def compute_V_via_matrix_pipeline(
    V_separation: int,
    k_sites_supercluster: np.ndarray,   # shape (C, Nc) with global k labels
    *,
    L: int,
    V0: float = 1.0,
    R_alpha: Optional[np.ndarray] = None,
    spinful: bool = False,
    return_mats: bool = True,
    tol: float = 1e-12
) -> Dict[str, object]:
    """
    Full 6-step pipeline. Returns QuSpin opstrings and (optionally) all matrices.
    """
    #temporary hack to fix the factor of 2
    V0=V0/2
    # 1-2: +n only in k-basis
    
    Hk_plus = build_V_matrix_k_only(L, V_separation, V0=V0)
    # 3: reorder to cluster-contiguous
    Hc_plus, perm = reorder_by_clusters(Hk_plus, k_sites_supercluster)
    # 4: block DFT to α-basis
    F_block = block_alpha_transform(k_sites_supercluster, R_alpha=R_alpha)
    H_alpha_plus = to_alpha_basis(Hc_plus, F_block)
    # 5: extract pairs (no h.c.)
    pairs = extract_pairs_from_H_alpha(H_alpha_plus, tol=tol)
    # 6: assemble QuSpin hoppings with h.c.
    static_spinless = v_term_quspin_from_pairs(pairs, spinful=False)
    out = {
        "pairs": pairs,
        "to_quspin_spinless": static_spinless,
        "Nc": int(np.asarray(k_sites_supercluster).shape[1]),
        "num_clusters": int(np.asarray(k_sites_supercluster).shape[0]),
    }
    if spinful:
        out["to_quspin_spinful"] = v_term_quspin_from_pairs(pairs, spinful=True)
    if return_mats:
        H_alpha_full = H_alpha_plus + H_alpha_plus.conj().T
        out.update({
            "Hk_plus": Hk_plus,
            "Hc_plus": Hc_plus,
            "F_block": F_block,
            "H_alpha_plus": H_alpha_plus,
            "H_alpha_full": H_alpha_full,
            "perm": perm,
        })
    return out




def translate_supercluster_to_matrix_order(k_sites_supercluster: np.ndarray) -> np.ndarray:
    """
    Translate the supercluster to the matrix order.
    translate_dict=Dict[k_point]:matrix_index
    (i.e. translate the k_point key into the matrix index value)
    """
    Nc=k_sites_supercluster.shape[1]

    flat_ordering=np.zeros(np.prod(k_sites_supercluster.shape),dtype=int)
    translate_dict={}

    
    for mu in range(k_sites_supercluster.shape[0]):
        for a in range(Nc):
            flat_ordering[mu*Nc+a]=k_sites_supercluster[mu,a]
            translate_dict[k_sites_supercluster[mu,a]]=mu*Nc+a

    return flat_ordering,translate_dict

    

def v_couplings_basis(    V_separation: int,
    k_sites_supercluster: np.ndarray,   # shape = (num_clusters_in_SC, Nc)
    interaction_cluster_separation: Optional[int] = None,  # unused; kept for API
    *,
    L: Optional[int] = None,            # total number of k points (modulo)
    V0: float = 1.0,                    # overall amplitude
    spinful: bool = False,              # emit spinful QuSpin lists too
    strict: bool = True,                # error if k+n exits this supercluster
    rtol_prune: float = 0.0, atol_prune: float = 0.0,  # prune tiny coeffs in output
    atol_val: float = 1e-8             # tolerance for validation
) -> Dict[str, object]:

    """
    A more direct and (hopefully) interpretable way to implement the V-term. Four stages.
    The basic ideas it construct the SP hopping matrixA

    1. Add the V_separation in units of the k site indices (i.e. +n) to each k site.
    Only do +m because -m is the hermitian conjugate which we'll ad expicitly at the end.

    2. Express the resulting pairs as a matrix of SP hoppings. 

    3. (Optional) Swap rows and colums to make the clusters next to each other 
    (equiv. use a basis ordering that imposes this.)

    4. Tansform the resulting hopping matrix to the alpha basis as a standard basis transform.
    (Note: if you choose a basis ordering that groups the clusters this should just be SC/Nc copies
    of the alpha-basis transform.)

    5. Extract the qupsin terms as the non-zero pairings in that matrix.
    (Remember that you'll need to reorder by the intial index of the terms if you swapped to align
    clusters!)

    6. Add the hermitian conjugate of each term.

    7. Return the quspin terms!
    """

    m=_step_from_ratio(L, V_separation)
    Nc=k_sites_supercluster.shape[1]

    print(f'V_sep:{V_separation},L={L}, m:{m}')

    forward_targets=(k_sites_supercluster+m)%L

    hoppings_matrix=np.zeros((L,L),dtype=np.complex64)

    #You need to choose an ordering for the hoppings matrix.
    #Let's make the clusters next to each other.
    flat_ordering,translate_dict=translate_supercluster_to_matrix_order(k_sites_supercluster)

    for mu in range(k_sites_supercluster.shape[0]):
        for a in range(Nc):
            hoppings_matrix[translate_dict[k_sites_supercluster[mu,a]],translate_dict[forward_targets[mu,a]]]+=V0
    
    print(f'hoppings_matrix: {hoppings_matrix}')
    




    

    return None
def test_v_terms(L=4, V_0=1, int_cluster_size=2, 
                 V_separation_ratio=(1,2), int_separation_ratio=(1,2),
                 visualize=True):
    """
    Test function for V-term computation and visualization.
    
    Parameters
    ----------
    L : int
        Total number of k-points
    V_0 : float
        V coupling strength
    int_cluster_size : int
        Size of interaction clusters
    V_separation_ratio : tuple
        (p, q) for V-separation = p/q * L
    int_separation_ratio : tuple
        (p, q) for interaction cluster separation
    visualize : bool
        Whether to create visualization HTML
        
    Returns
    -------
    res : dict
        Results from compute_V_couplings_bruteforce
    """
    print("Testing hamiltonian construction")
    print(f"L={L}, V_0={V_0}, int_cluster_size={int_cluster_size}")
    print(f"V_separation_ratio={V_separation_ratio}, int_separation_ratio={int_separation_ratio}")
    
    full_clusters = generate_clusters(L, int_cluster_size, int_separation_ratio, V_separation_ratio)
    full_clusters_k = convert_site_clusters_to_k(full_clusters, L)
    print(f'full_clusters shape: {full_clusters.shape}')
    print(f'full_clusters: {full_clusters}')

    test_super_cluster_sites = full_clusters[0]

    V_sep = int(L * V_separation_ratio[0] / V_separation_ratio[1])
    int_sep = int(L * int_separation_ratio[0] / int_separation_ratio[1])
    
    res = compute_V_via_matrix_pipeline(
        V_separation=V_sep,
        k_sites_supercluster=test_super_cluster_sites,
        L=L,
        V0=V_0,
        spinful=True,
    )

    # print("Validation OK?:", res["validation"]["ok"])
    # print("Max abs error:", res["validation"]["max_abs_err"])
    # print("Per-block errors:", res["validation"]["per_block_err"])

    print("\nalpha-basis pairs (i <- j : coeff):")
    for i, j, c in res["pairs"]:
        print(f"{i} <- {j} : {c}")

    print("\nQuSpin static (spinless):", res["to_quspin_spinless"])
    print("\nQuSpin static (spinful):", res["to_quspin_spinful"])

    if visualize:
        os.makedirs('large_files/viz', exist_ok=True)
        visualize_quspin_couplings_1d_chain(
            res['to_quspin_spinful'], 
            k_sites_supercluster=test_super_cluster_sites,
            output_file='large_files/viz/v_couplings_visualization.html',
            to_quspin_spinless=res['to_quspin_spinless']
        )
        print("Visualization saved to large_files/viz/v_couplings_visualization.html")
    
    return res







##########t_tilde implementations
def cosine_dispersion(t,k):
    return 2*t*np.cos(k)


def compute_t_tilde_terms(k_sites_supercluster:np.ndarray,
                          t:float=1.0,
                          dispersion:callable=cosine_dispersion,
                          separate_mu:bool=True,
                          separate_t_tilde:bool=True,
                          ):
    """
    A function to compute all of the terms that come
    from the diagonal dispersion in the alpha basis 
    on a given supercluster. 

    Should be pretty straightforward, its basically just all
    possible pairwise hoppings although perhaps some complications
    occur if you go above Nc=3 (because you get degenerate combinations).

    The output should be the quspin hoppings list, as in compute_V_couplings_bruteforce.
    """






if __name__ == "__main__":
    print(f'testing just the t terms.')
    
    L=4
    V_0=1
    int_cluster_size=2
    V_separation_ratio=(1,4)
    int_separation_ratio=(1,2)

    supercluster=generate_clusters(L,int_cluster_size,int_separation_ratio,V_separation_ratio)[0]
    supercluster_k=convert_site_clusters_to_k(supercluster,L)

    print(f'supercluster shape: {supercluster.shape}')
    print(f'supercluster: {supercluster}')
    print(f'supercluster k: {supercluster_k/np.pi}')

    #t_terms=alpha_terms_by_separation(supercluster,supercluster_k,t=1,spin="spinless")
    
    
    # v_terms=v_couplings_basis(V_separation=V_separation_ratio,
    #                           k_sites_supercluster=supercluster,
    #                           L=L,
    #                           V0=V_0)

    # v_terms=compute_V_via_matrix_pipeline(
    #     V_separation=V_separation_ratio,
    #     k_sites_supercluster=supercluster,
    #     L=L,
    #     V0=V_0)

    test_v_terms(L=L, V_0=V_0, int_cluster_size=int_cluster_size, 
                V_separation_ratio=V_separation_ratio, int_separation_ratio=int_separation_ratio)

    
    
    

    
    
    
    # Run v_terms test with default parameters
    #test_v_terms(L=4, V_0=1, int_cluster_size=2, 
                #V_separation_ratio=(1,4), int_separation_ratio=(1,2))