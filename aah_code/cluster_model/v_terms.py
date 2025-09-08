"""
Code for generating the Hamiltonian on a given supercluster entry.

"""

import numpy as np
import quspin
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
from typing import Any
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

def classify_V_on_blocks(
    L: int,
    blocks: np.ndarray | List[List[int]],   # shape (B, Nc), entries are site indices 0..L-1
    V_separation_ratio: Tuple[int,int],     # (p, q) => n = (p/q)*L
    include_minus: bool = True,
    keep_external: bool = False,            # if True, return hops that land outside this supercluster
):
    """
    Given one supercluster arranged as blocks (B x Nc), classify k -> k ± n hops
    into within-block vs between-block, and ALSO return hops indexed by the
    flattened supercluster position: flat = block_id * Nc + pos_in_block.

    Returns dict with:
      n_step : int
      within_by_block : { block_id -> int array [[from_pos, to_pos], ...] }
      between_by_blockpair : { (block_from, block_to) -> int array [[from_pos, to_pos], ...] }
      within_pairs_flat  : int array [[src_flat, dst_flat], ...]
      between_pairs_flat : int array [[src_flat, dst_flat], ...]
      (optional) externals : int array [[src_k, dst_k], ...] if keep_external=True
    """
    blocks = np.asarray(blocks, dtype=int)
    if blocks.ndim != 2:
        raise ValueError("blocks must be a 2D array of shape (B, Nc)")
    B, Nc = blocks.shape

    # map site -> (block_id, pos_in_block) for *this* supercluster
    site_to_block = np.full(L, -1, dtype=int)
    site_to_pos   = np.full(L, -1, dtype=int)
    for b in range(B):
        site_to_block[blocks[b]] = b
        site_to_pos[blocks[b]]   = np.arange(Nc, dtype=int)

    n = _step_from_ratio(L, V_separation_ratio)
    if n == 0:
        out = dict(
            n_step=n,
            within_by_block={},
            between_by_blockpair={},
            within_pairs_flat=np.empty((0,2), int),
            between_pairs_flat=np.empty((0,2), int),
        )
        if keep_external:
            out["externals"] = np.empty((0,2), int)
        return out

    # sources are exactly the sites present in this supercluster
    src = blocks.reshape(-1)

    # build destinations (+n and optionally -n), periodic wrap-around
    dst_plus  = (src + n) % L
    src_all   = src
    dst_all   = dst_plus
    if include_minus:
        dst_minus = (src - n) % L
        src_all   = np.concatenate([src_all, src])
        dst_all   = np.concatenate([dst_all, dst_minus])

    # keep only hops whose destination lies inside this supercluster (unless keeping externals)
    dst_block = site_to_block[dst_all]
    in_sc     = dst_block >= 0

    externals = None
    if keep_external:
        bad = ~in_sc
        externals = np.stack([src_all[bad], dst_all[bad]], axis=1) if np.any(bad) else np.empty((0,2), int)

    src_all   = src_all[in_sc]
    dst_all   = dst_all[in_sc]
    dst_block = dst_block[in_sc]

    src_block = site_to_block[src_all]
    src_pos   = site_to_pos[src_all]
    dst_pos   = site_to_pos[dst_all]

    # within/between masks
    within_mask = (src_block == dst_block)

    # ---- per-block (local indices) outputs (as before) ----
    within_by_block: Dict[int, List[tuple]] = {}
    for b, fp, tp in zip(src_block[within_mask], src_pos[within_mask], dst_pos[within_mask]):
        within_by_block.setdefault(int(b), []).append((int(fp), int(tp)))
    within_by_block = {b: np.array(v, dtype=int) for b, v in within_by_block.items()}

    between_by_blockpair: Dict[Tuple[int,int], List[tuple]] = {}
    for bf, bt, fp, tp in zip(src_block[~within_mask], dst_block[~within_mask],
                              src_pos[~within_mask], dst_pos[~within_mask]):
        key = (int(bf), int(bt))
        between_by_blockpair.setdefault(key, []).append((int(fp), int(tp)))
    between_by_blockpair = {k: np.array(v, dtype=int) for k, v in between_by_blockpair.items()}

    # ---- NEW: flattened supercluster indices ----
    src_flat = src_block * Nc + src_pos
    dst_flat = dst_block * Nc + dst_pos

    within_pairs_flat  = np.stack([src_flat[within_mask],  dst_flat[within_mask]],  axis=1) if np.any(within_mask) else np.empty((0,2), int)
    between_pairs_flat = np.stack([src_flat[~within_mask], dst_flat[~within_mask]], axis=1) if np.any(~within_mask) else np.empty((0,2), int)

    out = dict(
        n_step=n,
        within_by_block=within_by_block,
        between_by_blockpair=between_by_blockpair,
        within_pairs_flat=within_pairs_flat,
        between_pairs_flat=between_pairs_flat,
    )
    if keep_external:
        out["externals"] = externals
    return out

def build_V_within_quspin_couplings(
    out: Dict,
    Nc: int,
    V0: float,
    R0: np.ndarray | None = None,
    *,
    dedupe_pm: bool = False,
) -> List[List[float]]:
    """
    From classify_V_on_blocks(...) output `out`, build QuSpin couplings
    for the *within-cluster* V terms in the cluster Fourier basis (alpha indices).
    
    Returns a QuSpin-style coupling list:
      [[coeff, i, i], ...]   with i = block * Nc + alpha  (onsite terms only)

    Args
    ----
    out : dict
        Output from classify_V_on_blocks (must contain 'within_by_block').
    Nc : int
        Cluster size.
    V0 : float
        Potential amplitude (the final onsite coefficient is V0 * cos(R0[alpha] * delta)).
    R0 : array-like of shape (Nc,), optional
        Real-space positions of the cluster sites relative to the block origin.
        Defaults to np.arange(Nc) (i.e., unit lattice spacing).
    dedupe_pm : bool
        If True, dedupe ±(t1-t2) so cos() is not double-counted.

    Notes
    -----
    - If a block has multiple within-hops with different (t1-t2), their
      contributions are summed for that block & alpha.
    - If out['within_by_block'] is empty, returns an empty list.
    """
    within = out.get("within_by_block", {})
    if not within:
        return []  # nothing within-block → no onsite contributions

    if R0 is None:
        R0 = np.arange(Nc, dtype=float)
    else:
        R0 = np.asarray(R0, dtype=float)
        if R0.shape != (Nc,):
            raise ValueError(f"R0 must have shape ({Nc},), got {R0.shape}")

    # Accumulate onsite coefficients per (block, alpha)
    # coeffs[b, alpha] = sum_over_unique_d  V0 * cos(R0[alpha] * 2π d / Nc)
    two_pi_over_Nc = 2.0 * np.pi / float(Nc)
    coeffs_per_block = {}  # b -> np.ndarray shape (Nc,)

    for b, pairs in within.items():
        pairs = np.asarray(pairs, dtype=int)
        if pairs.size == 0:
            continue

        # Differences d = (t1 - t2) mod Nc
        d = (pairs[:, 0] - pairs[:, 1]) % Nc
        if dedupe_pm:
            # cos(θ) = cos(-θ) → fold d and (Nc - d) together; ignore d=0 duplicates
            d = np.unique(np.minimum(d, (-d) % Nc))
        else:
            d = np.unique(d)

        # Remove d=0 if present; that corresponds to "no hop"
        d = d[d % Nc != 0]
        if d.size == 0:
            continue

        # Sum contributions over unique d
        coeff_alpha = np.zeros(Nc, dtype=float)
        for di in d:
            delta = two_pi_over_Nc * float(di)
            coeff_alpha += V0 * np.cos(R0 * delta)

        coeffs_per_block[b] = coeff_alpha

    # Build QuSpin on-site coupling list [[coeff, i, i], ...], flatten as block * Nc + alpha
    couplings: List[List[float]] = []
    for b, vec in coeffs_per_block.items():
        for alpha, c in enumerate(vec):
            if c != 0.0:
                i = b * Nc + alpha
                couplings.append([float(c), int(i), int(i)])

    return couplings


#for now let's just construct each term separately and then add them all together
# at the end. The V term is probably going to be the most complicated, so let's try that first
def make_v_hamiltonian(V:float,
                    V_separation:Tuple[int,int],
                    supercluster_k:np.ndarray,
                    supercluster_idxs:np.ndarray,
                    L:int,#You need to specify L because you have wrap-around in the clustering (is this dealt with already by the clusters? - ah I think kinda. The clusters are constructed with this knowledge but you need it to know if V steps to another cluster)
                    V_separation_ratio:Tuple[int,int],
                    bc:str="periodic",
                    dtype:np.dtype=np.float64)->quspin.operators.hamiltonian:
    
    """
    Just construct the V term. The tricky part here
    is the within/between cluster behaviour. For V
    within a cluster, after the alpha transformation,
    its an onsite term. Where the V hopping goes between clusters
    its going to be a hopping term.
    Don't forget that the clustering is done in k-space, BEFORE any
    alpha transformation.


    Returns
    -------
    H : quspin.operators.hamiltonian
    basis : quspin.basis.spinful_fermion_basis_1d
    """
    # basis over spin-↑ and spin-↓ fermions; Nf fixes (N_up, N_down) sector if given


    total_cluster_size=int(np.prod(supercluster_k.shape))
    print(f'total_cluster_size: {total_cluster_size}')

    basis = spinful_fermion_basis_1d(total_cluster_size,Nf=None,double_occupancy=True)
    
    static=[]
    
    #First I need to derive the within cluster terms and the between cluster terms.
    supercluster_plus_v_steps=supercluster_k+V_separation
    supercluster_minus_v_steps=supercluster_k-V_separation

    #check which steps are within an interacting cluster and which steps take you between clusters:
    out = classify_V_on_blocks(L, supercluster_idxs, V_separation_ratio)
    print(f'out: {out}')

    return True


    # if bc == "periodic":
    #     sub_cluster_nn_bonds = [(within_cluster_indices[i], (within_cluster_indices[(i+1)%cluster_size])) for i in range(len(within_cluster_indices))]
    #     print(f'sub_cluster_nn_bonds: {sub_cluster_nn_bonds}')
    #     #sub_cluster_nnn_bonds = [(i, (i + 2) % cluster_size) for i in range(cluster_size)]
    # elif bc == "open":
    #     sub_cluster_nn_bonds = [(i, i + 1) for i in (within_cluster_indices - 1)]
    #     #sub_cluster_nnn_bonds = [(i, i + 2) for i in range(cluster_size - 2)]
    # else:
    #     raise ValueError("bc must be 'open' or 'periodic'")

    # #Add t_tilde within-cluster coupling terms:
    # t_tilde=get_t_tilde(t_0,basis_class)
    
    # print(f't_tilde: {t_tilde}')
    # sub_hop_NN_pm = [[t_tilde, i, j] for (i, j) in sub_cluster_nn_bonds]  # "+-" terms (c†_i c_j)
    # sub_hop_NN_mp = [[-t_tilde, i, j] for (i, j) in sub_cluster_nn_bonds]  # "-+" terms (c_i c†_j)
    
    # static.append(["+-|", sub_hop_NN_pm])
    # static.append(["-+|", sub_hop_NN_mp])
    # static.append(["|+-", sub_hop_NN_pm])
    # static.append(["|-+", sub_hop_NN_mp])

    # H = hamiltonian(static, [], basis=basis, dtype=dtype)

    # return H, basis

# def compute_V_couplings_bruteforce(
#     V_separation: int,
#     k_sites_supercluster: np.ndarray,   # shape = (num_clusters_in_SC, Nc)
#     interaction_cluster_separation: Optional[int] = None,  # unused; kept for API parity
#     *,
#     L: Optional[int] = None,            # total number of k points (modulo)
#     V0: float = 1.0,                    # overall amplitude
#     R_alpha: Optional[np.ndarray] = None,  # positions inside cluster; default [0..Nc-1]
#     spinful: bool = False,              # emit spinful QuSpin lists too
#     strict: bool = True,                # error if k+n exits this supercluster
#     rtol_prune: float = 0.0, atol_prune: float = 0.0,  # prune tiny coeffs in output
#     validate: bool = False,             # run permutation-conjugation validator
#     atol_val: float = 1e-12             # tolerance for validation comparisons
# ) -> Dict[str, object]:
#     """
#     Brute-force a-sum for the V-modulation term in the *canonical* finite-cluster alpha basis.
#     Implements (V0/2) * sum_k c†_{k+n} c_k + h.c.

#     Inputs
#     ------
#     V_separation : int
#         n in k-steps (i.e. Delta = 2*pi*n/L); hop k -> k + n (mod L).
#     k_sites_supercluster : (C, Nc) int array
#         Global k labels for one *supercluster*, organized by interaction clusters (first dim)
#         and within-cluster indices a=0..Nc-1 (second dim).
#     interaction_cluster_separation : int or None
#         Present for signature parity; not used.
#     L : int or None
#         Total number of global k points (period). Pass explicitly for correctness.
#         If None, uses max(k)+1, which only works if this SC reaches the max label.
#     V0 : float
#         Overall scale for V.
#     R_alpha : (Nc,) array or None
#         Real-space positions for alpha transform; default np.arange(Nc).
#     spinful : bool
#         If True, also produce spinful QuSpin static op lists.
#     strict : bool
#         If True, raise if k+n not found in provided supercluster; else skip.
#     rtol_prune, atol_prune : float
#         Prune |coeff| <= atol + rtol*row_norm in the flattened pair list.
#     validate : bool
#         If True, compute per-block P-matrices and check T ≈ (V0/2)*U^† P U.
#     atol_val : float
#         Absolute tolerance for validation.
#     """
#     k_sites = np.asarray(k_sites_supercluster, dtype=int)
#     num_clusters, Nc = k_sites.shape

#     if L is None:
#         L = int(k_sites.max()) + 1  # heuristic; better to pass L

#     if R_alpha is None:
#         R_alpha = np.arange(Nc, dtype=float)
#     else:
#         R_alpha = np.asarray(R_alpha, dtype=float)
#         assert R_alpha.shape == (Nc,)

#     # Canonical finite-cluster k-grid (same in each cluster)
#     kappa = 2.0 * np.pi * np.arange(Nc, dtype=float) / float(Nc)

#     # Map: global k -> (mu, a)
#     k_to_mu_a: Dict[int, Tuple[int, int]] = {}
#     for mu in range(num_clusters):
#         for a in range(Nc):
#             k = int(k_sites[mu, a])
#             if k in k_to_mu_a:
#                 raise ValueError(f"Duplicate k={k} found in this supercluster.")
#             k_to_mu_a[k] = (mu, a)

#     # Blocks (mu' <- mu) in alpha basis: T^{(mu->mu')} \in C^{Nc x Nc}
#     blocks: Dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros((Nc, Nc), dtype=np.complex128))

#     # Optional permutation matrices for validation
#     perms: Dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros((Nc, Nc), dtype=np.int8)) if validate else {}

#     # Prefactor from the two DFTs: (V0/2)*(1/Nc)
#     pref = (V0 / 2.0) * (1.0 / float(Nc))

#     # Cache exponentials
#     exp_cache_alpha: Dict[int, np.ndarray] = {}  # b -> e^{+i kappa_b R_alpha}
#     exp_cache_beta:  Dict[int, np.ndarray] = {}  # a -> e^{-i kappa_a R_beta} (reuse R_alpha shape)

#     for mu in range(num_clusters):
#         for a in range(Nc):
#             k = int(k_sites[mu, a])
#             k_prime = (k + V_separation) % L
#             hit = k_to_mu_a.get(k_prime, None)
#             if hit is None:
#                 if strict:
#                     raise ValueError(
#                         f"k'={k_prime} from (mu={mu}, a={a}, k={k}) not in this supercluster. "
#                         "Provide the full fused SC or set strict=False to skip."
#                     )
#                 else:
#                     continue

#             mu_prime, b = hit

#             # e^{+i kappa_b R_alpha}
#             if b not in exp_cache_alpha:
#                 exp_cache_alpha[b] = np.exp(1j * kappa[b] * R_alpha)
#             phase_alpha = exp_cache_alpha[b]  # (Nc,)

#             # e^{-i kappa_a R_beta}
#             if a not in exp_cache_beta:
#                 # use R_alpha just for vector length Nc
#                 exp_cache_beta[a] = np.exp(-1j * kappa[a] * R_alpha)
#             phase_beta = exp_cache_beta[a]   # (Nc,)

#             # Outer product over (alpha, beta)
#             contrib = pref * np.outer(phase_alpha, phase_beta)  # (Nc, Nc)
#             blocks[(mu_prime, mu)] += contrib

#             # Record permutation for validation
#             if validate:
#                 perms[(mu_prime, mu)][b, a] += 1  # should stay 0/1 for disjoint mapping

#     # Flatten to (i, j, coeff) using i = mu' * Nc + alpha, j = mu * Nc + beta
#     pairs: List[Tuple[int, int, complex]] = []
#     for (mu_p, mu), T in blocks.items():
#         # Optional pruning
#         if (atol_prune > 0.0) or (rtol_prune > 0.0):
#             row_norm = np.max(np.sum(np.abs(T), axis=1)) if T.size else 0.0
#             mask = np.abs(T) > (atol_prune + rtol_prune * row_norm)
#         else:
#             mask = np.ones_like(T, dtype=bool)

#         for alpha in range(Nc):
#             i = mu_p * Nc + alpha
#             for beta in range(Nc):
#                 if not mask[alpha, beta]:
#                     continue
#                 j = mu * Nc + beta
#                 coeff = T[alpha, beta]
#                 if coeff != 0.0:
#                     pairs.append((i, j, complex(coeff)))

#     # Build QuSpin static lists (spinless and optional spinful).
#     hop_ij = [[complex(c), int(i), int(j)] for (i, j, c) in pairs]
#     hop_ji_hc = [[complex(np.conjugate(c)), int(j), int(i)] for (i, j, c) in pairs]

#     to_quspin_spinless = [
#         ["+-", hop_ij],
#         ["-+", hop_ji_hc],
#     ]

#     out = {
#         "pairs": pairs,
#         "blocks": dict(blocks),
#         "Nc": Nc,
#         "num_clusters": num_clusters,
#         "to_quspin_spinless": to_quspin_spinless,
#     }

#     if spinful:
#         to_quspin_spinful = [
#             ["+-|", hop_ij],    # up
#             ["-+|", hop_ji_hc],
#             ["|+-", hop_ij],    # down
#             ["|-+", hop_ji_hc],
#         ]
#         out["to_quspin_spinful"] = to_quspin_spinful

#     # ===== Validation: T ?= (V0/2) * U^† P U =====
#     if validate:
#         # Build U once
#         U = (1.0 / np.sqrt(Nc)) * np.exp(-1j * np.outer(kappa, R_alpha))  # shape (Nc, Nc)
#         Udag = np.conjugate(U.T)

#         per_block_err = {}
#         max_abs_err = 0.0
#         ok = True

#         for key, T_bf in blocks.items():
#             P = perms[key]
#             # Check P sanity: each column has at most one 1; entries are 0/1
#             if not np.all((P == 0) | (P == 1)):
#                 raise ValueError(f"Permutation for block {key} has entries not in {{0,1}}.")
#             if np.any(np.sum(P, axis=0) > 1):
#                 raise ValueError(f"Permutation for block {key} maps some source 'a' to multiple targets.")

#             T_perm = (V0 / 2.0) * (Udag @ P @ U)
#             diff = T_bf - T_perm
#             err = float(np.max(np.abs(diff)))
#             per_block_err[key] = err
#             max_abs_err = max(max_abs_err, err)
#             if err > atol_val:
#                 ok = False

#         out["validation"] = {
#             "ok": ok,
#             "max_abs_err": max_abs_err,
#             "per_block_err": per_block_err,
#             "atol": atol_val,
#         }
#         # Also expose the raw permutations if you want to inspect
#         out["permutations"] = dict(perms)

#     return out


def compute_V_couplings_bruteforce(
    V_separation: int,
    k_sites_supercluster: np.ndarray,   # shape = (num_clusters_in_SC, Nc)
    interaction_cluster_separation: Optional[int] = None,  # unused; kept for API
    *,
    L: Optional[int] = None,            # total number of k points (modulo)
    V0: float = 1.0,                    # overall amplitude
    R_alpha: Optional[np.ndarray] = None,  # positions inside cluster; default [0..Nc-1]
    spinful: bool = False,              # emit spinful QuSpin lists too
    strict: bool = True,                # error if k+n exits this supercluster
    rtol_prune: float = 0.0, atol_prune: float = 0.0,  # prune tiny coeffs in output
    validate: bool = False,             # run permutation-conjugation validator
    atol_val: float = 1e-12             # tolerance for validation
) -> Dict[str, object]:
    """
    Brute-force a-sum for the V-modulation term in the *canonical* finite-cluster alpha basis.
    Implements (V0/2) * sum_k c†_{k+n} c_k + h.c.

    Returns dict with:
      - 'pairs': [(i,j,coeff)] for c_i^† c_j (no h.c. here)
      - 'to_quspin_spinless': [["+-", ...]] with h.c. folded into the SAME opstring
      - 'to_quspin_spinful' (if spinful=True): [["+-|", ...], ["|+-", ...]]
      - 'blocks': {(mu',mu): T_block} α-basis blocks
      - 'validation' (if validate=True): ok/errs
    """
    k_sites = np.asarray(k_sites_supercluster, dtype=int)
    num_clusters, Nc = k_sites.shape

    if L is None:
        L = int(k_sites.max()) + 1  # heuristic; pass L explicitly if possible

    if R_alpha is None:
        R_alpha = np.arange(Nc, dtype=float)
    else:
        R_alpha = np.asarray(R_alpha, dtype=float)
        assert R_alpha.shape == (Nc,)

    # Canonical finite-cluster k-grid (same in each cluster)
    kappa = 2.0 * np.pi * np.arange(Nc, dtype=float) / float(Nc)

    # Map: global k -> (mu, a)
    k_to_mu_a: Dict[int, Tuple[int, int]] = {}
    for mu in range(num_clusters):
        for a in range(Nc):
            k = int(k_sites[mu, a])
            if k in k_to_mu_a:
                raise ValueError(f"Duplicate k={k} found in this supercluster.")
            k_to_mu_a[k] = (mu, a)

    # Blocks (mu' <- mu) in alpha basis: T^{(mu->mu')} \in C^{Nc x Nc}
    blocks: Dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros((Nc, Nc), dtype=np.complex128))

    # Optional permutation matrices for validation
    perms: Dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros((Nc, Nc), dtype=np.int8)) if validate else {}

    # Prefactor from the two DFTs: (V0/2)*(1/Nc)
    pref = (V0 / 2.0) * (1.0 / float(Nc))

    # Cache exponentials
    exp_cache_alpha: Dict[int, np.ndarray] = {}  # b -> e^{+i kappa_b R_alpha}
    exp_cache_beta:  Dict[int, np.ndarray] = {}  # a -> e^{-i kappa_a R_beta} (reuse R_alpha shape)

    for mu in range(num_clusters):
        for a in range(Nc):
            k = int(k_sites[mu, a])
            k_prime = (k + V_separation) % L
            hit = k_to_mu_a.get(k_prime, None)
            if hit is None:
                if strict:
                    raise ValueError(
                        f"k'={k_prime} from (mu={mu}, a={a}, k={k}) not in this supercluster. "
                        "Provide the full fused SC or set strict=False to skip."
                    )
                else:
                    continue

            mu_prime, b = hit

            # e^{+i kappa_b R_alpha}
            if b not in exp_cache_alpha:
                exp_cache_alpha[b] = np.exp(1j * kappa[b] * R_alpha)
            phase_alpha = exp_cache_alpha[b]  # (Nc,)

            # e^{-i kappa_a R_beta}
            if a not in exp_cache_beta:
                # use R_alpha just for vector length Nc
                exp_cache_beta[a] = np.exp(-1j * kappa[a] * R_alpha)
            phase_beta = exp_cache_beta[a]   # (Nc,)

            # Outer product over (alpha, beta)
            contrib = pref * np.outer(phase_alpha, phase_beta)  # (Nc, Nc)
            blocks[(mu_prime, mu)] += contrib

            if validate:
                perms[(mu_prime, mu)][b, a] += 1  # should be 0/1 (disjoint mapping)

    # Flatten to (i, j, coeff) using i = mu' * Nc + alpha, j = mu * Nc + beta
    pairs: List[Tuple[int, int, complex]] = []
    for (mu_p, mu), T in blocks.items():
        # Optional pruning
        if (atol_prune > 0.0) or (rtol_prune > 0.0):
            row_norm = np.max(np.sum(np.abs(T), axis=1)) if T.size else 0.0
            mask = np.abs(T) > (atol_prune + rtol_prune * row_norm)
        else:
            mask = np.ones_like(T, dtype=bool)

        for alpha in range(Nc):
            i = mu_p * Nc + alpha
            for beta in range(Nc):
                if not mask[alpha, beta]:
                    continue
                j = mu * Nc + beta
                coeff = T[alpha, beta]
                if coeff != 0.0:
                    pairs.append((i, j, complex(coeff)))

    # ---- QuSpin assembly: ONLY use "+-" and add h.c. by reversing pairs in the SAME opstring.
    hop_ij = [[complex(c), int(i), int(j)] for (i, j, c) in pairs]
    hop_ji_hc = [[complex(np.conjugate(c)), int(j), int(i)] for (i, j, c) in pairs]
    hop_total = hop_ij + hop_ji_hc

    to_quspin_spinless = [
        ["+-", hop_total],  # all directed edges in "+-" (includes h.c. explicitly)
    ]

    out = {
        "pairs": pairs,                      # bare c_i^† c_j matrix elements
        "blocks": dict(blocks),              # α-basis blocks
        "Nc": Nc,
        "num_clusters": num_clusters,
        "to_quspin_spinless": to_quspin_spinless,
    }

    if spinful:
        # Up-spin "+-|" and Down-spin "|+-" — both add h.c. by reversing indices
        to_quspin_spinful = [
            ["+-|", hop_total],  # up spin
            ["|+-", hop_total],  # down spin
        ]
        out["to_quspin_spinful"] = to_quspin_spinful

    # ===== Validation: T ?= (V0/2) * U^† P U =====
    if validate:
        U = (1.0 / np.sqrt(Nc)) * np.exp(-1j * np.outer(kappa, R_alpha))  # shape (Nc, Nc)
        Udag = np.conjugate(U.T)

        per_block_err = {}
        max_abs_err = 0.0
        ok = True

        for key, T_bf in blocks.items():
            P = perms[key]
            if not np.all((P == 0) | (P == 1)):
                raise ValueError(f"Permutation for block {key} has entries not in {{0,1}}.")
            if np.any(np.sum(P, axis=0) > 1):
                raise ValueError(f"Permutation for block {key} maps some source 'a' to multiple targets.")
            T_perm = (V0 / 2.0) * (Udag @ P @ U)
            diff = T_bf - T_perm
            err = float(np.max(np.abs(diff)))
            per_block_err[key] = err
            max_abs_err = max(max_abs_err, err)
            if err > atol_val:
                ok = False

        out["validation"] = {
            "ok": ok,
            "max_abs_err": max_abs_err,
            "per_block_err": per_block_err,
            "atol": atol_val,
        }
        out["permutations"] = dict(perms)

    return out

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
    
    res = compute_V_couplings_bruteforce(
        V_separation=V_sep,
        k_sites_supercluster=test_super_cluster_sites,
        L=L,
        V0=V_0,
        spinful=True,
        validate=True,
        atol_val=1e-8
    )

    print("Validation OK?:", res["validation"]["ok"])
    print("Max abs error:", res["validation"]["max_abs_err"])
    print("Per-block errors:", res["validation"]["per_block_err"])

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


    
    
    
    # Run v_terms test with default parameters
    test_v_terms(L=4, V_0=1, int_cluster_size=2, 
                V_separation_ratio=(1,4), int_separation_ratio=(1,2))