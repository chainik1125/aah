"""
Code for generating the Hamiltonian on a given supercluster entry.

"""

import numpy as np
import quspin
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from typing import Tuple, Dict, List
from collections import defaultdict
from typing import Any
from aah_code.cluster_model.clustering import step_from_ratio
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from fractions import Fraction




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


if __name__ == "__main__":
    print("testing hamiltonian construction")
    L=4
    V_0=1
    int_cluster_size=2

    V_separation_ratio=(1,2)
    int_separation_ratio=(1,4)

    full_clusters=generate_clusters(L,int_cluster_size,int_separation_ratio,V_separation_ratio)
    full_clusters_k=convert_site_clusters_to_k(full_clusters,L)
    print(f'full_clusters shape: {full_clusters.shape}')
    
    print(f'full_clusters: {full_clusters}')
    
    make_v_hamiltonian(V_0,V_separation_ratio,
                    supercluster_k=full_clusters_k[0],
                    supercluster_idxs=full_clusters[0],
                    L=L,
                    V_separation_ratio=(1,2),
                    bc="periodic",
                    dtype=np.float64)