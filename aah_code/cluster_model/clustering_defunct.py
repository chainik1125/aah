"""
Functions no longer used, just for reference.
"""


from typing import List, Tuple
from math import gcd
from fractions import Fraction
import numpy as np
import logging
logger=logging.getLogger(__name__)

def generate_clusters_defunct(
    L: int,
    Nc: int,
    cluster_separation_ratio: Tuple[int, int],  # (p, q) meaning delta = 2π * p/q
    V_separation_ratio: Tuple[int, int],  # (p, q) meaning delta = 2π * p/q
):
    """
    Return a list of clusters (each a list of site indices in +m order),
    implementing the 'interaction tiling' (no ±n edges yet).

    Preconditions (enforced):
      - m = (p/q) * L must be integer  (i.e., (p*L) % q == 0)
      - Nc must divide q_m = L / gcd(L, m)

    Example:
      L=12, Nc=3, sep=(3,4) -> m = 9; q_m = 12/gcd(12,9)=12/3=4, Nc|q_m ✓
      Clusters are directed 3-segments along +9 (i.e., -3) steps.
    """
    if Nc <= 0 or L <= 0:
        raise ValueError("L and Nc must be positive integers.")
    p, q = cluster_separation_ratio
    if q == 0:
        raise ValueError("cluster_separation_ratio denominator q must be nonzero.")

    # Exact step m on the lattice
    frac_m = Fraction(p, q) * L
    if frac_m.denominator != 1:
        raise ValueError(
            f"Step m = L*(p/q) = {frac_m} is not integer; need q | L (or more generally (p*L) % q == 0)."
        )
    m = frac_m.numerator % L

    if m == 0:
        if Nc == 1:
            return [[j] for j in range(L)]
        raise ValueError("m ≡ 0 (mod L): orbit length is 1, cannot build Nc>1 clusters.")

    g = gcd(L, m) #g is the number of cosets
    qm = L // g  # qm is the number of elements (sites) in each coset
    if qm % Nc != 0:
        raise ValueError(
            f"Cannot tile: Nc={Nc} does not divide q_m=L/gcd(L,m)={qm} (L={L}, m={m}, g={g})."
        )

    int_clusters: List[List[int]] = []
    blocks_per_orbit = qm // Nc

    
    # For each orbit representative a, place seeds spaced by Nc*m steps
    for a in range(g):
        for b in range(blocks_per_orbit):
            seed = (a + (b * Nc * m)) % L #i.e. choose a coset representative.
            cluster = [(seed + t * m) % L for t in range(Nc)] #iterate over the coset elements (clusters)
            int_clusters.append(cluster)

    # (Optional) safety checks in debug contexts:
    # covered = {site for C in clusters for site in C}
    # assert len(clusters) == L // Nc and len(covered) == L
    #
    #My attempt to implement the V fusion:

    #first extract the steps:
    a,b=V_separation_ratio
    if b==0:
        raise ValueError("V_separation_ratio denominator b must be nonzero.")
    
    frac_n=Fraction(a,b)*L
    if frac_n.denominator != 1:
        raise ValueError(f"Step n = L*(b/a) = {frac_n} is not integer; need a | L (or more generally (b*L) % a == 0).")
    n_step=frac_n.numerator % L
    print(f"Step n = {n_step}")

    #I think a simple way to think about this is that
    #g_mn is the smallest `gap` in between the clusters
    #when you arrange them on the line.
    #Almost - its not so much a 'gap' as a 'stride' or step length.
    #Think of the superclusters as an arithmetic progrssion
    #with stride equal to g_mn.
    #The g_mn is the spacing within a supercluster, so indexing over these is what
    # gets you the seeds.
    
    g_mn=gcd(L,m,n_step)
    q_mn=L//g_mn
    
    #Issue a warning if V is coprime with L since that should mean that it will fuse everything
    superblocks_per_orbit=q_mn//Nc
    # supercluster_sites: List[List[int]] = [[(r + k * g) % L for k in range(superblocks_per_orbit)]
    #                                        for r in range(g)]
    
    superclusters_blocks: List[List[List[int]]] = [[] for _ in range(g_mn)]
    
    for C in int_clusters:
        r=C[0] % g_mn # pick any member; they all have the same residue, note you're using the residue as the index in the list.
        superclusters_blocks[r].append(C) # put the whole +m block into that residue class

    superclusters_array=np.array(superclusters_blocks)

    logger.info(f"Interaction cluster shape (k-site indices): {superclusters_array.shape}")
    logger.info(f"Interaction first cluster (k-site indices): {superclusters_array[0]}")
    if len(superclusters_array) > 2:
        logger.info(f"Interaction second cluster (k-site indices): {superclusters_array[1]}")
    
    logger.info(f"Interaction last cluster (k-site indices): {superclusters_array[-1]}")
    
    return superclusters_array

def generate_clusters_defunct_v2(L:int,int_cluster_size:int,cluster_separation_ratio:tuple[int,int],V_separation_ratio:tuple[int,int]):
    """
    Generate the clusters.
    I think we can work in site indices and then convert them to
    their k-space values.
    TODO: not sure what to do about situations like L=12, cluster_sep=(3,4) (i.e. 12,9) since you get points
    that are 12-9=3 apart going the "other" way.
    Maybe you should restrict this so you can't do that? Is the solution to have bi-directional hopping?
    But then again the chain _is_ genuinely periodic, so maybe that is correct.

    TODO: Generalize to higher dimensions.
    """
    
    #First get the interaction tiling:
    int_cluster_index_separation=int(L*cluster_separation_ratio[0]/cluster_separation_ratio[1])
    print(f"Interaction cluster index separation: {int_cluster_index_separation}")
    remaining_sites=[i for i in range(L)]
    clustered_sites=[]
    while len(remaining_sites) >= int_cluster_size:
        start_site=remaining_sites[0]
        for i in range(int_cluster_size):
            cluster_sites=[start_site]
            append_site_forward=start_site+int_cluster_index_separation*i
            append_site_backward=start_site+(L-int_cluster_index_separation*i)
            if append_site_forward in remaining_sites:
                cluster_sites.append(append_site_forward)
                remaining_sites.remove(append_site_forward)
            elif append_site_backward in remaining_sites:
                cluster_sites.append(append_site_backward)
                remaining_sites.remove(append_site_backward)
            else:
                raise ValueError(f"Needed to use a site that had already been added - clustering fails.")
        print(f"Cluster sites: {cluster_sites}")
        clustered_sites.append(cluster_sites)
                
    if len(remaining_sites) > 0:
        raise ValueError(f"Not all sites were clustered. {remaining_sites}")
    
    clustered_sites=np.array(clustered_sites)

    logger.info(f"Interaction cluster shape (k-site indices): {clustered_sites.shape}")
    logger.info(f"Interaction first cluster (k-site indices): {clustered_sites[0]}")
    logger.info(f"Interaction second cluster (k-site indices): {clustered_sites[1]}")
    logger.info(f"Interaction last cluster (k-site indices): {clustered_sites[-1]}")



    #Now we need to do the V fusion:

    return None

