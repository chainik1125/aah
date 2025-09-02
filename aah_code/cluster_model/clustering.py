"""
Code that implements the clustering functionality.
"""

import numpy as np
from math import gcd
from fractions import Fraction
from typing import List, Tuple, Union
from functools import reduce

import logging

logger=logging.getLogger(__name__)

def gcd_many(*vals: int) -> int:
    return reduce(gcd, vals)


def cluster_validation(L:int,int_cluster_size:int,cluster_separation_ratio:tuple[int,int],V_separation_ratios:Union[List[tuple[int,int]],Tuple[int,int]]):
    """
    First test if the separations are compatible with L.
    """
    #first test if the interaction tiling works:
    int_index_separation=int(L*cluster_separation_ratio[0]/cluster_separation_ratio[1])
    interaction_cluster_cycle_length=L/gcd(L,int_index_separation)
    

    if (interaction_cluster_cycle_length % int_cluster_size) != 0:
        raise ValueError(f"Interaction cluster size {int_cluster_size} is not compatible with L={L} and cluster separation ratio {cluster_separation_ratio} (index separation {int_index_separation})")
    else:
        #return the size and cycle length of the interaction cluster
        logger.info(f"Interaction cluster size {int_cluster_size} is compatible with L={L} and cluster separation ratio {cluster_separation_ratio} (index separation {int_index_separation})")
    #Now test if the super-clusters work when 

    #now test if the V tiling works:
    if isinstance(V_separation_ratios, tuple):
        V_separation_ratios = [V_separation_ratios]

    for V_separation_ratio in V_separation_ratios:
        v_index_separation=int(L*V_separation_ratio[0]/V_separation_ratio[1])
    v_separations=[int(L*V_separation_ratio[0]/V_separation_ratio[1]) for V_separation_ratio in V_separation_ratios]
    
    v_cluster_size=L/gcd_many(L, int_index_separation, *v_separations)

    logger.info(f"Clusters compatible. \n\
        Interaction cluster size: {interaction_cluster_cycle_length} \n\
        V cluster size: {v_cluster_size} \n\
        Interaction index separation: {int_index_separation} \n\
        V index separation: {v_index_separation} \n\
        Fused cluster size: {v_cluster_size} \n\
        ")
    


def step_from_ratio(L: int, ratio: Tuple[int,int]) -> int:
    p, q = ratio
    if q == 0:
        raise ValueError("denominator must be nonzero")
    val = Fraction(p, q) * L
    if val.denominator != 1:
        raise ValueError(f"(p/q)*L = {val} not integer; need (p*L) % q == 0")
    return val.numerator % L

def tiling_checks(L: int, Nc: int, m: int) -> tuple[int,int,int]:
    if L <= 0 or Nc <= 0: raise ValueError("L, Nc > 0")
    if m % L == 0:
        if Nc == 1:
            return (L, 1, 1)  # g1, qm, blocks_per_orbit
        raise ValueError("m ≡ 0 (mod L) with Nc>1")
    g1 = gcd(L, m)
    qm = L // g1
    if qm % Nc != 0:
        raise ValueError(f"Nc={Nc} does not divide q_m={qm}")
    return g1, qm, qm // Nc

def build_int_clusters(L: int, Nc: int, m: int, g1: int, blocks_per_orbit: int) -> List[List[int]]:
    clusters = []
    for a in range(g1):
        for b in range(blocks_per_orbit):
            seed = (a + b * Nc * m) % L
            clusters.append([(seed + t * m) % L for t in range(Nc)])
    return clusters



def fuse_by_hops(clusters: List[List[int]], L: int, m: int, ns: List[int]) -> tuple[int, List[List[List[int]]]]:
    g = gcd_many(L, m, *ns)   # #components
    groups = [[] for _ in range(g)]
    for C in clusters:
        groups[C[0] % g].append(C)  # each +m block stays in one residue mod g
    return g, groups



def generate_clusters(L: int, Nc: int, m_ratio: Tuple[int,int], v_ratios: Union[List[Tuple[int,int]],Tuple[int,int]] = ()):
    #Logic works for multiple V hoppings
    #but for now its nicer to add it as a single entry.
    if isinstance(v_ratios, tuple):
        v_ratios = [v_ratios]
    m = step_from_ratio(L, m_ratio)
    g1, qm, bpo = tiling_checks(L, Nc, m)
    int_clusters = build_int_clusters(L, Nc, m, g1, bpo)
    #Note - this deals with the case of 
    ns = [step_from_ratio(L, vr) for vr in v_ratios]
    #finds the superclusters that fuse under V
    g, superclusters = fuse_by_hops(int_clusters, L, m, ns)
    
    return np.array(superclusters)
    # return {
    #     "m": m, "g1": g1, "qm": qm, "blocks_per_orbit": bpo,
    #     "clusters": int_clusters, "ns": ns, "g": g,
    #     "blocks_per_supercluster": (L // g) // Nc,
    #     "superclusters_blocks": np.array(superclusters),
    # }

def convert_site_clusters_to_k(site_clusters:np.ndarray)->np.ndarray:
    """
    Convert the site clusters to k-space clusters.
    """
    k_clusters=-np.pi+(2*np.pi/(L))*site_clusters
    
        
    return k_clusters
    
        



    
        
    


if __name__ == "__main__":
    # Configure logging to see output
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG to see more detail
        format='%(levelname)s - %(message)s'
    )
    
    print('testing clustering')

    L=8
    int_cluster_size=2
    cluster_separation_ratio=(1,8)
    V_separation_ratio=(1,4)

    cluster_validation(L,int_cluster_size,cluster_separation_ratio,V_separation_ratio)
    test_clusters=generate_clusters(L,int_cluster_size,cluster_separation_ratio,V_separation_ratio)

    print(f"Test clusters superclusters shape: {test_clusters.shape}")
    print(f"Test clusters superclusters: {test_clusters}")

    print(f"clusters in k-space (pi multiples): {convert_site_clusters_to_k(test_clusters)/np.pi}")

    