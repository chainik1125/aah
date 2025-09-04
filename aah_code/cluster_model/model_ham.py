"""
Script for making the hamiltonian.
"""

from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import quspin
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from aah_code.cluster_model.t_tilde import alpha_terms_by_separation
from aah_code.cluster_model.v_terms import compute_V_couplings_bruteforce

def make_cluster_ham(supercluster_ks,
                    supercluster_idxs,
                    t,
                    V,
                    U,
                    mu_0,
                    L,
                    Nc,
                    int_sep_ratio,
                    v_sep_ratio,
                    ham_lib='quspin'):

    static=[]

    super_cluster_size=int(np.prod(supercluster_ks.shape))
    print(f"Creating basis for super_cluster_size: {super_cluster_size}")
    print(f"Supercluster k shape: {supercluster_ks.shape}")
    print(f"Supercluster indices shape: {supercluster_idxs.shape}")
    basis=spinful_fermion_basis_1d(super_cluster_size)

    V_sep = int(L * v_sep_ratio[0] / v_sep_ratio[1])
    int_sep = int(L * int_sep_ratio[0] / int_sep_ratio[1])

    v_terms = compute_V_couplings_bruteforce(
        V_separation=V_sep,
        k_sites_supercluster=supercluster_idxs,
        L=L,
        V0=V,
        spinful=True,
        validate=True,
        atol_val=1e-8
    )

    v_terms_static=v_terms["to_quspin_spinful"]
    
    
    static.extend(v_terms_static)

    t_tilde_terms=alpha_terms_by_separation(supercluster_idxs,supercluster_ks,t,spin='spinful')
    

    t_terms_static = []
	
    for s in sorted(t_tilde_terms.keys()):
        t_terms_static.extend(t_tilde_terms[s])

    static.extend(t_terms_static)


    
    #Add diagonal U
    U_list = [[U, i, i] for i in range(super_cluster_size)]
    static.append(["n|n", U_list])

    #Add onsite mu_0
    mu_0_list = [[-mu_0, i] for i in range(super_cluster_size)]
    static.append(["n|", mu_0_list])
    static.append(["|n", mu_0_list])

    H = hamiltonian(static, [], basis=basis, dtype=np.complex64)

    return H,basis



if __name__ == "__main__":
    L=8
    Nc=2
    t=1.0
    V=0.0
    U=0.0
    mu_0=0.0
    
    int_sep_ratio=(1,4)
    v_sep_ratio=(1,2)

    supercluster_idxs=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)[0]
    supercluster_k=convert_site_clusters_to_k(supercluster_idxs,L)

    print(f'supercluster_k: {supercluster_k/np.pi}')
    print(f'supercluster_idxs: {supercluster_idxs}')
    

    H, basis=make_cluster_ham(supercluster_k,supercluster_idxs,t,V,U,mu_0,L,Nc,int_sep_ratio,v_sep_ratio)

    print(f'H shape: {H.toarray().shape}')



    