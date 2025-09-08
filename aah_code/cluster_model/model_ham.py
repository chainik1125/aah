"""
Script for making the hamiltonian.
"""

from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import quspin
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
#from aah_code.cluster_model.t_tilde import alpha_terms_by_separation
from aah_code.cluster_model.v_terms import compute_V_couplings_bruteforce
from aah_code.hamiltonian import benchmark_sparse_vs_dense,sparse_diagonalize
from aah_code.cluster_model.v_terms_test import compute_V_via_matrix_pipeline
from aah_code.cluster_model.t_tilde_test import alpha_terms_by_separation



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

    # v_terms = compute_V_couplings_bruteforce(
    #     V_separation=V_sep,
    #     k_sites_supercluster=supercluster_idxs,
    #     L=L,
    #     V0=V,
    #     spinful=True,
    #     validate=True,
    #     atol_val=1e-8
    # )

    v_terms=compute_V_via_matrix_pipeline(
        V_separation=V_sep,
        k_sites_supercluster=supercluster_idxs,
        L=L,
        V0=V,
        spinful=True
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
    L=4
    Nc=2
    t=0.0
    V=2.0
    U=0.0
    mu_0=0.0
    
    int_sep_ratio=(1,2)
    v_sep_ratio=(1,4)

    supercluster_idxs=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)[0]
    supercluster_k=convert_site_clusters_to_k(supercluster_idxs,L)

    print(f'supercluster_k: {supercluster_k/np.pi}')
    print(f'supercluster_idxs: {supercluster_idxs}')
    

    H, basis=make_cluster_ham(supercluster_k,supercluster_idxs,t,V,U,mu_0,L,Nc,int_sep_ratio,v_sep_ratio)

    print(f'H shape: {H.toarray().shape}')

    
    
    
    #benchmark_results = benchmark_sparse_vs_dense(H, k=16)

    # One-line test: Uncomment to run sparse vs dense benchmark
    # benchmark_results = benchmark_sparse_vs_dense(H, k=32)
    #print(f'benchmark_results: {benchmark_results}')
    
    # Or just run sparse diagonalization for the lowest k eigenvalues
    # eigvals_sparse, eigvecs_sparse = sparse_diagonalize(H, k=32)
    # print(f'Lowest 32 eigenvalues (sparse): {eigvals_sparse[:10]}')  # Show first 10
    
    # Original dense diagonalization (comment out for large systems)
    #eigvals,eigvecs=np.linalg.eigh(H.toarray())
    #print(f'eigvals: {eigvals.shape}')


