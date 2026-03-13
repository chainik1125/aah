"""
Script for making the hamiltonian.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import quspin
from quspin.basis import spinful_fermion_basis_1d
from aah_code.quspin_utils import hamiltonian
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
#from aah_code.cluster_model.t_tilde import alpha_terms_by_separation
from aah_code.hamiltonian import benchmark_sparse_vs_dense
from aah_code.hamiltonian import sparse_diagonalize
from aah_code.cluster_model.v_terms_test import compute_V_via_matrix_pipeline
from aah_code.cluster_model.t_tilde_test import alpha_terms_by_separation


@dataclass
class SymmetryBlock:
    """Container describing a symmetry-preserving Hamiltonian sector."""
    Nup: int
    Ndn: int
    hamiltonian: hamiltonian
    basis: spinful_fermion_basis_1d


@dataclass
class SymmetryDecomposedHamiltonian:
    """Lightweight container that yields symmetry blocks lazily."""
    static_terms: List
    super_cluster_size: int
    dtype: object = np.complex64

    def iter_blocks(self):
        for n_up in range(self.super_cluster_size + 1):
            for n_dn in range(self.super_cluster_size + 1):
                basis_sector = spinful_fermion_basis_1d(
                    self.super_cluster_size,
                    Nf=(n_up, n_dn),
                    double_occupancy=True
                )
                if basis_sector.Ns == 0:
                    continue
                H_sector = hamiltonian(self.static_terms, [], basis=basis_sector, dtype=self.dtype)
                yield SymmetryBlock(Nup=n_up, Ndn=n_dn, hamiltonian=H_sector, basis=basis_sector)


def _build_static_terms(supercluster_ks,
                        supercluster_idxs,
                        t,
                        V,
                        U,
                        mu_0,
                        L,
                        Nc,
                        int_sep_ratio,
                        v_sep_ratio) -> Tuple[List, int]:
    """Construct quspin static terms and return them along with cluster size."""
    static = []
    super_cluster_size = int(np.prod(supercluster_ks.shape))

    print(f"Creating basis for super_cluster_size: {super_cluster_size}")
    print(f"Supercluster k shape: {supercluster_ks.shape}")
    print(f"Supercluster indices shape: {supercluster_idxs.shape}")

    V_sep = int(L * v_sep_ratio[0] / v_sep_ratio[1])
    int_sep = int(L * int_sep_ratio[0] / int_sep_ratio[1])

    v_terms = compute_V_via_matrix_pipeline(
        V_separation=V_sep,
        k_sites_supercluster=supercluster_idxs,
        L=L,
        V0=V,
        spinful=True
    )

    static.extend(v_terms["to_quspin_spinful"])

    t_tilde_terms = alpha_terms_by_separation(supercluster_idxs, supercluster_ks, t, spin='spinful')
    t_terms_static: List = []
    for s in sorted(t_tilde_terms.keys()):
        t_terms_static.extend(t_tilde_terms[s])
    static.extend(t_terms_static)

    # Add diagonal U
    U_list = [[U, i, i] for i in range(super_cluster_size)]
    static.append(["n|n", U_list])

    # Add onsite mu_0
    mu_0_list = [[-mu_0, i] for i in range(super_cluster_size)]
    static.append(["n|", mu_0_list])
    static.append(["|n", mu_0_list])

    return static, super_cluster_size


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
                    ham_lib='quspin',
                    use_symm=True):
    """Construct the cluster Hamiltonian, optionally enabling symmetry decomposition."""

    static_terms, super_cluster_size = _build_static_terms(
        supercluster_ks,
        supercluster_idxs,
        t,
        V,
        U,
        mu_0,
        L,
        Nc,
        int_sep_ratio,
        v_sep_ratio
    )

    basis = spinful_fermion_basis_1d(super_cluster_size)

    if use_symm:
        symm_container = SymmetryDecomposedHamiltonian(static_terms, super_cluster_size, dtype=np.complex64)
        return symm_container, basis

    H = hamiltonian(static_terms, [], basis=basis, dtype=np.complex64)
    return H, basis




#Need to recrete the current t\=0, V\=0 error...
#Ah you should write a func to make the k-space non-interacting hamiltonian...

def minimal_t_V_n3_failure():
    L=6
    Nc=3
    t=1.0
    V=1.0
    U=0
    mu_0=0

    v_sep=(1,6)
    int_sep_matched=(1,6) #int_sep=(1,6) neither works
    int_sep_unmatched=(1,3)

    supercluster_idxs_matched=generate_clusters(L,Nc,int_sep_matched,v_sep)[0]
    supercluster_k_matched=convert_site_clusters_to_k(supercluster_idxs_matched,L)

    supercluster_idxs_unmatched=generate_clusters(L,Nc,int_sep_unmatched,v_sep)[0]
    supercluster_k_unmatched=convert_site_clusters_to_k(supercluster_idxs_unmatched,L)

    print(f"Supercluster idxs matched shape: {supercluster_idxs_matched.shape}")
    print(f"Supercluster idxs matched: {supercluster_idxs_matched}")
    print(f"Supercluster k matched shape: {supercluster_k_matched.shape}")
    print(f"Supercluster k matched: {supercluster_k_matched/np.pi}")

    print(f"Supercluster idxs unmatched shape: {supercluster_idxs_unmatched.shape}")
    print(f"Supercluster idxs unmatched: {supercluster_idxs_unmatched}")
    print(f"Supercluster k unmatched shape: {supercluster_k_unmatched.shape}")
    print(f"Supercluster k unmatched: {supercluster_k_unmatched/np.pi}")

    H_matched, basis=make_cluster_ham(supercluster_k_matched,supercluster_idxs_matched,t,V,U,mu_0,L,Nc,int_sep_matched,v_sep)
    H_unmatched, basis=make_cluster_ham(supercluster_k_unmatched,supercluster_idxs_unmatched,t,V,U,mu_0,L,Nc,int_sep_unmatched,v_sep)

    print(f"H_matched shape: {H_matched.toarray().shape}")
    print(f"H_unmatched shape: {H_unmatched.toarray().shape}")

    H_matched_eigvals, H_matched_evecs=sparse_diagonalize(H_matched.toarray(),k=6)
    H_unmatched_eigvals, H_unmatched_evecs=sparse_diagonalize(H_unmatched.toarray(),k=6)

    print(f"H_matched eigvals: {H_matched_eigvals}")
    print(f"H_unmatched eigvals: {H_unmatched_eigvals}")

    # t_terms_static=alpha_terms_by_separation(supercluster_idxs_matched,supercluster_k_matched,t,spin='spinless')
    
    # v_terms=compute_V_via_matrix_pipeline(
    #     V_separation=v_sep,
    #     k_sites_supercluster=supercluster_idxs,
    #     L=L,
    #     V0=V,
    #     spinless=True
    # )

    # v_terms_static=v_terms["to_quspin_spinless"]


    return None


if __name__ == "__main__":

    minimal_t_V_n3_failure()
    exit()

    L=6
    Nc=3
    t=1.0
    V=0.0
    U=0.0
    mu_0=0.0
    
    int_sep_ratio=(1,3)
    v_sep_ratio=(1,6)

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
