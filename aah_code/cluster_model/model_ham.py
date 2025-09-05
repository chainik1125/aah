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



def sparse_diagonalize(H, k=16, which='SA', return_eigenvectors=True):
    """
    Diagonalize a sparse Hamiltonian using sparse eigensolvers.
    
    Parameters
    ----------
    H : quspin hamiltonian or scipy.sparse matrix
        The Hamiltonian to diagonalize
    k : int
        Number of eigenvalues/eigenvectors to compute (default: 16)
    which : str
        Which eigenvalues to find: 'SA' (smallest algebraic, default), 
        'LA' (largest algebraic), 'SM' (smallest magnitude), etc.
    return_eigenvectors : bool
        Whether to return eigenvectors (default: True)
    
    Returns
    -------
    eigenvalues : np.ndarray
        The k lowest eigenvalues
    eigenvectors : np.ndarray (if return_eigenvectors=True)
        The corresponding eigenvectors
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix
    import time
    
    # Convert to sparse matrix if it's a quspin hamiltonian
    if hasattr(H, 'tocsr'):
        H_sparse = H.tocsr()
    elif hasattr(H, 'toarray'):
        # If it's already sparse-like but not csr
        H_sparse = csr_matrix(H.toarray())
    else:
        # Assume it's already a sparse matrix
        H_sparse = H
    
    # Ensure k is not larger than matrix dimension - 1
    n_dim = H_sparse.shape[0]
    k_actual = min(k, n_dim - 1)
    
    if k_actual < k:
        print(f"Warning: Requested k={k} but matrix dimension is {n_dim}. Using k={k_actual}")
    
    # Use sparse eigenvalue solver
    if return_eigenvectors:
        eigenvalues, eigenvectors = eigsh(H_sparse, k=k_actual, which=which, return_eigenvectors=True)
        # Sort by eigenvalue
        idx = eigenvalues.argsort()
        return eigenvalues[idx], eigenvectors[:, idx]
    else:
        eigenvalues = eigsh(H_sparse, k=k_actual, which=which, return_eigenvectors=False)
        return np.sort(eigenvalues)


def benchmark_sparse_vs_dense(H, k=16):
    """
    Compare performance of sparse vs dense eigensolvers.
    
    Parameters
    ----------
    H : quspin hamiltonian
        The Hamiltonian to benchmark
    k : int
        Number of eigenvalues for sparse solver
    
    Returns
    -------
    dict
        Dictionary with timing and eigenvalue results
    """
    import time
    
    results = {}
    
    # Dense diagonalization
    print("Running dense diagonalization...")
    start_time = time.time()
    H_dense = H.toarray()
    eigvals_dense, _ = np.linalg.eigh(H_dense)
    dense_time = time.time() - start_time
    results['dense_time'] = dense_time
    results['dense_eigvals'] = eigvals_dense[:k]  # First k eigenvalues
    
    print(f"Dense diagonalization took {dense_time:.3f} seconds")
    
    # Sparse diagonalization
    print(f"Running sparse diagonalization (k={k})...")
    start_time = time.time()
    eigvals_sparse, _ = sparse_diagonalize(H, k=k)
    sparse_time = time.time() - start_time
    results['sparse_time'] = sparse_time
    results['sparse_eigvals'] = eigvals_sparse
    
    print(f"Sparse diagonalization took {sparse_time:.3f} seconds")
    
    # Compare results
    speedup = dense_time / sparse_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Check accuracy
    max_diff = np.max(np.abs(results['dense_eigvals'] - results['sparse_eigvals']))
    print(f"Maximum eigenvalue difference: {max_diff:.2e}")
    
    results['speedup'] = speedup
    results['max_diff'] = max_diff
    
    return results


if __name__ == "__main__":
    L=12
    Nc=3
    t=1.0
    V=2.0
    U=0.0
    mu_0=0.0
    
    int_sep_ratio=(1,6)
    v_sep_ratio=(1,6)

    supercluster_idxs=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)[0]
    supercluster_k=convert_site_clusters_to_k(supercluster_idxs,L)

    print(f'supercluster_k: {supercluster_k/np.pi}')
    print(f'supercluster_idxs: {supercluster_idxs}')
    

    H, basis=make_cluster_ham(supercluster_k,supercluster_idxs,t,V,U,mu_0,L,Nc,int_sep_ratio,v_sep_ratio)

    print(f'H shape: {H.toarray().shape}')

    benchmark_results = benchmark_sparse_vs_dense(H, k=16)

    # One-line test: Uncomment to run sparse vs dense benchmark
    # benchmark_results = benchmark_sparse_vs_dense(H, k=32)
    print(f'benchmark_results: {benchmark_results}')
    
    # Or just run sparse diagonalization for the lowest k eigenvalues
    # eigvals_sparse, eigvecs_sparse = sparse_diagonalize(H, k=32)
    # print(f'Lowest 32 eigenvalues (sparse): {eigvals_sparse[:10]}')  # Show first 10
    
    # Original dense diagonalization (comment out for large systems)
    #eigvals,eigvecs=np.linalg.eigh(H.toarray())
    #print(f'eigvals: {eigvals.shape}')


