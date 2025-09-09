#!/usr/bin/env python3
"""
Test sparse vs dense Hamiltonian diagonalization performance.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aah'))

from aah_code.cluster_model.model_ham import make_cluster_ham, benchmark_sparse_vs_dense
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k

# Test parameters - using same as model_ham.py main block
L = 12
Nc = 3  # Larger cluster for bigger Hamiltonian
t = 1.0
V = 2.0
U = 0.0
mu_0 = 0.0
int_sep_ratio = (1, 6)
v_sep_ratio = (1, 6)

# Generate proper supercluster using the same method as model_ham.py
supercluster_idxs = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)[0]
supercluster_k = convert_site_clusters_to_k(supercluster_idxs, L)

print("=" * 60)
print("Testing Sparse vs Dense Hamiltonian Diagonalization")
print("=" * 60)
print(f"System parameters:")
print(f"  L={L}, Nc={Nc}")
print(f"  t={t}, V={V}, U={U}, mu_0={mu_0}")
print(f"  int_sep_ratio={int_sep_ratio}, v_sep_ratio={v_sep_ratio}")

# Create the Hamiltonian
print("\nCreating Hamiltonian...")
H, basis = make_cluster_ham(
    supercluster_k, 
    supercluster_idxs, 
    t, V, U, mu_0, L, Nc, 
    int_sep_ratio, 
    v_sep_ratio
)

print(f"Hamiltonian shape: {H.toarray().shape}")
print(f"Basis dimension: {basis.Ns}")

# Benchmark sparse vs dense
print("\n" + "=" * 60)
print("Benchmarking Sparse vs Dense Diagonalization")
print("=" * 60)

# Test with different values of k
k_values = [4, 8, 16, 32]

for k in k_values:
    if k >= basis.Ns:
        print(f"\nSkipping k={k} (larger than basis dimension {basis.Ns})")
        continue
        
    print(f"\nTesting with k={k} eigenvalues:")
    print("-" * 40)
    
    results = benchmark_sparse_vs_dense(H, k=k)
    
    print(f"\nResults for k={k}:")
    print(f"  Dense time: {results['dense_time']:.3f}s")
    print(f"  Sparse time: {results['sparse_time']:.3f}s")
    print(f"  Speedup: {results['speedup']:.2f}x")
    print(f"  Max eigenvalue difference: {results['max_diff']:.2e}")
    
    # Show first few eigenvalues
    print(f"\n  First {min(5, k)} eigenvalues:")
    print(f"    Dense:  {results['dense_eigvals'][:5]}")
    print(f"    Sparse: {results['sparse_eigvals'][:5]}")

print("\n" + "=" * 60)
print("Benchmark Complete!")
print("=" * 60)