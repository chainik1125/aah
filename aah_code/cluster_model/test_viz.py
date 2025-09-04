#!/usr/bin/env python3
"""Test script to verify visualization with proper k-value labeling"""

import numpy as np
from clustering import generate_clusters, convert_site_clusters_to_k
from model_ham import compute_V_couplings_bruteforce
from visualization_helpers import visualize_quspin_couplings_1d_chain
import os

print("Testing visualization with L=12")

L = 12
V_0 = 1
int_cluster_size = 3
V_separation_ratio = (1, 6)
int_separation_ratio = (1, 3)

# Generate clusters
full_clusters = generate_clusters(L, int_cluster_size, int_separation_ratio, V_separation_ratio)
full_clusters_k = convert_site_clusters_to_k(full_clusters, L)

print(f'Full clusters shape: {full_clusters.shape}')
print(f'Full clusters (sites): \n{full_clusters}')
print(f'Full clusters (k-values): \n{full_clusters_k}')

# Use first supercluster for testing
test_super_cluster_sites = full_clusters[0]
test_super_cluster_k = full_clusters_k[0]

V_sep = int(L * V_separation_ratio[0] / V_separation_ratio[1])

# Compute V couplings
res = compute_V_couplings_bruteforce(
    V_separation=V_sep,
    k_sites_supercluster=test_super_cluster_sites,
    L=L,
    V0=V_0,
    spinful=True,
    validate=True,
    atol_val=1e-8
)

print(f"\nValidation OK?: {res['validation']['ok']}")
print(f"Number of pairs: {len(res['pairs'])}")

# Create visualization
os.makedirs('large_files/viz', exist_ok=True)
visualize_quspin_couplings_1d_chain(
    res['to_quspin_spinful'], 
    k_sites_supercluster=test_super_cluster_sites,
    output_file='large_files/viz/test_k_labeling.html',
    to_quspin_spinless=res['to_quspin_spinless']
)

print("\nVisualization created at large_files/viz/test_k_labeling.html")
print("Check that:")
print("1. Sites are labeled with their k-values (not 0,1,2...)")
print("2. Sites are colored by cluster (same cluster = same color)")
print("3. Sites are positioned according to their k-values on the x-axis")