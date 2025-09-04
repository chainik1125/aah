#!/usr/bin/env python3
"""Simple test to verify visualization k-value labeling"""

import numpy as np
from visualization_helpers import visualize_quspin_couplings_1d_chain
import os

print("Testing visualization with simple example")

# Create a test supercluster with specific k-values
# Simulating interleaved clusters as expected from your clustering
# For L=12, cluster_size=3, we might have:
# Cluster 0: k-values [0, 4, 8]
# Cluster 1: k-values [1, 5, 9]  
# Cluster 2: k-values [2, 6, 10]
# Cluster 3: k-values [3, 7, 11]

test_super_cluster_sites = np.array([
    [0, 4, 8],   # Cluster 0
    [1, 5, 9],   # Cluster 1
    [2, 6, 10],  # Cluster 2
    [3, 7, 11]   # Cluster 3
])

print(f'Test supercluster:\n{test_super_cluster_sites}')

# Create dummy coupling data for visualization
# Using spinful format: [["+-|", hop_ij], ["-+|", hop_ji_hc], ["|+-", hop_ij], ["|-+", hop_ji_hc]]
dummy_couplings = []

# Add some test couplings between different sites
# Format: [coefficient, target_i, source_j]
test_hops = [
    [0.5+0j, 0, 1],   # From flat index 1 to 0
    [0.5+0j, 3, 4],   # From flat index 4 to 3
    [0.3+0j, 6, 7],   # From flat index 7 to 6
    [0.3+0j, 9, 10],  # From flat index 10 to 9
]

# Create hermitian conjugate pairs
test_hops_hc = [[np.conj(c), j, i] for c, i, j in test_hops]

to_quspin_spinful = [
    ["+-|", test_hops],      # spin up
    ["-+|", test_hops_hc],   # spin up h.c.
    ["|+-", test_hops],      # spin down
    ["|-+", test_hops_hc],   # spin down h.c.
]

to_quspin_spinless = [
    ["+-", test_hops + test_hops_hc],  # All hops in one operator
]

# Create visualization
os.makedirs('large_files/viz', exist_ok=True)
visualize_quspin_couplings_1d_chain(
    to_quspin_spinful, 
    k_sites_supercluster=test_super_cluster_sites,
    output_file='large_files/viz/test_k_labeling_simple.html',
    to_quspin_spinless=to_quspin_spinless,
    title='Test: K-value Labeling and Cluster Coloring'
)

print("\nVisualization created at large_files/viz/test_k_labeling_simple.html")
print("\nExpected behavior:")
print("1. Sites should be labeled: 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11")
print("   (the actual k-values from the supercluster)")
print("2. Sites should be positioned on x-axis according to k-value")
print("   (so k=0 at left, k=11 at right, properly spaced)")
print("3. Cluster colors:")
print("   - Cluster 0 (k=0,4,8): one color")
print("   - Cluster 1 (k=1,5,9): another color")
print("   - Cluster 2 (k=2,6,10): third color")
print("   - Cluster 3 (k=3,7,11): fourth color")