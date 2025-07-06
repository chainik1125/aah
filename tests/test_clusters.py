import pytest
import numpy as np
from aah.aah.clusters import ClusterExperiment


class TestClusterExperiment:
    def test_generate_clusters_all_1_points_appear_once(self):
        """Test that ALL k points from base_grid appear exactly once in clusters."""
        cluster_experiment = ClusterExperiment(
            cluster_size=2,
            lattice_points=8,
            cluster_k_generator=1,
        )
        
        clusters = cluster_experiment.generate_clusters()
        
        # Flatten the clusters to get all k-points
        all_k_points = clusters.flatten()
        
        # Get all k-points from base_grid
        base_k_points = cluster_experiment.base_grid.flatten()
        
        # Sort both arrays for comparison
        all_k_points_sorted = np.sort(all_k_points)
        base_k_points_sorted = np.sort(base_k_points)
        
        # Check that ALL k-points from base_grid appear exactly once
        np.testing.assert_array_almost_equal(
            all_k_points_sorted, 
            base_k_points_sorted,
            err_msg="Not all k-points from base_grid appear exactly once in clusters"
        )

    def test_generate_clusters_different_parameters_1d(self):
        """Test ALL k-points appear exactly once for 1D systems with different parameters."""
        test_cases = [
            {"cluster_size": 2, "lattice_points": 8, "cluster_k_generator": 1},  # Simple adjacent clustering
            {"cluster_size": 4, "lattice_points": 8, "cluster_k_generator": 1},  # 2 clusters of 4
            {"cluster_size": 2, "lattice_points": 6, "cluster_k_generator": 3},  # Non-adjacent clustering
        ]
        
        for params in test_cases:
            cluster_experiment = ClusterExperiment(**params)
            clusters = cluster_experiment.generate_clusters()
            
            # Check that ALL k-points from base_grid appear exactly once
            all_k_points = np.sort(clusters.flatten())
            base_k_points = np.sort(cluster_experiment.base_grid.flatten())
            
            np.testing.assert_array_almost_equal(
                all_k_points, 
                base_k_points,
                err_msg=f"Not all k-points appear exactly once for parameters {params}"
            )
            
    def test_generate_clusters_raises_error_incomplete_clustering(self):
        """Test that ValueError is raised when clustering parameters don't allow all k-points to be used."""
        # This configuration should fail because it can't cluster all 10 points
        with pytest.raises(ValueError, match="Cannot form complete clusters"):
            cluster_experiment = ClusterExperiment(
                cluster_size=2,
                lattice_points=10,
                cluster_k_generator=3,
            )
            cluster_experiment.generate_clusters()

