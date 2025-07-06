from clusters import ClusterExperiment







def main():
    cluster_experiment = ClusterExperiment(
        cluster_size=2,
        lattice_points=100,
        cluster_k_generator=25,
    )
    
    test_grid=cluster_experiment.generate_clusters()




if __name__ == "__main__":
    main()

    