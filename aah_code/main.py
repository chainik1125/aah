from aah_code.clusters import ClusterExperiment
from aah_code.hamiltonian import HamiltonianParams,Hubbard1D,FullSpectrum,SpectrumSolver
from aah_code.real_space_dmrg import RealSpaceHubbard1D






def main():
    cluster_experiment = ClusterExperiment(
        cluster_size=2,
        lattice_points=100,
        cluster_k_generator=25,
    )
    
    test_grid=cluster_experiment.generate_clusters()




if __name__ == "__main__":
    main()

    