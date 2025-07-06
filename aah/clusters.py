import numpy as np
from dataclasses import dataclass
import itertools
import logging



logger=logging.getLogger(__name__)



# logging.basicConfig(
#     level=logging.DEBUG,                      # default level
#     format="%(levelname)s:%(name)s:%(message)s",
# )

# If you want debug output for just this module:
#logging.getLogger("aah.kblock").setLevel(logging.INFO)

class ClusterExperiment:
    def __init__(self,
                cluster_size:int,
                lattice_points:int,
                cluster_k_generator:int,
                system_dim:int=1
                ):
        self.cluster_size = cluster_size
        self.lattice_points = lattice_points
        self.cluster_k_generator = cluster_k_generator #TODO: make it take several generators
        self.system_dim = system_dim

        grid_idx=np.array(list(itertools.product(range(self.lattice_points),repeat=self.system_dim)))
        self.base_grid=-np.pi+(2*np.pi/(self.lattice_points))*grid_idx

        # --- Basic parameter validation ---
        if self.cluster_size <= 0:
            raise ValueError("cluster_size must be positive")
        if self.lattice_points <= 0:
            raise ValueError("lattice_points must be positive")
        if self.cluster_k_generator <= 0:
            raise ValueError("cluster_k_generator must be positive")

    def generate_clusters(self,return_indices:bool=False)->np.ndarray:
        """Tile the 1-D Brillouin zone into clusters.
        TODO: Extend to arbitrary dimensions.

        Each cluster is defined by ``cluster_size`` successive k-points separated by
        ``cluster_k_generator`` indices.  For a 2-site dimer, for example, the
        first cluster could be ``[k_0, k_{0+g}]`` where ``g`` is the generator.

        Returns
        -------
        np.ndarray
            Shape ``(n_clusters, cluster_size, system_dim)`` where ``system_dim``
            defaults to ``1`` (scalar k).  The array is trimmed in case the last
            cluster would overrun the grid.
        """
        
        used_indices: set[int] = set()
        cluster_list=[]
        index_list=[]

        #TODO: This is inefficient because you should be dropping idx's you've counted already.
        for start_idx in range(0, self.lattice_points):
            # Build the list of lattice indices that belong to this cluster.
            idxs = [ (start_idx + self.cluster_k_generator * i) % self.lattice_points
                     for i in range(self.cluster_size) ]

            # Skip if any of the indices has already been assigned to a cluster.
            if any(i in used_indices for i in idxs):
                continue
            
            cluster_list.append(self.base_grid[idxs])
            index_list.append(idxs)

            used_indices.update(idxs)
        
        clusters=np.stack(cluster_list,axis=0)
        index_array=np.stack(index_list,axis=0)
        
        
        # Ensure every lattice point was assigned exactly once.
        if len(used_indices) != self.lattice_points:
            # We failed to cover the Brillouin zone completely.
            raise ValueError("Cannot form complete clusters given these parameters")
        

        #logging for reference:
        logger.info(f'size test_grid: {clusters.shape}')
        logger.info(f'first grid point: {clusters[0]}')
        logger.info(f'last grid point: {clusters[-1]}')
            
        delta_k=2*np.pi/self.lattice_points
        logger.info(f'k lattice vector:{delta_k}, last grid point:{np.pi-delta_k}')
        

        if return_indices:
            return index_array
        else:
            return clusters
    

    

if __name__ == "__main__":
    cluster_experiment = ClusterExperiment(
        cluster_size=2,
        lattice_points=100,
        cluster_k_generator=25,
    )
    
    test_grid=cluster_experiment.generate_clusters()

    
        




