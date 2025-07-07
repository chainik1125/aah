import numpy as np
from dataclasses import dataclass
import itertools
import logging
import tenpy as tp
from tenpy.models import lattice
from global_params import StatesParams



logger=logging.getLogger(__name__)




class LocalClusterBasis:
    """
    This class is used to generate the information for the 
    local cluster basis. It will get fed into the Hamiltonian
    generator to generate the cluster Hamiltonian.

    I'll try to have the option to output this in either TenPy or 
    in an enumerated list of the basis states for the Hamiltonian.
    """
    def __init__(self,
                cluster_k_points:np.ndarray,
                cluster_state_params:StatesParams,
                ):
        
            self.cluster_k_points=cluster_k_points
            self.cluster_state_params=cluster_state_params

    # Initialize d.o.f on each site
    def init_sites(self):
        if self.cluster_state_params.spin_states==2:
            site=tp.networks.site.SpinHalfFermionSite(cons_N=None, cons_Sz=None)
        #TODO: Add better logic for sublattice and orbital sites
        else:
            raise ValueError(f"Have only implemented spin-1/2 fermions as local basis for now - will do more later!")
        return site

    # Set 1D lattice
    def init_cluster_lattice(self):
        L = self.cluster_k_points.shape[0] # number of k point sites
        bc = self.cluster_state_params.cluster_chain_boundary_conditions  # always use 'open'
        bc_MPS = self.cluster_state_params.cluster_mps_boundary_conditions  # 'infinite' does iDMRG (still use open in 'bc' because periodic will break it.)
        lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites())
        return lat
    
    def enumerate_single_site_basis(self,lattice):
        mps_sites=lattice.mps_sites()
        single_site_labels = mps_sites[0].state_labels

        return single_site_labels
    
    
    


        
    

    

if __name__ == "__main__":
    print('the main character')
    
    state_params=StatesParams(spin_states=2)
    cluster_k_points=np.array([[-np.pi],
                                [0]])
    
    print(f'cluster_k_points shape:{cluster_k_points.shape}')

    test_basis=LocalClusterBasis(cluster_k_points,state_params)

    local_lattice=test_basis.init_cluster_lattice()

    print(f'local_lattice self.pairs keys:{local_lattice.pairs.keys()}')

    print(f' can I get all pairs? {local_lattice.pairs['nearest_neighbors']}')
    exit()
    

    print(vars(local_lattice).keys())
    print(f'local lattice unit cell:{local_lattice.unit_cell_positions}')

    for alpha in range(len(local_lattice.unit_cell)):
        for u_alpha in local_lattice.unit_cell_positions:
            for beta in range(alpha+1,len(local_lattice.unit_cell)):
                for u_beta in local_lattice.unit_cell_positions:
                    dx=(alpha+u_alpha)-(beta+u_beta)
                    print(f'alpha:{alpha}, u_alpha:{u_alpha}, beta:{beta}, u_beta:{u_beta}, dx:{dx}')
    




    

   

    
        




