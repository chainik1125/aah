from pathlib import Path
import yaml
from pydantic import BaseModel
import numpy as np

from log import logger

from global_params import HamiltonianParams


def calculate_dispersion_from_hopping(hamiltonian_params:HamiltonianParams):
    dispersion_dict={}
    for hopping_key,hopping_tuple in hamiltonian_params.hoppings.items():
        hopping_separation=hopping_tuple[1]
        hopping_value=hopping_tuple[0]

        def dispersion(k):
            return 2*hopping_value*np.cos(k*hopping_separation)
        
        dispersion_dict[hopping_key]=dispersion
    
    return dispersion_dict

#for now, let me just stick to the NN hopping case

def cosine_dispersion(k_point):
    return 2*np.cos(k_point)

def mu_tilde_coefficient(hopping,cluster_basis,dispersion=cosine_dispersion):
    """
    The onsite hopping coefficient for the (\delta_k=0) term in the alpha basis.
    
    """

    n_cluster_sites=cluster_basis.cluster_k_points.shape[0]
    nearest_neighbor_hopping=hopping

    return (1/(n_cluster_sites))*np.array([nearest_neighbor_hopping*dispersion(k) for k in cluster_basis.cluster_k_points]).sum()

# def t_tilde(cluster_basis,dispersion=cosine_dispersion):
#     """
#     The nearest neighbor hopping term in the alpha basis.
#     """
#     n_cluster_sites=cluster_basis.n_cluster_sites.shape[0]

#     return (1/(n_cluster_sites))*np.array([dispersion(k) for k in cluster_basis.cluster_k_points]).sum()