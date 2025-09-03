"""
This is the base class for the FULL cluster model.
This is the top level class that should contain
everything necessary to define the cluster model.
Namely:
1. It should generate the necessary clusters and super-clusters.
2. It should generate the hamiltonian on a given (super-)cluster.
3. It should solve for the eigensystem.
4. It should collate the results up to get the thermodynamics.

"""

import numpy as np
from typing import Union
from dataclasses import dataclass


#Note - adding frozen=True makes the config immutable
#so it doesn't get overwritten by some later computation.
@dataclass
class PhysicalParams:
    U:float=0.0
    V:float=0.0
    t:float=1.0
    mu_0:float=0.0

@dataclass(frozen=True)
class ClusterModelConfig:
    
    """Initialization params:
    int_cluster_size: interaction cluster size (how many k's are clustered together)
    int_cluster_sep: intial cluster separation (the separation between the k's in the interaction cluster)
    NOTE! This is currently expressed as the denominator of L (i.e. 3 means L/3 or 2pi/3 in k-space)
    V_sep: intial V separation (the separation between terms in k space in V)
    NOTE! This is currently expressed as the denominator of L (i.e. 3 means L/3 or 2pi/3 in k-space)
    L: length of the system
    physical_params: physical parameters (optional argument to add physical parameters)
    In theory it seems better to add these after the clustering since a clustering could take many physical params.
    """

    int_cluster_size:int
    cluster_separation_ratio:tuple[int,int] # (numerator, denominator) as fraction of 2π
    V_separation_ratio:tuple[int,int] # e.g., (1, 2) means π spacing
    L:int
    physical_params:Union[PhysicalParams,None]=None
    ham_lib:str='quspin'
    model_bc:Union[str,'periodic','open']='periodic'
    int_cluster_bc:Union[str,'periodic','open']='periodic'
    super_cluster_bc:Union[str,'periodic','open']='periodic'


    def __post_init__(self):
        # Quick check that cluster_separation and V_separation are compatible with L
        if (self.L * self.cluster_separation_ratio[0]) % self.cluster_separation_ratio[1] != 0:
            raise ValueError(f"L={self.L} incompatible with cluster ratio {self.cluster_separation_ratio}")
    
        if (self.L * self.V_separation_ratio[0]) % self.V_separation_ratio[1] != 0:
            raise ValueError(f"L={self.L} incompatible with V ratio {self.V_separation_ratio}")


class ClusterModel:
    """
    This is the base class for the FULL cluster model.
    """
    def __init__(self,config:ClusterModelConfig):
        self.config=config
    
        
        
    def generate_clusters(self)->np.ndarray:
        """
        Function to generate the super-clusters.
        Should return a numpy array of shape: [L//n_supercluster,int clusters in a supercluster,int_cluster,system_dim]
        e.g. for an interacting cluster of size 2 with a V that fuses two of them,
        (as you would get for, say, int_cluster_sep=4, v_sep=2)
        on a 100 site system with the default 1d chain setup it would be
        [25,2,2,1].
        NOTE: Not sure if you should break out the int cluster and supercluster, but probably makes sense.
        returns: np.ndarray of shape: [L//n_supercluster,int clusters in a supercluster,int_cluster,system_dim]
        """
        return None
    
    def generate_hamiltonian(self,super_cluster:np.ndarray)->np.ndarray:
        """
        Function to generate the Hamiltonian on a given super-cluster.
        ham_lib: string to specify the Hamiltonian library to use.
        Currently supported: 'quspin' (default), tenpy ALTHOUGH TENPY IS BROKEN BECAUSE OF NNN issue!
        NOTE: if you go to large supercluster you may have to use sparse methods or symmetry reduction but let's leave that for now

        returns: Tuple(hamiltonian,basis)
        NOTE: Hamiltonian not yet converted to np array, use .toarray() to get it.
        """
        return None
    
        
    
    #####################################################################
    #After this point, you should just be able to use the refactored versions of your existing functions.
    #####################################################################

    def solve_hamiltonian(self,hamiltonian:object)->np.ndarray:
        """
        Function to solve the hamiltonian.
        Assumes hamiltonian input is the quspin object, not the numpy array.
        ham_shape: [(d)**super_cluster_size,d**(super_cluster_size)]
        returns: np.ndarray of shape: [d**(super_cluster_size)]
        """
        if self.config.ham_lib=='quspin':
            eigvals,eigvecs=np.linalg.eigh(hamiltonian.toarray())
        elif self.config.ham_lib=='tenpy':
            raise NotImplementedError('Tenpy solve_hamiltonian is not implemented yet')
        else:
            raise ValueError(f'Hamiltonian library {self.config.ham_lib} not implemented yet')
        return eigvals,eigvecs
    
    def get_operator_spectra(self,eigvals,eigvecs,operator:str,ham_object:object)->np.ndarray:
        """
        Function to get the spectra of common operators.
        eigvals: np.ndarray of shape: [d**(super_cluster_size)]
        eigvecs: np.ndarray of shape: [d**(super_cluster_size),d**(super_cluster_size)]
        NOTE: To be compatible with Tenpy and Quspin, I'll pass both the hamiltonian object and the basis object as one
        because quspin needs the basis object to get the operator spectra.
        As before, will extract the energy, number, and spin spectra.
        Maybe it's better to do one operator at a time?
        """
        return None
    
    def get_thermodynamic_expectation(self,cluster_grid:np.ndarray,cluster_spectra:np.ndarray,temperature:Union[None,float]=None):
        """
        Function to get the thermodynamics.
        cluster_grid: np.ndarray of shape: [L//n_supercluster,int clusters in a supercluster,int_cluster,system_dim]
        cluster_spectra: np.ndarray of shape: [L//n_supercluster,int clusters in a supercluster,int_cluster,system_dim]
        temperature: temperature to get the thermodynamics at. If None, will return zero temperature expectations.
        Don't forget to deal with the degenerate case at zero temperature!
        returns: operator_system_expectation: np.ndarray of shap equal to the cluster grid:
        [L//n_supercluster,int clusters in a supercluster]
        NOTE: I'm doing it here in every cluster so that I can see the distribution if need be.
        It would probably be nicer to get site-resolved expectations for seeing what the system looks like.
        In theory, you could expand this into a subclass for broader thermodynamic operators like compressibility etc...
        """
        return None
    
    def run_experiment(self,physical_params:object):
        """
        Function to extract the ground state expectations
        for a given configuration of physical parameters.
        physical_params: physical parameters (U,V,t,mu_0,temperature,... others)
        ham_lib: string to specify the Hamiltonian library to use.

        returns: energy_expectation: float, number_expectation: float, spin_expectation: float
        """
        return None
    
    
    
        
        
        