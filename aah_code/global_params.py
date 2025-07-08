from dataclasses import dataclass
from typing import Optional,Callable,Dict,Tuple,Any,List

"""
This code is meant to contain all the parameters that are defined globally.
This is the physical basis of the underlying states (spin, sublattice, etc ...)
and any other things that are common to everything
"""


@dataclass
class StatesParams:
    """
    This class is used to define the physical basis of the underlying states.
    """
    spin_states:int=2,
    sublattice_states:Optional[int]=None,
    orbital_states:Optional[int]=None,
    cluster_chain_boundary_conditions:Optional[str]='periodic',
    cluster_mps_boundary_conditions:Optional[str]='finite'

@dataclass
class HamiltonianParams:
    """
    This class just gives the physical parameters of the ORIGINAL, REAL SPACE, Hamiltonian.
    The Hamiltonian class will take these and do the conversions to the alpha basis
    there - right now implemented by hand. 
    """
    #hoppings are give as param name as key and then
    #the tuple T, (value,real space lattice vectors separation)
    
    #hoppings:Dict[str,Tuple[float,int]]={'t':(1,1)}
    #TODO: Add generalized hoppings later
    U:float
    V:float
    hopping:float=1.0
    mu_0:float=0.0



                
    
        
        
        
        
        


    
    