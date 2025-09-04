"""
Custom FullSpectrum class for the general cluster method.
This integrates with the new clustering approach and custom Hamiltonian construction.
"""

import numpy as np
from typing import Union, Tuple, Optional
from dataclasses import dataclass
from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from aah_code.cluster_model.model_ham import make_cluster_ham
from aah_code.cluster_model.run_scripts_simple import make_simple_cluster_ham
from quspin.operators import hamiltonian


@dataclass
class GeneralClusterParams:
    """Parameters for general cluster method."""
    L: int  # Total system size
    Nc: int  # Cluster size
    int_sep_ratio: Tuple[int, int]  # Interaction cluster separation
    v_sep_ratio: Tuple[int, int]  # V-term separation
    use_simple_ham: bool = True  # Use simplified Hamiltonian (V=0) for now


@dataclass  
class PhysicalParams:
    """Physical parameters matching the original HamiltonianParams."""
    U: float
    V: float  
    hopping: float  # t parameter
    mu_0: float


class FullSpectrumCustom:
    """
    Custom FullSpectrum class that works with the general cluster method.
    Follows the same interface as the original FullSpectrum class.
    """
    
    def __init__(
        self, 
        cluster_params: GeneralClusterParams,
        physical_params: PhysicalParams,
        temperature: Union[None, float] = None,
        ham_lib: str = 'quspin'
    ):
        self.cluster_params = cluster_params
        self.physical_params = physical_params
        self.temperature = temperature
        self.ham_lib = ham_lib
        
        # Generate all superclusters
        self.all_superclusters = generate_clusters(
            cluster_params.L, 
            cluster_params.Nc,
            cluster_params.int_sep_ratio,
            cluster_params.v_sep_ratio
        )
        self.num_superclusters = self.all_superclusters.shape[0]
        
        print(f"Initialized FullSpectrumCustom with {self.num_superclusters} superclusters")
        print(f"Supercluster shape: {self.all_superclusters.shape}")
        
    def get_full_spectrum(self, return_ham: bool = False):
        """
        Get the full spectrum for all superclusters.
        Returns the same format as original FullSpectrum.get_full_spectrum()
        """
        k_points = []
        energy_spectrum = []
        number_spectrum = []
        spin_spectrum = []
        ham_objects = []
        
        for sc_idx in range(self.num_superclusters):
            # Get this supercluster
            supercluster_idxs = self.all_superclusters[sc_idx]
            supercluster_k = convert_site_clusters_to_k(
                self.all_superclusters[sc_idx:sc_idx+1], 
                self.cluster_params.L
            )[0]
            
            # Create Hamiltonian
            if self.cluster_params.use_simple_ham:
                # Use simplified version (V=0)
                H, basis = make_simple_cluster_ham(
                    supercluster_k,
                    supercluster_idxs,
                    self.physical_params.hopping,
                    self.physical_params.V,
                    self.physical_params.U,
                    self.physical_params.mu_0,
                    self.cluster_params.L,
                    self.cluster_params.Nc,
                    self.cluster_params.int_sep_ratio,
                    self.cluster_params.v_sep_ratio
                )
            else:
                # Use full version (would need to fix V term indexing)
                H, basis = make_cluster_ham(
                    supercluster_k,
                    supercluster_idxs,
                    self.physical_params.hopping,
                    self.physical_params.V,
                    self.physical_params.U,
                    self.physical_params.mu_0,
                    self.cluster_params.L,
                    self.cluster_params.Nc,
                    self.cluster_params.int_sep_ratio,
                    self.cluster_params.v_sep_ratio,
                    self.ham_lib
                )
            
            # Diagonalize
            H_matrix = H.toarray()
            eigvals, eigvecs = np.linalg.eigh(H_matrix)
            
            # Calculate number and spin expectations for all eigenstates
            super_cluster_size = int(np.prod(supercluster_k.shape))
            
            # Number operators
            n_list_up = [[1.0, i] for i in range(super_cluster_size)]
            n_list_down = [[1.0, i] for i in range(super_cluster_size)]
            static_n_up = [["n|", n_list_up]]
            static_n_down = [["|n", n_list_down]]
            N_up_op = hamiltonian(static_n_up, [], basis=basis, dtype=np.float64)
            N_down_op = hamiltonian(static_n_down, [], basis=basis, dtype=np.float64)
            
            # Calculate expectations for all eigenstates
            n_ups = np.array([np.real(np.conj(eigvecs[:, i]) @ N_up_op.toarray() @ eigvecs[:, i]) 
                             for i in range(len(eigvals))])
            n_downs = np.array([np.real(np.conj(eigvecs[:, i]) @ N_down_op.toarray() @ eigvecs[:, i])
                               for i in range(len(eigvals))])
            n_tot = n_ups + n_downs
            
            # Store results
            k_points.append(supercluster_k)
            energy_spectrum.append(eigvals)
            number_spectrum.append(n_tot)
            
            # For spin spectrum, we need per-site information
            # For now, just store total up/down spins (can be refined later)
            spin_data = np.stack([n_ups, n_downs], axis=0)
            spin_spectrum.append(spin_data)
            
            if return_ham:
                ham_objects.append((H, basis))
        
        # Stack results
        k_points = np.stack(k_points, axis=0)
        energy_spectrum = np.stack(energy_spectrum, axis=0)
        number_spectrum = np.stack(number_spectrum, axis=0)
        spin_spectrum = np.stack(spin_spectrum, axis=0)
        
        if return_ham:
            return ham_objects
        else:
            return k_points, energy_spectrum, number_spectrum, spin_spectrum
    
    def get_cluster_thermodynamic_expectations(
        self, 
        full_spectrum_4tuple: tuple,
        temperature: Union[None, float] = None
    ):
        """
        Calculate thermodynamic expectations.
        Returns the same format as original FullSpectrum.get_cluster_thermodynamic_expectations()
        """
        if temperature is None:
            temperature = self.temperature
            
        cluster_energy_expectations = []
        cluster_number_expectations = []
        cluster_spin_expectations = []
        
        k_points, full_energy_spectrum, full_number_spectrum, full_spin_spectrum = full_spectrum_4tuple
        
        for i, k_point in enumerate(k_points):
            energy_spectrum = full_energy_spectrum[i]
            number_spectrum = full_number_spectrum[i]
            spin_spectrum = full_spin_spectrum[i]
            
            if temperature is None:
                # Zero temperature - ground state
                cluster_energy_argmin = np.argmin(energy_spectrum)
                cluster_energy_expectations.append(energy_spectrum[cluster_energy_argmin])
                cluster_number_expectations.append(number_spectrum[cluster_energy_argmin])
                
                # Spin polarization (up - down)
                ground_state_spins = spin_spectrum[:, cluster_energy_argmin]  # (2,) for total up/down
                spin_polarization = ground_state_spins[0] - ground_state_spins[1]  # up - down
                cluster_spin_expectations.append(spin_polarization)
            else:
                # Finite temperature
                beta = 1 / temperature
                boltzmann_weights = np.exp(-beta * energy_spectrum)
                sum_partition_function = np.sum(boltzmann_weights)
                
                # Thermal averages
                cluster_energy_expectations.append(
                    np.sum(energy_spectrum * boltzmann_weights) / sum_partition_function
                )
                cluster_number_expectations.append(
                    np.sum(number_spectrum * boltzmann_weights) / sum_partition_function
                )
                
                # Spin polarization
                weighted_spins = np.sum(spin_spectrum * boltzmann_weights[np.newaxis, :], axis=1)
                spin_polarization = (weighted_spins[0] - weighted_spins[1]) / sum_partition_function
                cluster_spin_expectations.append(spin_polarization)
        
        # Calculate system-wide expectations
        total_energy = np.sum(cluster_energy_expectations)
        total_number = np.sum(cluster_number_expectations)
        total_spin = np.sum(cluster_spin_expectations)
        
        system_expectations = (total_energy, total_number, total_spin)
        cluster_expectations = (cluster_energy_expectations, cluster_number_expectations, cluster_spin_expectations)
        
        return system_expectations, cluster_expectations


def run_general_cluster_method(
    U: float, 
    mu_0: float,
    V: float = 0,
    t: float = 1,
    L: int = 8,
    Nc: int = 2,
    int_sep_ratio: Tuple[int, int] = (1, 4),
    v_sep_ratio: Tuple[int, int] = (1, 2),
    use_simple_ham: bool = True,
    temperature: Union[None, float] = None
) -> Tuple[float, float]:
    """
    Run the general cluster method using the custom FullSpectrum class.
    This matches the interface of the original run_cluster_method.
    """
    
    # Set up parameters
    cluster_params = GeneralClusterParams(
        L=L,
        Nc=Nc, 
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        use_simple_ham=use_simple_ham
    )
    
    physical_params = PhysicalParams(
        U=U,
        V=V,
        hopping=t,
        mu_0=mu_0
    )
    
    # Create FullSpectrum object
    full_spectrum_object = FullSpectrumCustom(
        cluster_params,
        physical_params, 
        temperature=temperature
    )
    
    # Get full spectrum
    cluster_spectra = full_spectrum_object.get_full_spectrum()
    
    # Get ground state expectations (zero temperature)
    system_expectations, cluster_expectations = full_spectrum_object.get_cluster_thermodynamic_expectations(
        cluster_spectra, 
        temperature=temperature
    )
    
    energy, filling, spin = system_expectations
    
    return energy, filling


if __name__ == "__main__":
    """Test the custom FullSpectrum class."""
    
    print("Testing custom FullSpectrum class")
    print("=" * 60)
    
    # Test parameters
    L = 8
    Nc = 2
    int_sep_ratio = (1, 4)
    v_sep_ratio = (1, 2)
    
    U_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    V = 0.0
    t = 1.0
    
    results = []
    
    for U in U_values:
        mu_0 = U / 2  # Half-filling
        
        energy, filling = run_general_cluster_method(
            U=U,
            mu_0=mu_0,
            V=V,
            t=t,
            L=L,
            Nc=Nc,
            int_sep_ratio=int_sep_ratio,
            v_sep_ratio=v_sep_ratio,
            use_simple_ham=True
        )
        
        # Calculate per-site quantities
        energy_per_site = energy / L
        filling_per_site = filling / L
        
        results.append((U, energy_per_site, filling_per_site))
        print(f"U={U:.2f}: E/site={energy_per_site:.6f}, n/site={filling_per_site:.6f}")
    
    print("\nTest completed successfully!")