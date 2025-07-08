import pytest
import numpy as np
from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.hamiltonian import Hubbard1D
from aah_code.global_params import StatesParams, HamiltonianParams
import tenpy as tp
import logging




log = logging.getLogger(__name__)




class TestHamiltonian:
    def test_hamiltonian_hermitian(self):
        """Test that the Hamiltonian is Hermitian."""
        state_params=StatesParams(spin_states=2)
        physical_params=HamiltonianParams(U=3.0,V=2.0,hopping=1.0)
        cluster_k_points=np.array([[-np.pi],
                                [0]])
        test_basis=LocalClusterBasis(cluster_k_points,state_params)
        
        ham_dict = {
            'basis_class': test_basis,
            'V': physical_params.V,
            't': physical_params.hopping,
            'mu': 0,
            'U': physical_params.U,
        }
        test_ham=Hubbard1D(ham_dict)

        #def get matrix:
        test_ham_mat=tp.algorithms.exact_diag.get_numpy_Hamiltonian(test_ham)

        log.debug(f'Hamiltonian shape: {test_ham_mat.shape}')

        np.testing.assert_array_almost_equal(
            test_ham_mat, 
            test_ham_mat.conj().T,
            err_msg="Hamiltonian is not Hermitian"
        )


    def test_noninteracting_limit(self):
        """
        Recover non-interacting energy
        """
        state_params=StatesParams(spin_states=2)
        t=1
        physical_params=HamiltonianParams(U=0,V=0,hopping=t)

        sampled_cluster_k_points=[np.array([[-np.pi],[0]]),
                                np.array([[0],[2*np.pi]]),
                                np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]]),
                                np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]]),
                                np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]])
                                ]
        
        for i, chosen_k_points in enumerate(sampled_cluster_k_points):
            cluster_k_points=chosen_k_points
            test_basis=LocalClusterBasis(cluster_k_points,state_params)
            
            ham_dict = {
                'basis_class': test_basis,
                'V': physical_params.V,
                't': physical_params.hopping,
                'mu': 0,
                'U': physical_params.U,
            }

            test_ham=Hubbard1D(ham_dict)

        
            non_interacting_gs=2*np.heaviside(0-2*t*np.cos(cluster_k_points),0)*(2*t*np.cos(cluster_k_points))
            non_interacting_gs=non_interacting_gs.sum()

            #log.debug(f'non_interacting_gs:{non_interacting_gs}, remember that the dispersion is -2tcos if you double count the hc!')

            test_ham_mat=tp.algorithms.exact_diag.get_numpy_Hamiltonian(test_ham)
            test_ham_eigvals,test_ham_eigvecs=np.linalg.eigh(test_ham_mat)

            #log.debug(f'test_ham_mat in limit U-->0 gs: {test_ham_eigvals[0]}')
            
            # Store values for detailed reporting
            expected_energy = non_interacting_gs
            actual_energy = test_ham_eigvals[0]
            k_array_info = f"Array {i}: {cluster_k_points.flatten()}"
            
            try:
                np.testing.assert_array_almost_equal(
                    actual_energy,
                    expected_energy,
                    err_msg="Non-interacting energy is not recovered"
                )
                log.info(f"✓ Test PASSED for {k_array_info}")
                log.info(f"  Expected energy: {expected_energy}")
                log.info(f"  Actual energy: {actual_energy}")
            except AssertionError as e:
                log.error(f"✗ Test FAILED for {k_array_info}")
                log.error(f"  Expected energy: {expected_energy}")
                log.error(f"  Actual energy: {actual_energy}")
                log.error(f"  Difference: {abs(actual_energy - expected_energy)}")
                raise AssertionError(f"Energy mismatch for {k_array_info}. Expected: {expected_energy}, Got: {actual_energy}") from e



        return None
    
    def test_noninteracting_limit_self_consistency(self):
        """
        Test that the spectrum is the sum of single particle energies.
        """
        state_params=StatesParams(spin_states=2)
        t=1
        physical_params=HamiltonianParams(U=0,V=0,hopping=t)

        sampled_cluster_k_points=[np.array([[-np.pi],[0]]),
                        np.array([[0],[2*np.pi]]),
                        np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]]),
                        np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]]),
                        np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]])
                        ]

        for i, chosen_k_points in enumerate(sampled_cluster_k_points):
            cluster_k_points=chosen_k_points
            test_basis=LocalClusterBasis(cluster_k_points,state_params)
            
            
    
    def test_half_filling(self):
        """
        test that the 
        """
