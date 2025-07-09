import pytest
import numpy as np
from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.hamiltonian import Hubbard1D
from aah_code.global_params import StatesParams, HamiltonianParams
import tenpy as tp
import logging
from tenpy.algorithms import exact_diag




log = logging.getLogger(__name__)






class TestHamiltonian:


    def make_test_clusters(self,state_params,test_type:str='default'):
        """
        make the initial cluster points t
        """

        rand_point=np.random.rand()*2*np.pi
        special_k_points=[np.array([[-np.pi],[0]]), #\mu_0=0
                np.array([[rand_point],[rand_point+np.pi]]), #\t_tilde=0
                np.array([[-np.pi],[np.pi]]), #\t_tilde=0
                np.array([[rand_point],[rand_point+2*np.pi]]), #\t_tilde=0
                ]
        if test_type=='default':
            rand_points=3
            random_k_points=[np.array([[-np.pi+np.random.rand()*2*np.pi],[-np.pi+np.random.rand()*2*np.pi]]) for _ in range(rand_points)]

        cluster_k_points=special_k_points+random_k_points
        #est_clusters=[LocalClusterBasis(k_points) for k_points in cluster_k_points]
        
        return cluster_k_points
    
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

            ham_dict = {
                'basis_class': test_basis,
                'V': physical_params.V,
                't': physical_params.hopping,
                'mu': 0,
                'U': physical_params.U,
            }

            test_ham=Hubbard1D(ham_dict)
            test_ham_mat=tp.algorithms.exact_diag.get_numpy_Hamiltonian(test_ham)
            test_ham_eigvals,test_ham_eigvecs=np.linalg.eigh(test_ham_mat)

            #One particle sector...

    def test_two_particle_gs(self):
        """
        Test the two particle limit. 
        Note that to run the same half-filling argument
        we need to add mu_tilde onto mu_0=U/2 otherwise you can't guarantee 
        half-filling (of course in general we can find the ground state we'd just have
        to do each particle sector indiviudally and take the min.)


        """
        state_params=StatesParams(spin_states=2)
        t=1
        U=3
        
        physical_params=HamiltonianParams(U=U,V=0,hopping=t)

        
        

        sampled_cluster_k_points=[np.array([[-np.pi],[0]]),
                        np.array([[0],[2*np.pi]]),
                        np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]]),
                        np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]]),
                        np.array([[np.random.rand()*2*np.pi],[np.random.rand()*2*np.pi]])
                        ]

        for i, chosen_k_points in enumerate(sampled_cluster_k_points):
            cluster_k_points=chosen_k_points
            test_basis=LocalClusterBasis(cluster_k_points,state_params)

            mu_tilde=(1/2)*(2*t*np.cos(cluster_k_points)).sum()

            ham_dict = {
                'basis_class': test_basis,
                'V': physical_params.V,
                't': physical_params.hopping,
                'U': physical_params.U,
                'mu':physical_params.U/2+mu_tilde,
            }

            test_ham=Hubbard1D(ham_dict)
            test_ham_mat=tp.algorithms.exact_diag.get_numpy_Hamiltonian(test_ham)
            test_ham_eigvals,test_ham_eigvecs=np.linalg.eigh(test_ham_mat)

            alpha_k=np.array([[0],[np.pi]])
            #mu_tilde=(1/2)*(2*t*np.cos(cluster_k_points)).sum()
            t_tilde=(1/2)*(2*t*np.cos(cluster_k_points[0])-2*t*np.cos(cluster_k_points[1])).sum()
            
            mu_0=ham_dict['mu']





            print(f'U={U}, mu_tilde={mu_tilde}, mu_0={mu_0}, t_tilde={t_tilde}')
            analytic_two_particle_gs=2*mu_tilde-2*mu_0+(1/2)*(U-np.sqrt(U**2+(4*t_tilde)**2))
            #analytic_two_particle_gs=analytic_two_particle_gs*np.heaviside(0-analytic_two_particle_gs,0)

            log.debug(f'Cluster Ham GS Energy: {test_ham_eigvals[0]}, Analytic energy: {analytic_two_particle_gs}')
            
            np.testing.assert_array_almost_equal(
                test_ham_eigvals[0],
                analytic_two_particle_gs,
                err_msg="Two-particle ground state energy is not recovered"
            )
            
            
            
    
    def test_twoparticle_with_V(self):
        """
        test that the two particle ground state energy is correct with V
        """
        state_params=StatesParams(spin_states=2)
        
        #physical_params=HamiltonianParams(U=0,V=2,hopping=1)
        t=1
        U=10
        V=50

        test_clusters=self.make_test_clusters(state_params,test_type='default')
    
        for test_cluster_ks in test_clusters:
            
            cluster_object=LocalClusterBasis(test_cluster_ks,state_params)

            mu_tilde=(1/2)*(2*t*np.cos(test_cluster_ks)).sum()
            
            mu_0=U/2+mu_tilde

            ham_dict = {
                'basis_class': cluster_object,
                'V': V,
                't': t,
                'U': U,
                'mu':mu_0,
            }

            test_ham=Hubbard1D(ham_dict)
            
            test_ham_mat=tp.algorithms.exact_diag.get_numpy_Hamiltonian(test_ham)
            test_ham_eigvals,test_ham_eigvecs=np.linalg.eigh(test_ham_mat)

            # 2) dense ED
            ed = exact_diag.ExactDiag(test_ham)                 # solver instance  :contentReference[oaicite:0]{index=0}
            ed.build_full_H_from_bonds() 
            ed.full_diagonalization()                    # fills ed.full_H  :contentReference[oaicite:1]{index=1}
            E0, psi_vec = ed.groundstate()    
            
            psi_mps = ed.full_to_mps(psi_vec)               # returns E0, eigen-vector  :contentReference[oaicite:2]{index=2}

            n_up   = psi_mps.expectation_value('Nu')     # array, one value per site
            n_down = psi_mps.expectation_value('Nd')
            n_tot  = n_up + n_down                  # or psi.expectation_value('Ntot')

            log.debug(f'Number expectations for ham GS: n_up: {n_up}, n_down: {n_down}, n_tot: {n_tot}')

            #log.debug(f'Number expectations for ham GS: n_up: {n_up}, n_down: {n_down}, n_tot: {n_tot}')

            t_tilde=(1/2)*(2*t*np.cos(test_cluster_ks[0])-2*t*np.cos(test_cluster_ks[1])).sum()

            log.debug(f'shape t_tilde: {t_tilde.shape}')

            two_particle_spin_zero_sector=np.array([[U+V,-t_tilde,t_tilde,0],
                                                    [-t_tilde,0,0,t_tilde],
                                                    [t_tilde,0,0,-t_tilde],
                                                    [0,t_tilde,-t_tilde,U-V]])
            
            
            
            two_particle_spin_zero_sector=two_particle_spin_zero_sector-2*(mu_0-mu_tilde)*np.eye(np.shape(two_particle_spin_zero_sector)[0])


            analytic_two_particle_gs_V_zero=2*mu_tilde-2*mu_0+(1/2)*(U-np.sqrt(U**2+(4*t_tilde)**2))

            log.debug(f'two particle exact hermitian? {np.allclose(two_particle_spin_zero_sector,two_particle_spin_zero_sector.conj().T)}')
            
            exact_gs=np.linalg.eigvals(two_particle_spin_zero_sector).min()

            log.debug(f'Cluster Ham GS Energy: {test_ham_eigvals[0]}, Analytic energy: {exact_gs}, V zero energy: {analytic_two_particle_gs_V_zero}')

            try:
                np.testing.assert_array_almost_equal(
                    test_ham_eigvals[0],
                    exact_gs,
                    err_msg="Two-particle ground state energy is not recovered"
                )
                log.info(f"✓ Test PASSED for {test_cluster_ks}. Expected energy: {exact_gs}, Actual energy: {test_ham_eigvals[0]}, error: {100*abs(test_ham_eigvals[0]-exact_gs)/abs(exact_gs)}%")
            except AssertionError as e:
                log.error(f"✗ Test FAILED for {test_cluster_ks}. Expected energy: {exact_gs}, Actual energy: {test_ham_eigvals[0]}, error: {100*abs(test_ham_eigvals[0]-exact_gs)/abs(exact_gs)}%")
                raise AssertionError(f"Energy mismatch for {test_cluster_ks}. Expected: {exact_gs}, Got: {test_ham_eigvals[0]}") from e

        
