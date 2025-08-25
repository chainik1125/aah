import logging
import numpy as np
import itertools
import matplotlib.pyplot as plt
from aah_code.quspin.quspin_test import check_quspin_vs_tenpy_pi_pi
from aah_code.quspin.quspin_hamiltonian import QuSpinHamiltonian
from aah_code.basis import LocalClusterBasis
from aah_code.clusters import ClusterExperiment
from aah_code.global_params import StatesParams,HamiltonianParams
from aah_code.main import run_cluster_method, run_dmrg_method
from aah_code.quspin.quspin_utils import extract_single_particle_hamiltonian_dense
from aah_code.quspin.quspin_hamiltonian import hubbard_V_pi_int_half_pi
from aah_code.hamiltonian import test_quick_mismatched, MismatchedQuick, get_spectra

log = logging.getLogger(__name__)

def analytic_nonint_energies_pi_half_pi(t,V,mu_0):
    E0=-np.sqrt(V**2+(2*t)**2)
    E1=-V
    E2=+V
    E3=np.sqrt(V**2+(2*t)**2)
    return np.array([E0,E1,E2,E3])-mu_0

def get_reconstruct_many_body_from_sp(many_body_ham,sp_evals,spin_projected:bool=True):
    local_hilbert_dim=(2*2) #Assuming spinful
    site_count=int(np.log(many_body_ham.shape[0])/np.log(local_hilbert_dim))
    log.debug(f'site count: {site_count}')
    many_body_eigvals=[]
    if spin_projected:
        sp_evals=np.repeat(sp_evals,2)
        
    
    
    for i in range(len(sp_evals)+1):
        particle_spectrum=np.array(list(itertools.combinations(sp_evals,i)))
        many_body_eigvals.extend(np.sum(particle_spectrum,axis=1))
    
    many_body_eigvals=np.array(many_body_eigvals)
    many_body_eigvals=np.sort(many_body_eigvals)


    return many_body_eigvals

class TestHamiltonian:
    def test_quspin_pi_pi_construction(self):
        system_size=10
        t=1
        mu_0=0
        V=0
        U=0
        run_cluster_method(U=U,mu_0=mu_0,V=V,t=t,system_size=system_size,ham_lib='quspin')
    
    def test_single_particle_reconstruction(self):
        system_size=4
        t=1
        mu_0=0
        V=2
        U=0
        mismatched_ks=np.array([[[-np.pi],[-np.pi/2]],[[0],[np.pi/2]]])
        states_params=StatesParams(spin_states=2)
        basis_classes=[LocalClusterBasis(mismatched_ks[0],states_params),LocalClusterBasis(mismatched_ks[1],states_params)]
        ham_dict={'L':system_size,'t':t,'V':V,'U':U,'mu':mu_0,'basis_classes':basis_classes}
        many_body_ham,basis=hubbard_V_pi_int_half_pi(ham_dict)
        sp_ham,sp_basis=extract_single_particle_hamiltonian_dense(many_body_ham,basis,spin='up')


        # log.debug(f'many body ham: {sp_ham}')
        # log.debug(f'sp basis: {sp_basis}')
        
        #First test: single particle energies should match analytic results
        analytic_energies=analytic_nonint_energies_pi_half_pi(t,V,mu_0)
        sp_evals=np.linalg.eigvalsh(sp_ham)
        
        
        
        np.testing.assert_allclose(sp_evals,analytic_energies,atol=1e-9)

        #Second test: many body energies should match single particle reconstruction
        reconstructed_many_body_eigvals=get_reconstruct_many_body_from_sp(many_body_ham,sp_evals)
        many_body_eigvals=np.linalg.eigvalsh(many_body_ham.toarray())

        # log.debug(f'reconstructed many body eigvals: {reconstructed_many_body_eigvals[:4]}')
        # log.debug(f'many body eigvals: {many_body_eigvals[:4]}')
        # log.debug(f'diff: {(reconstructed_many_body_eigvals-many_body_eigvals)[:4]}')
        # log.debug(f'single particle energies: {sp_evals}')


        
        np.testing.assert_allclose(reconstructed_many_body_eigvals,many_body_eigvals,atol=1e-9)
    
    def test_mismatched_nonint_dmrg_match(self):
        
        lattice_points=20
        U=0
        V=0
        t=10
        mu_0=0
        physical_params=HamiltonianParams(U,V,t,mu_0)

        cluster_size=2 #remember this is the int cluster size
        
        
        system_expectations,cluster_expectations=test_quick_mismatched(lattice_points,cluster_size,physical_params,ham_lib='quspin')
        total_energy,total_filling,total_spin=system_expectations

        sub_en_density=(total_energy+mu_0*total_filling)/lattice_points

        log.debug(f'site energy density: {sub_en_density}')
        log.debug(f'filling density: {total_filling/lattice_points}')
        log.debug(f'site spin density: {total_spin/lattice_points}')

        energy_dmrg, filling_dmrg, psi = run_dmrg_method(U, mu_0, V, t, cluster_size, chi=80)

        energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)

        log.debug(f'dmrg energy density: {energy_dmrg_subtracted/cluster_size}')
        log.debug(f'dmrg filling density: {filling_dmrg}')

        log.debug(f'DMRG energy density: {energy_dmrg_subtracted}')
        log.debug(f'cluster energy density: {sub_en_density}')
        log.debug(f'% Error: {np.abs((energy_dmrg_subtracted-sub_en_density)/sub_en_density)*100}')
    

    def test_mismatched_nonint_dmrg_v_grid(self):
        
        lattice_points=20
        U=0
        t=1
        mu_0=U/2
        V_values=np.linspace(0,10,5)

        cluster_energies=[]
        cluster_fillings=[]
        dmrg_energies=[]
        dmrg_fillings=[]

        for V in V_values:
            physical_params=HamiltonianParams(U,V,t,mu_0)

            cluster_size=2 #remember this is the int cluster size
            
            
            system_expectations,cluster_expectations=test_quick_mismatched(lattice_points,cluster_size,physical_params,ham_lib='quspin')
            total_energy,total_filling,total_spin=system_expectations

            sub_en_density=(total_energy+mu_0*total_filling)/lattice_points

            log.debug(f'site energy density: {sub_en_density}')
            log.debug(f'filling density: {total_filling/lattice_points}')
            log.debug(f'site spin density: {total_spin/lattice_points}')

            cluster_energies.append(sub_en_density)
            cluster_fillings.append(total_filling/lattice_points)


            energy_dmrg, filling_dmrg, psi = run_dmrg_method(U, mu_0, V, t, cluster_size, chi=80)

            energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)

            log.debug(f'dmrg energy density: {energy_dmrg_subtracted/cluster_size}')
            log.debug(f'dmrg filling density: {filling_dmrg}')

            log.debug(f'DMRG energy density: {energy_dmrg_subtracted}')
            log.debug(f'cluster energy density: {sub_en_density}')
            log.debug(f'% Error: {np.abs((energy_dmrg_subtracted-sub_en_density)/sub_en_density)*100}')

            dmrg_energies.append(energy_dmrg_subtracted)
            dmrg_fillings.append(filling_dmrg)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(V_values, cluster_energies, label='cluster')
        ax1.plot(V_values, dmrg_energies, label='dmrg')
        ax1.set_xlabel('V')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.set_title('Energy vs V')
        
        ax2.plot(V_values, cluster_fillings, label='cluster')
        ax2.plot(V_values, dmrg_fillings, label='dmrg')
        ax2.set_xlabel('V')
        ax2.set_ylabel('Filling')
        ax2.legend()
        ax2.set_title('Filling vs V')
        
        plt.tight_layout()
        plt.show()

    
    def test_filling_spectrum(self):

        lattice_points=20
        t=1
        V=2
        U=0
        mu_0=U/2

        state_params=StatesParams(spin_states=2)
        physical_params=HamiltonianParams(U,V,t,mu_0)
        cluster_size=2
        int_lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//4)
        print(type(int_lattice_object))
        test=MismatchedQuick(int_lattice_object,physical_params,lattice_points//2)
        cluster_ks,cluster_idxs=test.recluster()
        #print(f'cluster ks shape: {cluster_idxs.shape}')
        #print(f'cluster idxs: {cluster_idxs}')

        #k_points,energies,number_spectrum,spin_spectrum=get_spectra(cluster_ks)
        spectra_4tuple=get_spectra(cluster_ks, state_params, physical_params,ham_lib='quspin')
        
        log.debug(f'k points shape: {spectra_4tuple[0].shape},\n energies shape: {spectra_4tuple[1].shape},\n number_spectrum shape: {spectra_4tuple[2].shape}, spin spectrum shape: {spectra_4tuple[3].shape}')

        k_points,energies,number_spectrum,spin_spectrum=spectra_4tuple



        log.debug(f'number gs spectrum: {number_spectrum[:,0]}')
        log.debug(f'first four energies: {energies[:,:4]}')

        log.debug(f'cluster gs energy spectrum: {energies[:,0].sum()/lattice_points}')
        log.debug(f'cluster gs number spectrum: {number_spectrum.sum(axis=-1)[:,0].sum()/(lattice_points)}')

        energy_dmrg,filling_dmrg,psi=run_dmrg_method(U,mu_0,V,t,lattice_points,32)

        
        log.debug(f'sub DMRG energy: {energy_dmrg+filling_dmrg*mu_0}')

        log.debug(f'filling DMRG: {filling_dmrg}')


        

        
        

        


        






    
        
        

    
        


        
        