import pytest
import numpy as np
from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.hamiltonian import Hubbard1D, FullSpectrum, inspect_hamiltonian_terms
from aah_code.hamiltonian import QuickHubbard1D, get_spectra, MismatchedQuick
from aah_code.global_params import StatesParams, HamiltonianParams


import tenpy as tp
import logging
from tenpy.algorithms import exact_diag
import pandas as pd

import numpy as np
from tenpy.algorithms import exact_diag
from aah_code.matrix_display import matrix_to_dataframe, print_matrix
import tabulate


def latex_matrix_with_labels(H, basis_labels=None, precision=3, col_align="c"):
    r"""
    Build a LaTeX string showing the matrix and its basis in one array‑block.

    Parameters
    ----------
    H : array_like, shape (n,n)
        The matrix you want to display (e.g. from `single_particle_block`).
    basis_labels : list[str] or None
        Human‑readable labels for each basis ket in the same order as `H`.
        If None, simple indices 0…n‑1 are used.
    precision : int
        Decimal places to print (uses general‐format `{:.<p>g}`).
    col_align : str
        Column alignment for LaTeX array (`c`, `r`, or `l`).

    Returns
    -------
    latex : str
        Ready‑to‑copy LaTeX code (enclosed in `\[ … \]`).
    """
    H = np.asarray(H)
    n = H.shape[0]
    if basis_labels is None:
        basis_labels = [f"${i}$" for i in range(n)]

    # --- build the pmatrix body ------------------------------------------------
    num_fmt = f"{{:.{precision}g}}"
    body_rows = [
        " & ".join(num_fmt.format(x) for x in H[i]) for i in range(n)
    ]
    pmatrix = "\\begin{pmatrix}\n" + " \\\\\n".join(body_rows) + "\n\\end{pmatrix}"

    # --- assemble the outer array with row/col headers -------------------------
    col_header = " & ".join([""] + basis_labels)
    row_labels = " \\\\\n".join(basis_labels)  # end each line with \\
    outer = (
        "\\[\n"
        "\\begin{array}{" + col_align * (n + 1) + "}\n"
        + col_header + " \\\\\n\\hline\n"
        + pmatrix + " & \\begin{array}{c}\n" + row_labels + "\n\\end{array}\n"
        "\\end{array}\n"
        "\\]\n"
    )
    return outer

def single_particle_block(model, spin=None, from_mpo=True):
    """
    Return the 1‑particle Hamiltonian in the requested spin sector.

    Parameters
    ----------
    model : CouplingMPOModel
        Your TeNPy Hubbard model (already initialised).
    spin  : {None, 'up', 'down'}, optional
        • None  → keep both spins (default, size 2L × 2L)  
        • 'up'  → project onto N=1, S_z=+½   (size L × L)  
        • 'down'→ project onto N=1, S_z=‑½   (size L × L)
    from_mpo : bool
        Passed straight to `exact_diag.get_numpy_Hamiltonian`.

    Returns
    -------
    H1 : (n,n) complex ndarray
        Dense matrix in the chosen sub‑space.
    """
    # full Hamiltonian in occupation basis |σ₁σ₂…σ_L⟩
    H = exact_diag.get_numpy_Hamiltonian(model, from_mpo=from_mpo)

    L           = model.lat.N_sites
    occ_per_st  = np.array([0, 1, 1, 2], dtype=np.uint8)  # |0>,|↑>,|↓>,|↑↓>
    spin_per_st = np.array([ 0, 1,-1, 0], dtype=np.int8)  #   0 , +1 , -1 ,  0

    want_spin = {'up': 1, 'down': -1, None: 0}[spin]      # 0 ⇒ accept ±1

    keep = []
    for idx in range(4**L):
        tmp, n, sz = idx, 0, 0
        for _ in range(L):
            st   = tmp & 3           # %4, faster
            n   += occ_per_st [st]
            sz  += spin_per_st[st]
            tmp >>= 2                # //4
            if n > 1:                # early exit
                break
        if n == 1 and (want_spin == 0 or sz == want_spin):
            keep.append(idx)

    keep = np.asarray(keep, dtype=np.int64)
    return H[np.ix_(keep, keep)]



def extract_single_particle_hamiltonian(mpo_hamiltonian):
	"""
	Extract the 1-particle Hamiltonian matrix from MPO hamiltonians (Hubbard1D or QuickHubbard1D).
	
	This extracts the matrix representation of the Hamiltonian restricted to the 1-particle subspace.
	For a system with n_c cluster sites:
	- Hubbard1D: extracts 2n_c x 2n_c matrix (1 particle in 2n_c spin-orbitals)  
	- QuickHubbard1D: extracts 2n_c x 2n_c matrix from each cluster
	
	Args:
		mpo_hamiltonian: Either Hubbard1D or QuickHubbard1D instance
		
	Returns:
		np.ndarray: 1-particle Hamiltonian matrix in the single-particle subspace
	"""
	# Get the full Hamiltonian matrix
	full_H = exact_diag.get_numpy_Hamiltonian(mpo_hamiltonian)
	
	# Get lattice information
	L = mpo_hamiltonian.lat.Ls[0]  # Number of sites
	
	# Generate all possible basis states and identify 1-particle states
	# Each site has 4 states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩ 
	# For 1-particle states, we want states with exactly one particle
	
	# Generate all basis states (4^L total states)
	single_particle_indices = []
	basis_states = []
	
	for state_idx in range(4**L):
		# Convert state index to occupation numbers for each site
		temp_idx = state_idx
		site_occupations = []
		total_particles = 0
		
		for site in range(L):
			site_occ = temp_idx % 4
			site_occupations.append(site_occ)
			temp_idx //= 4
			
			# Count particles: |0⟩=0, |↑⟩=1, |↓⟩=1, |↑↓⟩=2
			if site_occ == 1 or site_occ == 2:  # |↑⟩ or |↓⟩
				total_particles += 1
			elif site_occ == 3:  # |↑↓⟩
				total_particles += 2
		
		# Keep only states with exactly 1 particle
		if total_particles == 1:
			single_particle_indices.append(state_idx)
			basis_states.append(site_occupations)
	
	# Extract the 1-particle subspace Hamiltonian
	n_1p = len(single_particle_indices)
	single_particle_H = np.zeros((n_1p, n_1p), dtype=complex)
	
	for i, idx_i in enumerate(single_particle_indices):
		for j, idx_j in enumerate(single_particle_indices):
			single_particle_H[i, j] = full_H[idx_i, idx_j]
	
	return single_particle_H








def get_single_matched(cluster_size,lattice_points,ham_dict_matched_base,state_params):
    #TODO: This should all be one function - namely the one you use in your loops!
    matched_lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//2)
    matched_ks=matched_lattice_object.generate_clusters()
    single_particle_hams=[]
    eigvals=[]
    #log.debug(f'matched k shape: {matched_ks.shape}')
    d = lattice_points//4
    N = matched_ks.shape[0]
    if N < d + 1:
        raise ValueError("Need at least d+1 slices to form one pair")
    n_pairs = N - d           # here: 4 − 2 = 2
    first  = matched_ks[:n_pairs]        # A[0], A[1]    → shape (2,2,1)
    second = matched_ks[d:d + n_pairs]   # A[2], A[3]    → shape (2,2,1)
    matched_clusters = np.stack((first, second), axis=1)

    

    for matched_cluster in matched_clusters:
        cluster_eigvals=[]
        cluster_hams=[]
        for k in matched_cluster:
            test_basis=LocalClusterBasis(k,state_params)
            ham_dict_matched_base['basis_class']=test_basis
			# ham_dict_matched = {
            #     'basis_class': test_basis,
            #     'V': V,
            #     't': t,
            #     'mu': mu,
            #     'U': U,
            # }
            matched_ham=Hubbard1D(ham_dict_matched_base)
            single_particle_matched=single_particle_block(matched_ham,spin='up')
            matched_evals,matched_evecs=np.linalg.eigh(single_particle_matched)
            cluster_eigvals.append(matched_evals)
            cluster_hams.append(single_particle_matched)
    
        #NOTE! Don't confuse the single particle and the filled spectrum!
        combined_energy_vals=np.stack([cluster_eigvals[i][j] for i in range(cluster_eigvals[0].shape[0]) for j in range(cluster_eigvals[1].shape[0])],axis=0)
        combined_energy_vals=np.sort(combined_energy_vals,axis=0)

        eigvals.append(combined_energy_vals)
        single_particle_hams.append(cluster_hams)

    
    return np.array(eigvals), single_particle_hams,matched_clusters



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

		
	def test_fourparticle_reduces(self):
		"""
		Test that for V=0, the four particle Hamiltnoian gives the same energies
		as two two particle ones at the correct separation.
		"""

		shift=np.pi
		n=3
		#start=np.array([[-np.pi/4],[np.pi/4]])
		
		starting_kpoints=[
			np.array([[-np.pi],[-np.pi/2]])
		]

		starting_kpoints=starting_kpoints + [np.array([[k],[k + np.pi/2]]) for k in np.random.rand(n) * np.pi] 


		

		#start=np.array([[np.pi],[np.pi/2]])

		for start in starting_kpoints:
			test_ks=[start,start+shift]
			stacked_test_ks=np.stack([np.stack([start,start+shift],axis=0)],axis=0)
			log.debug(f'test_ks shape: {np.array(test_ks).shape}')
			log.debug(f'stacked shape: {stacked_test_ks.shape}')
			
			state_params=StatesParams(spin_states=2)

			U=1
			V=0
			t=1
			mu_0=U/2
			physical_params = HamiltonianParams(U, V, t, mu_0)

			#Get energies for the two two-particle clusters
			full_spectrum_object = FullSpectrum(test_ks, state_params, physical_params)
			cluster_spectra = full_spectrum_object.get_full_spectrum()
			k_points,energy_spectrum,number_spectrum,spin_spectrum=cluster_spectra

			log.debug(f"energy_spectrum shape: {energy_spectrum.shape}")

			#inspect ham:
			ham_objects=full_spectrum_object.get_full_spectrum(return_ham=True)
			#print(f'first cluster two site')
			#inspect_hamiltonian_terms(ham_objects[0])

			
			

			#Now do the same for the four site
			

			spectra_4tuple=get_spectra(stacked_test_ks, state_params, physical_params)
			hams=get_spectra(stacked_test_ks, state_params, physical_params,return_ham=True)
			inspect_hamiltonian_terms(hams[0])
			
			k_points_4,energy_spectrum_4,number_spectrum_4,spin_spectrum_4=spectra_4tuple

			log.debug(f'k_points shape: {k_points_4.shape},energy shape: {energy_spectrum_4.shape}')


			log.debug(f'first two energies clusters: {energy_spectrum[:,:2]}, first two energies 4: {energy_spectrum_4[:,:2]}')

			#Now check that they are the same across the spectrum
			#first I need to form the spectrum
			combined_energy_vals=np.stack([energy_spectrum[0,i]+energy_spectrum[1,j] for i in range(energy_spectrum.shape[1]) for j in range(energy_spectrum.shape[1])],axis=0)
			combined_energy_vals=np.sort(combined_energy_vals,axis=0)
			
			
			try:
				np.testing.assert_array_almost_equal(
					combined_energy_vals[:5],
					energy_spectrum_4[0,:5],
					err_msg="Clusters didn't match"
				)
				log.info(f"✓ Test PASSED for {stacked_test_ks}.%")
			except AssertionError as e:
				log.error(f"✗ Test FAILED for {stacked_test_ks}.")
				raise AssertionError(f"Energy mismatch for {stacked_test_ks}") from e		
		
		return None
	
	def test_foursite_hubbard_zero_U(self):
		"""
		test first the minimal example of the four site Hubbard model.
		"""
		starting_kpoints_pi=[
			np.array([[-np.pi],[0]])
		]

		starting_kpoints_halfpi=[
			np.array([[-np.pi],[-np.pi/2]])
		]

		shift=np.pi
		test_ks_pi_clustering=[starting_kpoints_pi[0],starting_kpoints_pi[0]+np.pi/2]
		stacked_test_ks=np.stack([np.stack([starting_kpoints_halfpi[0],starting_kpoints_halfpi[0]+np.pi],axis=0)],axis=0)

		
		log.info(f'stacked shape: {stacked_test_ks.shape}')
		

		U=0
		V=1
		t=1
		mu_0=U/2
		physical_params = HamiltonianParams(U, V, t, mu_0)
		state_params=StatesParams(spin_states=2)
		#Get energies for the two two-particle clusters
		full_spectrum_object = FullSpectrum(test_ks_pi_clustering, state_params, physical_params)
		cluster_spectra = full_spectrum_object.get_full_spectrum()
		k_points,energy_spectrum,number_spectrum,spin_spectrum=cluster_spectra

		log.debug(f"energy_spectrum shape: {energy_spectrum.shape}")

		#inspect ham:
		ham_objects=full_spectrum_object.get_full_spectrum(return_ham=True)
		#print(f'first cluster two site')
		#inspect_hamiltonian_terms(ham_objects[0])

		
		

		#Now do the same for the four site
		

		spectra_4tuple=get_spectra(stacked_test_ks, state_params, physical_params)
		
		#hams=get_spectra(stacked_test_ks, state_params, physical_params,return_ham=True)
		#inspect_hamiltonian_terms(hams[0])
		
		k_points_4,energy_spectrum_4,number_spectrum_4,spin_spectrum_4=spectra_4tuple

		log.debug(f'k_points shape: {k_points_4.shape},energy shape: {energy_spectrum_4.shape}')
		


		log.debug(f'first four energies clusters: {energy_spectrum[:,:4]}, first 2 energies 4: {energy_spectrum_4[:,:2]}')
		
		mu_tilde_array=(2*t*np.cos(stacked_test_ks[:,0,:,:])+2*t*np.cos(stacked_test_ks[:,1,:,:]))/2
		t_tilde_array=(2*t*np.cos(stacked_test_ks[:,0,:,:])-2*t*np.cos(stacked_test_ks[:,1,:,:]))/2
		gs_energy=2*(mu_tilde_array-np.sqrt(t_tilde_array**2+V**2))
		gs=(gs_energy*np.heaviside(mu_0-gs_energy,0)).sum()
		log.debug(f'exact: {gs}')

		print(f't_tilde_array: {t_tilde_array}')
		return None
		

		#Now check that they are the same across the spectrum
		#first I need to form the spectrum
		
		combined_energy_vals=np.stack([energy_spectrum[0,i]+energy_spectrum[1,j] for i in range(energy_spectrum.shape[1]) for j in range(energy_spectrum.shape[1])],axis=0)
		combined_energy_vals=np.sort(combined_energy_vals,axis=0)

		print(f'combined_energy_vals first 2: {combined_energy_vals[:2]}')
		print(f'4 particle first 2: {energy_spectrum_4[0,:2]}')

		return None
		try:
			np.testing.assert_array_almost_equal(
				combined_energy_vals[:5],
				energy_spectrum_4[0,:5],
				err_msg="Clusters didn't match"
			)
			log.info(f"✓ Test PASSED for {stacked_test_ks}.%")
		except AssertionError as e:
			log.error(f"✗ Test FAILED for {stacked_test_ks}.")
			raise AssertionError(f"Energy mismatch for {stacked_test_ks}") from e		


	
	def test_mismatched_matched_at_zero_U(self):
		"""
		We are not applying any approximations to V for small, finite modulation periods,
		and so the different clustering schemes should agree for all V and t at U=0.

		"""
		shift=np.pi
		n=3
		#start=np.array([[-np.pi/4],[np.pi/4]])
		
		starting_kpoints=[
			np.array([[-np.pi],[-np.pi/2]]),
			np.array([[-np.pi/4],[np.pi/4]]),
		]

		starting_kpoints=starting_kpoints + [np.array([[k],[k + np.pi/2]]) for k in np.random.rand(n) * np.pi] 


		
		pass
	
	def test_extract_single_particle_hamiltonian(self):
		"""Test the extract_single_particle_hamiltonian function."""
		state_params = StatesParams(spin_states=2)
		t = 1.0
		U = 0
		V = 2
		mu = 0
		
		# Test with a simple 2-site cluster
		cluster_k_points = np.array([[-np.pi], [0]])
		test_basis = LocalClusterBasis(cluster_k_points, state_params)

		ham_dict_matched = {
			'basis_class': test_basis,
			'V': V,
			't': t,
			'mu': mu,
			'U': U,
		}
		full_ham_matched = Hubbard1D(ham_dict_matched)

		
		#Test QuickHubbard1D
		cluster_k_points=np.array([[-np.pi], [-np.pi/2]])
		cluster_k_points=np.stack([cluster_k_points,cluster_k_points+np.pi],axis=0)
		test_basis_1=LocalClusterBasis(cluster_k_points[0],state_params)
		test_basis_2=LocalClusterBasis(cluster_k_points[1],state_params)
		
		ham_dict_mismatched={
			'basis_classes':[test_basis_1,test_basis_2],
					'L':cluster_k_points[0].shape[0]*cluster_k_points[1].shape[0],
					'L_cluster':cluster_k_points[0].shape[0],
					'V':V,
					't':t,
					'U':U,
					'mu':mu,
					
		}
		
		# Create full Hamiltonian with interactions

		full_ham_mismatched=QuickHubbard1D(ham_dict_mismatched)
		
		
		test_single_particle=single_particle_block(full_ham_mismatched,spin='up')

		log.debug(f'test single particle shape: {test_single_particle.shape}')
		log.debug(f'test single particle ham:')
		
		log.debug(test_single_particle)
		
		labels = [f"|site {i}⟩" for i in range(test_single_particle.shape[0])]

		df = matrix_to_dataframe(test_single_particle, labels, precision=2)
		log.debug(print_matrix(df, style="tabulate", tablefmt="grid"))

		mismatched_evals,mismatched_evecs=np.linalg.eigh(test_single_particle)
		log.debug(f'eigenvals: {mismatched_evals}')

		return None
	

	def test_full_single_particle(self):
		"""Test the extract_single_particle_hamiltonian function."""

		#Initial definitions
		state_params = StatesParams(spin_states=2)
		lattice_points=4
		cluster_size=2
		#physical params
		t = 1.0
		U = 0
		V = 2
		mu = 0
		physical_params=HamiltonianParams(U,V,t,mu)

		
		#function for making and then extracting the single particle hamiltonian of matched case

		

		def get_single_mismatched():
			mismatched_lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//4)
			mismatch_obj=MismatchedQuick(mismatched_lattice_object,physical_params,V_k_period=lattice_points//2)
			mismatched_ks=mismatch_obj.recluster()[0]

			
			

			eigvals=[]
			single_particle_hams=[]
			for k in mismatched_ks:
				log.debug(f"matched k (pi units): {k/np.pi}")
				sub_cluster_1=LocalClusterBasis(k[0],state_params)
				sub_cluster_2=LocalClusterBasis(k[1],state_params)
				ham_dict_mismatched={
					'basis_classes':[sub_cluster_1,sub_cluster_2],
							'L':k[0].shape[0]*k[1].shape[0],
							'L_cluster':k[0].shape[0],
							'V':V,
							't':t,
							'U':U,
							'mu':mu,			
				}

				mismatched_ham=QuickHubbard1D(ham_dict_mismatched)
				mismatched_single_particle=single_particle_block(mismatched_ham,spin='up')
				mismatched_evals,mismatched_evecs=np.linalg.eigh(mismatched_single_particle)

				eigvals.append(mismatched_evals)
				single_particle_hams.append(mismatched_single_particle)
			


			
			mismatched_combined_eigvals=np.array(eigvals)
			return mismatched_combined_eigvals,single_particle_hams,mismatched_ks
			

			
		ham_dict_matched_base = {
				'V': V,
				't': t,
				'mu': mu,
				'U': U,
			}
		
		matched_combined_evals,matched_single_particle_hams,matched_ks=get_single_matched(cluster_size,lattice_points,ham_dict_matched_base,state_params)
		
		
		mismatched_combined_eigvals,mismatched_single_particle_hams,mismatched_ks=get_single_mismatched()

		print(f'matched ks shape: {matched_ks.shape},mismatched ks shape: {mismatched_ks.shape}')
		print(f'matched ks: {matched_ks/np.pi},mismatched ks: {mismatched_ks/np.pi}')
		#return None
		print(f'eigvals shape: {matched_combined_evals},mismatched eigvals shape: {mismatched_combined_eigvals.shape}')
		print(f'eigvals matched: {matched_combined_evals}\n eigvals mismatched: {mismatched_combined_eigvals}')

		try:
			np.testing.assert_array_almost_equal(
				matched_combined_evals,
				mismatched_combined_eigvals,
				err_msg="Clusters didn't match"
			)
			log.info(f"✓ Test PASSED for four site single particle")
		except AssertionError as e:
			log.error(f"✗ Test FAILED for four site single particle.")
			raise AssertionError(f"Energy mismatch for four site single particle") from e	

		
		#return None


		#print(f'mismatched spectrum: {mis_eigvals[0].round(2)}')
		
		log.debug('matched case hams')
		for cluster in matched_single_particle_hams:
			for ham in cluster:
				labels = [f"|site {i}⟩" for i in range(ham.shape[0])]
				df = matrix_to_dataframe(ham, labels, precision=2)
				print_matrix(df, style="tabulate", tablefmt="grid")
				log.debug("\n" + tabulate.tabulate(df.values, headers=df.columns, tablefmt="grid", showindex=True))
		
		log.debug('mismatched case ham')
		for ham in mismatched_single_particle_hams:
			
			labels = [f"|site {i}⟩" for i in range(ham.shape[0])]

			df = matrix_to_dataframe(ham, labels, precision=2)
			print_matrix(df, style="tabulate", tablefmt="grid")
			log.debug("\n" + tabulate.tabulate(df.values, headers=df.columns, tablefmt="grid", showindex=True))
		

		log.debug(f'matched spectrum: {matched_combined_evals.round(2)}')
		log.debug(f'mismatched spectrum: {mismatched_combined_eigvals.round(2)}')

		return None
	
	def test_manybody_sum_single_particle(self):
		"""
		A test to check whether the many-body spectrum reduces to a sum of one-particle spectra.
		"""
		

		return None




		

		
		
		
		
		


