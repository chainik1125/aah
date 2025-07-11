"""
This module is meant to generate the Hamiltonian on a given cluster.
For performance, you should generate a template and fill in the Hamiltonian
from a template of values.
There's a slightly tricky issue of how the Hamiltonian interacts with the clusters
since you're really defining an interaction cluster rather than a full Hamiltonian cluster.
"""
import logging
from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.global_params import StatesParams,HamiltonianParams
from aah_code.utils import mu_tilde_coefficient,cosine_dispersion
from tenpy.models import CouplingMPOModel,NearestNeighborModel,lattice
import tenpy as tp
from tenpy.algorithms import exact_diag
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set Plotly to use browser renderer to avoid nbformat issues
pio.renderers.default = "browser"


logger=logging.getLogger(__name__)


class Hubbard1D(CouplingMPOModel, NearestNeighborModel):
	"""
	Input is a dictionary called model_params that includes:
		'L' (length), 'bc' ('open' or 'periodic'), 'bc_MPS' ('finite' or 'infinite'),
		't' (hopping strength), 'U' (Hubbard interaction strength), 'filling' (1 is half-filling)
	"""

	# Initialize spin-1/2 fermion d.o.f. on each site
	def init_sites(self, model_params):
		# Remove both particle number and spin conservation to allow DMRG to explore all sectors
		if 'basis_class' in model_params:
			site=model_params['basis_class'].init_sites()
		else:
			site = tp.networks.site.SpinHalfFermionSite(cons_N=None, cons_Sz=None)
			raise Warning("No basis class provided - using default spin-1/2 fermion site")
			
		return site

	# Set 1D lattice
	def init_lattice(self, model_params):
		if 'basis_class' in model_params:
			# Use cluster size from basis_class
			L = len(model_params['basis_class'].cluster_k_points)
			bc = 'periodic'  # always use 'open'
			bc_MPS = 'finite'  # 'infinite' does iDMRG (still use open in 'bc')
			lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites(model_params))
		else:
			L = model_params.get('L', 4) # size with default
			bc = 'periodic'  # always use 'open'
			bc_MPS = 'finite'  # 'infinite' does iDMRG (still use open in 'bc')
			lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites(model_params))
			raise Warning("No basis class provided - using default chain with period chain bc and finite MPS ")
		
		return lat

	def init_terms(self, model_params):
		# default is U=1, t=1
		U = model_params.get('U', 0.0)
		t = model_params.get('t', 0.0)
		mu_0 = model_params.get('mu', 0.0)
		V=model_params.get('V', 0.0)
		L_cells = self.lat.Ls[0] 
		# nearest neighbor hopping -t
		if 'basis_class' in model_params:
			basis_object=model_params['basis_class']
			mu_tilde=mu_tilde_coefficient(t,basis_object,dispersion=cosine_dispersion)

			#Add the mu_tilde term and the mu_0 term
			mu_eff=mu_0-mu_tilde
			for alpha in range(len(self.lat.unit_cell)):
				self.add_onsite(-mu_eff, alpha, 'Nu')  # chemical potential n_up
				self.add_onsite(-mu_eff, alpha, 'Nd')  # chemical potential n_down
		
			#Add the t_tilde term

			number_cell_positions=len(self.lat.unit_cell_positions)
			if number_cell_positions>1:
				raise ValueError("I haven't implemented the t_tilde term for more than one unit cell position")
			else:
				# n_alpha_sites=len(self.lat.unit_cell)
				# for alpha in range(n_alpha_sites):
				# 	for beta in range(alpha+1,n_alpha_sites):
				# 		dx=(alpha)-(beta)
				# 		t_tilde=(1/n_alpha_sites)*np.array([2*t*np.cos(dx*2*np.pi*j/n_alpha_sites) for j in range(n_alpha_sites)]).sum()
				# 		self.add_coupling(t_tilde, alpha, 'Cdd', beta, 'Cd', dx, plus_hc=True)  # Cdagger_down C_down + h.c.
				# 		self.add_coupling(t_tilde, alpha, 'Cdu', beta, 'Cu', dx, plus_hc=True)  # Cdagger_up C_up + h.c.

						 # number of unit cells in the 1-D chain:contentReference[oaicite:6]{index=6}
				for dx in range(1, L_cells//2+1):      # 1 … L_cells-1   (dx = 0 already handled)
					# here beta could be alpha (same orbital) or something else
					for alpha in range(len(self.lat.unit_cell)):
						#t_tilde = 2 * t * np.cos(dx * 2 * np.pi / L_cells) / L_cells
						cluster_k_points=basis_object.cluster_k_points
						#There are two factor of 1/2:
						#1. Comes from the 1/2 in the t_tilde definition
						#2. Comes from double counting when including the hc - if you get confused about
						# this again remember the two site model hopping eigenenergies are not \pm 2t but \pm t !
						t_tilde=(1/2)*(1/L_cells)*np.array([2*t*np.cos(cluster_k_points[j])*(1/2)*2*t*np.cos(dx*2*np.pi*j/L_cells) for j in range(L_cells)]).sum()
						self.add_coupling(t_tilde, alpha, 'Cdd', alpha, 'Cd', dx, plus_hc=True)
						self.add_coupling(t_tilde, alpha, 'Cdu', alpha, 'Cu', dx, plus_hc=True)
			
			
			
			
			#Add the onsite alpha U
			for alpha in range(len(self.lat.unit_cell)):
				self.add_onsite(U, alpha, 'NuNd')  # Hubbard n_up n_down term
				
			# #Add the onsite V
			# for alpha in range(len(self.lat.unit_cell)):
			# 	if abs(V) > 0:        # i = 0 … L-1
			# 		sign =  +V/2 if (alpha % 2 == 0) else -V/2   # even sites +V, odd sites –V
			# 		self.add_onsite(sign, alpha, 'Nu')       # n↑ part
			# 		self.add_onsite(sign, alpha, 'Nd')

			if abs(V) > 0:
				# shape (L_cells,)  →  [+V/2, -V/2, +V/2, …]
				stagger = np.asarray([ +V/2 if (x % 2 == 0) else -V/2
									for x in range(L_cells) ])
				for alpha in range(len(self.lat.unit_cell)):      # usually alpha == 0
					self.add_onsite(stagger, alpha, 'Nu')         # n↑   term
					self.add_onsite(stagger, alpha, 'Nd')         # n↓   term

		else:
			raise ValueError("No basis class provided")
		
class QuickHubbard1D(CouplingMPOModel):
	"""
	Input is a dictionary called model_params that includes:
		'L' (length), 'bc' ('open' or 'periodic'), 'bc_MPS' ('finite' or 'infinite'),
		't' (hopping strength), 'U' (Hubbard interaction strength), 'filling' (1 is half-filling)
	"""

	# Initialize spin-1/2 fermion d.o.f. on each site
	def init_sites(self, model_params):
		# Remove both particle number and spin conservation to allow DMRG to explore all sectors
		if 'basis_classes' in model_params:
			site=model_params['basis_classes'][0].init_sites()
		else:
			site = tp.networks.site.SpinHalfFermionSite(cons_N=None, cons_Sz=None)
			raise Warning("No basis class provided - using default spin-1/2 fermion site")
			
		return site

	# Set 1D lattice
	def init_lattice(self, model_params):
		if 'basis_classes' in model_params:
			# Use the total system size L instead of just the first basis class
			L = model_params['L'] # total system size
			bc = 'periodic'  # periodic boundary conditions
			bc_MPS = 'finite'  # finite MPS
			lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites(model_params))
			# Keep periodic boundary conditions for V coupling
			lat.bc = [True]
		else:
			
			L = model_params['L'] # size
			bc = 'periodic'  # always use 'open'
			bc_MPS = 'finite'  # 'infinite' does iDMRG (still use open in 'bc')
			lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites(model_params))

			raise Warning("No basis class provided - using default chain with period chain bc and finite MPS ")
		
		return lat

	def init_terms(self, model_params):
		
		# default is U=1, t=1
		U = model_params.get('U', 0.0)
		t = model_params.get('t', 0.0)
		mu_0 = model_params.get('mu', 0.0)
		V=model_params.get('V', 0.0)
		L_cells = model_params.get('L',4)
		L_int_cluster=model_params['L_cluster']
		# nearest neighbor hopping -t
		if 'basis_classes' in model_params:
			for bc_index,basis_class in enumerate(model_params['basis_classes']):
				basis_object=model_params['basis_classes'][bc_index]
				mu_tilde=mu_tilde_coefficient(t,basis_object,dispersion=cosine_dispersion)
				
				# Calculate the site range for this subcluster
				cluster_size = len(basis_object.cluster_k_points)
				L_start = bc_index * cluster_size
				L_end = L_start + cluster_size

				#Add the mu_tilde term and the mu_0 term only to sites in this subcluster
				mu_eff=mu_0-mu_tilde

				
				

				for alpha in range(len(self.lat.unit_cell)):
					for site_idx in range(L_start, L_end):
						#self.add_onsite(-mu_eff/4, alpha, 'Nu', site_idx)  # chemical potential n_up
						#self.add_onsite(-mu_eff/4, alpha, 'Nd', site_idx)  # chemical potential n_down

						self.add_onsite_term(-mu_eff, site_idx, 'Nu')
						self.add_onsite_term(-mu_eff, site_idx, 'Nd')
			
				#Add the t_tilde term

				number_cell_positions=len(self.lat.unit_cell_positions)
				if number_cell_positions>1:
					raise ValueError("I haven't implemented the t_tilde term for more than one unit cell position")
				else:
					# n_alpha_sites=len(self.lat.unit_cell)
					# for alpha in range(n_alpha_sites):
					# 	for beta in range(alpha+1,n_alpha_sites):
					# 		dx=(alpha)-(beta)
					# 		t_tilde=(1/n_alpha_sites)*np.array([2*t*np.cos(dx*2*np.pi*j/n_alpha_sites) for j in range(n_alpha_sites)]).sum()
					# 		self.add_coupling(t_tilde, alpha, 'Cdd', beta, 'Cd', dx, plus_hc=True)  # Cdagger_down C_down + h.c.
					# 		self.add_coupling(t_tilde, alpha, 'Cdu', beta, 'Cu', dx, plus_hc=True)  # Cdagger_up C_up + h.c.

							# number of unit cells in the 1-D chain:contentReference[oaicite:6]{index=6}
					# Only add t_tilde couplings within this subcluster
					# For 2-site clusters, only add nearest neighbor hopping within the cluster
					if cluster_size == 2:  # Only nearest neighbor within 2-site clusters
						cluster_k_points=basis_object.cluster_k_points
						# Calculate t_tilde for dx=1 within this cluster
						t_tilde=(1/2)*(1/cluster_size)*np.array([2*t*np.cos(cluster_k_points[j])*(1/2)*2*t*np.cos(1*2*np.pi*j/cluster_size) for j in range(cluster_size)]).sum()
						
						# Add hopping between the two sites in this cluster: L_start <--> L_start+1
						i1, i2 = L_start, L_start + 1
						self.add_coupling_term(t_tilde, i1, i2, 'Cdd', 'Cd', plus_hc=True)
						self.add_coupling_term(t_tilde, i1, i2, 'Cdu', 'Cu', plus_hc=True)
				
				
				
				
				#Add the onsite alpha U only to sites in this subcluster
				# for alpha in range(len(self.lat.unit_cell)):
				# 	for site_idx in range(L_start, L_end):
				# 		self.add_onsite(U, alpha, 'NuNd', site_idx)  # Hubbard n_up n_down term
			for alpha in range(len(self.lat.unit_cell)):
				self.add_onsite(U, alpha, 'NuNd')  # Hubbard n_up n_down term
					
			#Add V as next-nearest neighbor coupling (site 0<->2, site 1<->3)
			if abs(V) > 0:
				# Use dx=2 for next-nearest neighbor with periodic BC
				for alpha in range(len(self.lat.unit_cell)):
					#NOTE! I think it's V/4 here because I added it as a NNN hopping term
					#with hermitian conjugates in each so I double count.
					self.add_coupling(V/4, alpha, 'Cdd', alpha, 'Cd', 2, plus_hc=True)
					self.add_coupling(V/4, alpha, 'Cdu', alpha, 'Cu', 2, plus_hc=True)

		else:
			raise ValueError("No basis class provided")

class SpectrumSolver():
	"""
	Class that will solve for the spectrum
	save: either false or folder path
	"""
	def __init__(self,hamiltonian,cluster_object:LocalClusterBasis,solver:str='tenpy_ED',states_retained:Union[int,'all']='all',save:Union[bool,str]=False):
		self.hamiltonian=hamiltonian
		self.solver=solver
		self.states_retained=states_retained
		self.cluster_object=cluster_object
		self.save=save

	def solve_spectrum(self):
		if self.solver=='tenpy_ED':
			#np_ham=tp.algorithms.exact_diag.get_numpy_Hamiltonian(self.hamiltonian)
			
			
			ed = exact_diag.ExactDiag(self.hamiltonian)                 # solver instance  :contentReference[oaicite:0]{index=0}
			
			# Use MPO-based exact diagonalization to handle next-nearest neighbor couplings
			ed.build_full_H_from_mpo()
			ed.full_diagonalization()                    # fills ed.full_H  :contentReference[oaicite:1]{index=1}
			E = ed.E
			V = ed.V
			#E0, psi_vec = ed.groundstate()
			#
			
			
			eigvals=[]
			eigvecs=[]
			n_ups=[]
			n_downs=[]
			for i,E_i in enumerate(E):
				psi_vec=V[...,i]
				psi_mps=ed.full_to_mps(psi_vec)
				n_up=psi_mps.expectation_value('Nu')
				n_down=psi_mps.expectation_value('Nd')
				eigvals.append(E_i)
				eigvecs.append(psi_vec)
				n_ups.append(n_up)
				n_downs.append(n_down)
			n_ups=np.stack(n_ups,axis=0)
			n_downs=np.stack(n_downs,axis=0)
			n_tot=n_ups+n_downs
			
			return eigvals,eigvecs,n_ups,n_downs,n_tot
			#TODO:add functionality to efficiently get smaller number of total states.

		else:
			raise ValueError(f'Solver {self.solver} not implemented yet')
		
#Maybe I'll leave this for later
#class SpectrumContainer():

class FullSpectrum():
	"""
	Class to get the full spectrum of the Hamiltonian
	None temperature is zero temperature
	"""
	def __init__(self,clustered_k_points:np.ndarray,state_params:StatesParams,physical_params:HamiltonianParams,temperature:Union[None,float]=None):
		self.clustered_k_points=clustered_k_points
		self.state_params=state_params
		self.physical_params=physical_params
		self.temperature=temperature

		

	def get_full_spectrum(self):
		k_points=[]
		energy_spectrum=[]
		number_spectrum=[]
		spin_spectrum=[]
		for cluster_k in self.clustered_k_points:		
			cluster_object=LocalClusterBasis(cluster_k,self.state_params)
			hamiltonian_object=Hubbard1D({'basis_class':cluster_object,
											'V':self.physical_params.V,
											't':self.physical_params.hopping,
											'mu':self.physical_params.mu_0,
											'U':self.physical_params.U,
											})
			
			
			spectrum_solver=SpectrumSolver(hamiltonian_object,cluster_object)
			eigvals,eigvecs,n_ups,n_downs,n_tot=spectrum_solver.solve_spectrum()
			
			k_points.append(cluster_k)
			energy_spectrum.append(eigvals)
			number_spectrum.append(n_tot)
			spin_spectrum.append(np.array([n_ups,n_downs]))

			
		
		k_points=np.stack(k_points,axis=0)
		energy_spectrum=np.stack(energy_spectrum,axis=0)
		number_spectrum=np.stack(number_spectrum,axis=0)
		spin_spectrum=np.stack(spin_spectrum,axis=0)

		logger.debug(f'spin spectrum shape: {spin_spectrum.shape}')
		

		return k_points,energy_spectrum,number_spectrum,spin_spectrum
		

	def get_cluster_thermodynamic_expectations(self,full_spectrum_4tuple:tuple,temperature:Union[None,float]=None):
		cluster_energy_expectations=[]
		cluster_number_expectations=[]
		cluster_spin_expectations=[]

		k_points,full_energy_spectrum,full_number_spectrum,full_spin_spectrum=full_spectrum_4tuple

		for i,k_point in enumerate(k_points):
			energy_spectrum=full_energy_spectrum[i]
			number_spectrum=full_number_spectrum[i]
			spin_spectrum=full_spin_spectrum[i]

			if temperature is None:
				logger.info('Temperature is None - returning zero temperature expectations')
				cluster_energy_argmin=np.argmin(energy_spectrum,axis=-1)
				cluster_energy_expectations.append(energy_spectrum[cluster_energy_argmin])
				cluster_number_expectations.append(number_spectrum[cluster_energy_argmin])
				spin_multiplier=np.array([1,-1])#to give +1 to up spins and -1 to down spins.
				# spin_spectrum[i] has shape (2, 256, 4): (spins, eigenstates, sites)
				# We need to extract the ground state and sum over sites for each spin type
				ground_state_spins = spin_spectrum[:, cluster_energy_argmin, :]  # shape (2, 4)
				spin_polarization = np.sum(ground_state_spins * spin_multiplier[:, np.newaxis], axis=(0,1))  # sum over spins and sites
				cluster_spin_expectations.append(spin_polarization)
			else:
				logger.info(f'Temperature is {temperature} - returning temperature dependent expectations')
				beta=1/temperature
				#TODO:CHECK!! I think you DONT include mu_0 N if you already added this 
				#term to the Hamiltonian when finding the energies
				sum_partition_function=np.sum(np.exp(-beta*(energy_spectrum)))
				cluster_energy_expectations.append(np.sum(energy_spectrum*np.exp(-beta*energy_spectrum))/sum_partition_function)
				cluster_number_expectations.append(np.sum(number_spectrum*np.exp(-beta*energy_spectrum))/sum_partition_function)
				spin_multiplier=np.array([1,-1])#to give +1 to up spins and -1 to down spins.
				# Handle temperature-dependent case with proper broadcasting
				boltzmann_weights = np.exp(-beta*energy_spectrum)  # shape (256,)
				# spin_spectrum has shape (2, 256, 4), multiply by weights and sum
				weighted_spins = np.sum(spin_spectrum * boltzmann_weights[np.newaxis, :, np.newaxis], axis=1)  # sum over eigenstates, shape (2, 4)
				spin_polarization = np.sum(weighted_spins * spin_multiplier[:, np.newaxis], axis=(0,1))  # sum over spins and sites
				cluster_spin_expectations.append(spin_polarization / sum_partition_function)
		
		system_energy=np.sum(cluster_energy_expectations)
		system_number=np.sum(cluster_number_expectations)
		system_spin=np.sum(cluster_spin_expectations)

		system_expectations=(system_energy,system_number,system_spin)
		cluster_expectations=(cluster_energy_expectations,cluster_number_expectations,cluster_spin_expectations)
		
		return system_expectations,cluster_expectations
	


				
class MismatchedQuick():
	def __init__(self,cluster_experiment:ClusterExperiment,physical_params,V_k_period:int=2):
		self.cluster_experiment=cluster_experiment
		self.V_k_period=V_k_period
		self.physical_params=physical_params
	
	def recluster(self):
		int_cluster_ks_idxs=self.cluster_experiment.generate_clusters(return_indices=True)
		lattice_points=self.cluster_experiment.lattice_points
		logger.info(f'int cluster K shape: {int_cluster_ks_idxs.shape}')
		v_k_step=2*np.pi*(self.V_k_period/lattice_points)

		accounted_k_clusters=set()
		new_k_clusters_idxs=[]
		for i in range(len(int_cluster_ks_idxs)):
			cluster_k_idxs=int_cluster_ks_idxs[i]
			# Apply modular arithmetic to handle periodic boundary conditions
			stepped_k_idxs=(cluster_k_idxs+self.V_k_period) % lattice_points
			
			# Convert arrays to tuples for set membership checking
			cluster_tuple = tuple(cluster_k_idxs.flatten())
			stepped_tuple = tuple(stepped_k_idxs.flatten())
			
			if cluster_tuple in accounted_k_clusters:
				continue
			else:
				# Stack the two size-2 clusters to maintain cluster index dimension
				paired_clusters = np.stack([cluster_k_idxs, stepped_k_idxs], axis=0)
				new_k_clusters_idxs.append(paired_clusters)
				accounted_k_clusters.add(stepped_tuple)
				accounted_k_clusters.add(cluster_tuple)
		
		if len(new_k_clusters_idxs) > 0:
			new_k_clusters_idxs=np.stack(new_k_clusters_idxs,axis=0)
			
			# Convert indices back to actual k-point values
			k_spacing = 2*np.pi/lattice_points
			new_k_clusters = new_k_clusters_idxs * k_spacing
			
			logger.info(f'shape new_k_clusters:{new_k_clusters.shape}')
			return new_k_clusters,new_k_clusters_idxs
		else:
			logger.info('No new k-clusters found')
			return np.array([])

def get_spectra(cluster_ks, state_params, physical_params):
	#Lets try to make the hamiltonian

	k_points=[]
	energy_spectrum=[]
	number_spectrum=[]
	spin_spectrum=[]

	for cluster_k in cluster_ks:
		total_cluster_size=cluster_k.shape[0]*cluster_k.shape[1]
		test_basis_1=LocalClusterBasis(cluster_k[0],state_params)
		test_basis_2=LocalClusterBasis(cluster_k[1],state_params)
		test_ham=QuickHubbard1D({'basis_classes':[test_basis_1,test_basis_2],
					'L':total_cluster_size,
					'L_cluster':cluster_k.shape[0],
					'V':physical_params.V,
					't':physical_params.hopping,
					'U':physical_params.U,
					'mu':physical_params.mu_0,
					})
		
		
		
		
		solver=SpectrumSolver(test_ham,None)#basis object never explicitly used anyway
		eigvals,eigvecs,n_ups,n_downs,n_tot=solver.solve_spectrum()
		


		k_points.append(cluster_k)
		energy_spectrum.append(eigvals)
		number_spectrum.append(n_tot)
		spin_spectrum.append(np.array([n_ups,n_downs]))

			
		
	k_points=np.stack(k_points,axis=0)
	energy_spectrum=np.stack(energy_spectrum,axis=0)
	number_spectrum=np.stack(number_spectrum,axis=0)
	spin_spectrum=np.stack(spin_spectrum,axis=0)

	logger.info(f'spin spectrum shape: {spin_spectrum.shape}')
		

	return k_points,energy_spectrum,number_spectrum,spin_spectrum



		

		

		

	return None

	




def test_quick_mismatched(lattice_points,cluster_size,physical_params):
	#lattice_points=16
	#cluster_size=2
	#physical_params=HamiltonianParams(U=10,V=5,hopping=1,mu_0=10/2)
	state_params=StatesParams(spin_states=2)
	int_lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//4)
	print(type(int_lattice_object))
	test=MismatchedQuick(int_lattice_object,physical_params,lattice_points//2)
	cluster_ks,cluster_idxs=test.recluster()
	print(f'cluster ks shape: {cluster_idxs.shape}')
	print(f'cluster idxs: {cluster_idxs}')

	#k_points,energies,number_spectrum,spin_spectrum=get_spectra(cluster_ks)
	spectra_4tuple=get_spectra(cluster_ks, state_params, physical_params)
	
	print(f'k points shape: {spectra_4tuple[0].shape},\n energies shape: {spectra_4tuple[1].shape},\n number_spectrum shape: {spectra_4tuple[2].shape}, spin spectrum shape: {spectra_4tuple[3].shape}')

	full_spectrum_obj=FullSpectrum(None,state_params,physical_params,None)
	system_expectations,cluster_expectations=full_spectrum_obj.get_cluster_thermodynamic_expectations(spectra_4tuple,None)
	return system_expectations,cluster_expectations
	# print(f"system energy density: {system_expectations[0]/lattice_points}",
	#    	f"system energy density mu_subtracted: {(system_expectations[0]+system_expectations[1]*physical_params.mu_0)/lattice_points} "
	#    	f"system filling density: {system_expectations[1]/lattice_points}",
	# 	f"system spin density:{system_expectations[2]/lattice_points}")


	
		
	
def inspect_hamiltonian_terms(hamiltonian):
	"""
	Inspect and tabulate all terms in a TenPy Hamiltonian
	"""
	print("="*80)
	print("HAMILTONIAN TERMS INSPECTION")
	print("="*80)
	
	# Method 1: Summary table with actual term values
	print("\n1. SUMMARY TABLE:")
	print("-" * 80)
	print("| Term Type | Site(s) | Operator(s) | Strength | dx | Description |")
	print("|-----------|---------|-------------|----------|--------|-------------|")
	
	# Onsite terms - need to extract from the OnsiteTerms objects
	if hasattr(hamiltonian, 'onsite_terms') and hamiltonian.onsite_terms:
		for site_idx, onsite_term in hamiltonian.onsite_terms.items():
			# Try to access the terms dictionary directly
			if hasattr(onsite_term, '_terms'):
				for op_name, strength in onsite_term._terms.items():
					description = f"Site {site_idx}"
					print(f"| Onsite    | {site_idx:7} | {op_name:11} | {strength:8.3f} | N/A    | {description:11} |")
	
	# Coupling terms - need to extract from CouplingTerms objects
	if hasattr(hamiltonian, 'coupling_terms') and hamiltonian.coupling_terms:
		for term_name, coupling_term in hamiltonian.coupling_terms.items():
			if hasattr(coupling_term, '_terms'):
				for term_key, strength in coupling_term._terms.items():
					# term_key format: ((u1, u2), (i1, i2), dx)
					u_pair = term_key[0] if len(term_key) > 0 else (0,0)
					site_pair = term_key[1] if len(term_key) > 1 else (0,0)
					dx = term_key[2] if len(term_key) > 2 else 0
					sites = f"{site_pair}"
					description = f"dx={dx}, u={u_pair}"
					print(f"| Coupling  | {sites:7} | {term_name:11} | {strength:8.3f} | {dx:6} | {description:11} |")
	
	# Method 2: Alternative inspection using dir() 
	print("\n2. DETAILED TERM INSPECTION:")
	print("-" * 50)
	
	# Try to access terms via different attributes
	for attr in ['onsite_terms', 'coupling_terms']:
		if hasattr(hamiltonian, attr):
			terms_dict = getattr(hamiltonian, attr)
			print(f"\n{attr}:")
			for key, term_obj in terms_dict.items():
				print(f"  {key}: {type(term_obj)}")
				# Try to find the actual terms
				for attr_name in dir(term_obj):
					if 'term' in attr_name.lower() and not attr_name.startswith('_'):
						try:
							value = getattr(term_obj, attr_name)
							print(f"    .{attr_name}: {value}")
						except:
							pass
	
	print("\n" + "="*80)

def test_hamiltonian_inspection():
	"""
	Test function to inspect a single QuickHubbard1D Hamiltonian
	"""
	print("Creating test QuickHubbard1D Hamiltonian...")
	
	# Create test parameters
	state_params = StatesParams(spin_states=2)
	physical_params = HamiltonianParams(U=2.0, V=0.5, hopping=1.0, mu_0=1.0)
	
	# Create two test clusters with different k-points
	cluster_k_1 = np.array([0.0, np.pi])  # k-points for cluster 1
	cluster_k_2 = np.array([np.pi/2, 3*np.pi/2])  # k-points for cluster 2
	
	test_basis_1 = LocalClusterBasis(cluster_k_1, state_params)
	test_basis_2 = LocalClusterBasis(cluster_k_2, state_params)
	
	# Create QuickHubbard1D with these clusters
	test_ham = QuickHubbard1D({
		'basis_classes': [test_basis_1, test_basis_2],
		'L': 4,  # 2 sites per cluster × 2 clusters
		'L_cluster': 2,
		'V': physical_params.V,
		't': physical_params.hopping,
		'U': physical_params.U,
		'mu': physical_params.mu_0,
	})
	
	print(f"Hamiltonian created with lattice size: {test_ham.lat.Ls}")
	print(f"Physical params: U={physical_params.U}, V={physical_params.V}, t={physical_params.hopping}, mu={physical_params.mu_0}")
	
	# Inspect the Hamiltonian terms
	inspect_hamiltonian_terms(test_ham)
	
	return test_ham


def quick_spectrum_test_vary_U(U_values,V):
	fig=make_subplots(
		rows=1, cols=2,
		subplot_titles=['Energy vs U', 'Filling vs U']
	)
	lattice_points=16
	cluster_size=2
	subtracted_energies_4site=[]
	fillings_4site=[]
	for U in U_values:
		physical_params=HamiltonianParams(U,V,1,U/2)
		system_expectations,cluster_expectations=test_quick_mismatched(lattice_points,cluster_size,physical_params)
		total_energy,total_filling,total_spin=system_expectations
		subtracted_energies_4site.append((total_energy+physical_params.mu_0*total_filling)/lattice_points)
		fillings_4site.append(total_filling/lattice_points)
	
	# Add energy trace
	fig.add_trace(
		go.Scatter(
			x=U_values,
			y=np.array(subtracted_energies_4site),
			mode='lines+markers',
			name='Energy Density',
			line=dict(color='blue', width=2),
			marker=dict(size=8)
		),
		row=1, col=1
	)
	
	# Add filling trace
	fig.add_trace(
		go.Scatter(
			x=U_values,
			y=np.array(fillings_4site),
			mode='lines+markers',
			name='Filling Density',
			line=dict(color='red', width=2),
			marker=dict(size=8)
		),
		row=1, col=2
	)
	
	# Update layout and axis labels
	fig.update_layout(
		title=f'Cluster Method Results vs U (V={V})',
		showlegend=True
	)
	fig.update_xaxes(title_text="U", row=1, col=1)
	fig.update_yaxes(title_text="Energy Density", row=1, col=1)
	fig.update_xaxes(title_text="U", row=1, col=2)
	fig.update_yaxes(title_text="Filling Density", row=1, col=2)

	return fig

def compare_all_methods_vs_U(U_values, V=0, t=1):
	"""
	Compare DMRG, two-site, cluster method (from main.py), and four-site cluster method
	Similar to compare_half_filling_U_fixed_V but with all methods
	"""
	from aah_code.main import run_cluster_method, run_dmrg_method, run_twosite
	
	fig = make_subplots(
		rows=1, cols=2,
		subplot_titles=['Energy Density vs U (Half-Filling)', 'Filling Density vs U (Half-Filling)']
	)
	
	# Storage for results
	energies_dmrg = []
	energies_twosite = []
	energies_cluster_2site = []
	energies_cluster_4site = []
	
	fillings_dmrg = []
	fillings_twosite = []
	fillings_cluster_2site = []
	fillings_cluster_4site = []
	
	system_size = 100  # For DMRG and 2-site cluster method
	lattice_points = 100  # For 4-site cluster method
	cluster_size = 2
	chi = 32
	
	print(f"Comparing all methods for U values: {U_values}")
	print(f"V = {V}, t = {t}")
	
	for U in U_values:
		mu_0 = U / 2  # Half-filling condition
		print(f"\n--- U = {U}, μ₀ = {mu_0} ---")
		
		# 1. DMRG method
		print("Running DMRG...")
		energy_dmrg, filling_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
		#finite dmrg
		# energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg * system_size)
		# energies_dmrg.append(energy_dmrg_subtracted / system_size)
		# fillings_dmrg.append(filling_dmrg)
		#infinite dmrg
		energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)
		energies_dmrg.append(energy_dmrg_subtracted)
		fillings_dmrg.append(filling_dmrg)
		# 2. Two-site analytical
		print("Running two-site analytical...")
		energy_twosite = run_twosite(U, mu_0, V, t, system_size)
		energies_twosite.append(energy_twosite)
		fillings_twosite.append(1.0)  # Half-filling by construction
		
		# 3. Cluster method (2-site clusters from main.py)
		print("Running 2-site cluster method...")
		energy_cluster_2, filling_cluster_2 = run_cluster_method(U, mu_0, V, t, system_size)
		energy_cluster_2_subtracted = energy_cluster_2 + mu_0 * filling_cluster_2
		energies_cluster_2site.append(energy_cluster_2_subtracted / system_size)
		fillings_cluster_2site.append(filling_cluster_2 / system_size)
		
		# 4. Four-site cluster method (from quick_spectrum_test_vary_U)
		print("Running 4-site cluster method...")
		physical_params = HamiltonianParams(U, V, t, mu_0)
		system_expectations, cluster_expectations = test_quick_mismatched(lattice_points, cluster_size, physical_params)
		total_energy, total_filling, total_spin = system_expectations
		energy_cluster_4_subtracted = (total_energy + physical_params.mu_0 * total_filling) / lattice_points
		energies_cluster_4site.append(energy_cluster_4_subtracted)
		fillings_cluster_4site.append(total_filling / lattice_points)
		
		print(f"DMRG:              E={energies_dmrg[-1]:.3f}, n={fillings_dmrg[-1]:.3f}")
		print(f"Two-site:          E={energies_twosite[-1]:.3f}, n={fillings_twosite[-1]:.3f}")
		print(f"Cluster (2-site):  E={energies_cluster_2site[-1]:.3f}, n={fillings_cluster_2site[-1]:.3f}")
		print(f"Cluster (4-site):  E={energies_cluster_4site[-1]:.3f}, n={fillings_cluster_4site[-1]:.3f}")
	
	# Convert to numpy arrays
	energies_dmrg = np.array(energies_dmrg)
	energies_twosite = np.array(energies_twosite)
	energies_cluster_2site = np.array(energies_cluster_2site)
	energies_cluster_4site = np.array(energies_cluster_4site)
	
	fillings_dmrg = np.array(fillings_dmrg)
	fillings_twosite = np.array(fillings_twosite)
	fillings_cluster_2site = np.array(fillings_cluster_2site)
	fillings_cluster_4site = np.array(fillings_cluster_4site)
	
	# Plot energies (left column)
	fig.add_trace(
		go.Scatter(
			x=U_values, y=energies_dmrg,
			mode='lines+markers',
			name='DMRG',
			line=dict(color='red', width=2),
			marker=dict(size=8)
		),
		row=1, col=1
	)
	
	fig.add_trace(
		go.Scatter(
			x=U_values, y=energies_twosite,
			mode='lines+markers',
			name='Two-site Analytical',
			line=dict(color='green', width=2),
			marker=dict(size=8)
		),
		row=1, col=1
	)
	
	fig.add_trace(
		go.Scatter(
			x=U_values, y=energies_cluster_2site,
			mode='lines+markers',
			name='Cluster (2-site)',
			line=dict(color='blue', width=2),
			marker=dict(size=8)
		),
		row=1, col=1
	)
	
	fig.add_trace(
		go.Scatter(
			x=U_values, y=energies_cluster_4site,
			mode='lines+markers',
			name='Cluster (4-site)',
			line=dict(color='purple', width=2),
			marker=dict(size=8)
		),
		row=1, col=1
	)
	
	# Plot fillings (right column)
	fig.add_trace(
		go.Scatter(
			x=U_values, y=fillings_dmrg,
			mode='lines+markers',
			name='DMRG',
			line=dict(color='red', width=2),
			marker=dict(size=8),
			showlegend=False
		),
		row=1, col=2
	)
	
	fig.add_trace(
		go.Scatter(
			x=U_values, y=fillings_twosite,
			mode='lines+markers',
			name='Two-site Analytical',
			line=dict(color='green', width=2),
			marker=dict(size=8),
			showlegend=False
		),
		row=1, col=2
	)
	
	fig.add_trace(
		go.Scatter(
			x=U_values, y=fillings_cluster_2site,
			mode='lines+markers',
			name='Cluster (2-site)',
			line=dict(color='blue', width=2),
			marker=dict(size=8),
			showlegend=False
		),
		row=1, col=2
	)
	
	fig.add_trace(
		go.Scatter(
			x=U_values, y=fillings_cluster_4site,
			mode='lines+markers',
			name='Cluster (4-site)',
			line=dict(color='purple', width=2),
			marker=dict(size=8),
			showlegend=False
		),
		row=1, col=2
	)
	
	# Update layout
	fig.update_layout(
		title=f'Method Comparison: DMRG vs Analytical vs Cluster Methods (V={V})',
		showlegend=True,
		height=500
	)
	
	# Update axes
	fig.update_xaxes(title_text="U", row=1, col=1)
	fig.update_yaxes(title_text="Energy Density", row=1, col=1)
	fig.update_xaxes(title_text="U", row=1, col=2)
	fig.update_yaxes(title_text="Filling Density", row=1, col=2)
	
	return fig

if __name__ == "__main__":
	print('main')

	# Test Hamiltonian inspection
	#test_hamiltonian_inspection()
	
	#exit('Inspected Hamiltonian - check that coupling is correct')
	
	#lattice_points=16
	#cluster_size=2
	#test_quick_mismatched(lattice_points,cluster_size)
	U_values=np.linspace(3,10,7)
	#fig=quick_spectrum_test_vary_U(U_values,V=0)
	#fig.show()

	compare_all_methods_vs_U(U_values,V=0,t=1).show()
	#fig.write_html("quick_spectrum_test.html")
	#print("Quick spectrum test saved as 'quick_spectrum_test.html'")

	
	exit('testing 4 site')

	#test the full spectrum code
	lattice_points=10
	cluster_size=2
	lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//2)
	k_points=lattice_object.generate_clusters()
	
	full_spectrum_object=FullSpectrum(k_points,state_params,physical_params)
	cluster_spectra=full_spectrum_object.get_full_spectrum()

	system_expectations,cluster_expectations=full_spectrum_object.get_cluster_thermodynamic_expectations(cluster_spectra,temperature=None)

	print(f'system energy: {system_expectations[0]},system number: {system_expectations[1]},system_spins: {system_expectations[2]}')


	


	
	

	

	

	
		  



