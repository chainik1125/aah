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
			lat=model_params['basis_class'].init_cluster_lattice()
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
			ed.build_full_H_from_bonds() 
			E,V=ed.full_diagonalization()                    # fills ed.full_H  :contentReference[oaicite:1]{index=1}
			#E0, psi_vec = ed.groundstate()
			#
			eigvals=[]
			eigvecs=[]
			n_ups=[]
			n_downs=[]
			for i,E_i in enumerate(eigvals):
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
		self.temperature=temperature

		

	def get_full_spectrum(self):
		k_points=[]
		energy_spectrum=[]
		number_spectrum=[]
		spin_spectrum=[]
		for cluster_k in self.clustered_k_points:
			cluster_object=LocalClusterBasis(cluster_k,state_params)
			hamiltonian_object=Hubbard1D({'basis_class':cluster_object,
											'V':physical_params.V,
											't':physical_params.hopping,
											'mu':0,
											'U':physical_params.U,
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
				cluster_spin_expectations.append(spin_spectrum[cluster_energy_argmin])
			else:
				logger.info(f'Temperature is {temperature} - returning temperature dependent expectations')
				beta=1/temperature
				#TODO:CHECK!! I think you DONT include mu_0 N if you already added this 
				#term to the Hamiltonian when finding the energies
				sum_partition_function=np.sum(np.exp(-beta*(energy_spectrum)))
				cluster_energy_expectations.append(np.sum(energy_spectrum*np.exp(-beta*energy_spectrum))/sum_partition_function)
				cluster_number_expectations.append(np.sum(number_spectrum*np.exp(-beta*energy_spectrum))/sum_partition_function)
				cluster_spin_expectations.append(np.sum(spin_spectrum*np.exp(-energy_spectrum/temperature))/sum_partition_function)
		
		system_energy=np.sum(cluster_energy_expectations)
		system_number=np.sum(cluster_number_expectations)
		system_spin=np.sum(cluster_spin_expectations)

		system_expectations=(system_energy,system_number,system_spin)
		cluster_expectations=(cluster_energy_expectations,cluster_number_expectations,cluster_spin_expectations)
		
		return system_expectations,cluster_expectations
	


				


	
	
		
	

if __name__ == "__main__":
	print('main')

	state_params=StatesParams(spin_states=2)
	physical_params=HamiltonianParams(U=3.0,V=2.0,hopping=1.0)
	cluster_k_points=np.array([[-np.pi],
								[0]])

	print(f'cluster_k_points shape:{cluster_k_points.shape}')

	test_basis=LocalClusterBasis(cluster_k_points,state_params)

	local_lattice=test_basis.init_cluster_lattice()
	#model = Hubbard1D({'L': 2, 'U':physical_params.U, 't':physical_params.hopping, 'bc':'open', 'bc_MPS':'finite', 'mu':physical_params.mu})
	ham_dict={'basis_class':test_basis,
		   		'V':physical_params.V,
				't':physical_params.hopping,
				'mu':0,
				'U':physical_params.U,
				}
	
	test_ham=Hubbard1D(ham_dict)

	#print(vars(test_ham.all_coupling_terms()))

	#test the full spectrum code
	lattice_object=ClusterExperiment(test_basis,physical_params)
	full_spectrum=FullSpectrum(lattice_object,state_params,physical_params)
	


	
	

	

	

	
		  



