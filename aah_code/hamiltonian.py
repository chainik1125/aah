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
import sys
import io
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from typing import Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tenpy.networks import mps
from tqdm import tqdm
from aah_code.quspin.quspin_hamiltonian import QuSpinHamiltonian
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from aah_code.real_space_dmrg import run_dmrg_method, get_gnd_infinite,get_gnd
# Set Plotly to use browser renderer to avoid nbformat issues
#pio.renderers.default = "browser"


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
			bc = 'open'  # always use 'open'
			bc_MPS = 'periodic'  # 'infinite' does iDMRG (still use open in 'bc')
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

			if abs(V) > 0:
				# shape (L_cells,)  →  [+V/2, -V/2, +V/2, …]
				stagger = np.asarray([ +V if (x % 2 == 0) else -V
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
			#bc = 'periodic'  # periodic boundary conditions
			#bc_MPS = 'finite'  # finite MPS
			bc='open'
			bc_MPS='finite'
			lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites(model_params))
			# Keep periodic boundary conditions for V coupling
			#lat.bc = [True]
		else:
			
			L = model_params['L'] # size
			#bc = 'periodic'  # always use 'open'
			#bc_MPS = 'finite'  # 'infinite' does iDMRG (still use open in 'bc')
			bc='open'
			bc_MPS='finite'
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

				
				# for alpha in range(len(self.lat.unit_cell)):
				# 	for site_idx in range(L_start, L_end):
				# 		#self.add_onsite(-mu_eff/4, alpha, 'Nu', site_idx)  # chemical potential n_up
				# 		#self.add_onsite(-mu_eff/4, alpha, 'Nd', site_idx)  # chemical potential n_down
				# 		#TODO:change to add_onsite
				# 		self.add_onsite_term(-mu_eff, site_idx, 'Nu')
				# 		self.add_onsite_term(-mu_eff, site_idx, 'Nd')
			
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
						
						
						#NOTE!IMPORTANT!: here you are only adding once, so you dont need to halve
						#so to correct you should mutiply t_tilde by 2
						# Add hopping between the two sites in this cluster: L_start <--> L_start+1
						#OK, temporary caveman way to add a single bond
						i1, i2 = L_start, L_start + 1
						dx=1
						u=0 #the sublattice

						# n_cells=L_cells-1 #This is from the periodic case - TODO: change this to automatical
						
						# str_arr=np.zeros(n_cells,dtype=float)
						# str_arr[i1]=2*t_tilde


						shape, shift = self.lat.coupling_shape((dx,))  # authoritative length & shift
						str_arr = np.zeros(shape, dtype=float)
						#periodic if bc[0] == False
						periodic = (self.lat.bc[0] == False)
						idx = (i1 - shift[0]) % shape[0] if periodic else (i1 - shift[0])
						if not periodic and not (0 <= idx < shape[0]):
							raise ValueError("bond would leave chain under OBC")

						str_arr[idx]=2*t_tilde
						
						
						self.add_coupling(str_arr, u, 'Cdd',u, 'Cd',dx, plus_hc=True) #spin-down
						self.add_coupling(str_arr, u, 'Cdu',u, 'Cu',dx, plus_hc=True) #spin-up
				
				
				
				
				#Add the onsite alpha U only to sites in this subcluster
				# for alpha in range(len(self.lat.unit_cell)):
				# 	for site_idx in range(L_start, L_end):
				# 		self.add_onsite(U, alpha, 'NuNd', site_idx)  # Hubbard n_up n_down term
			for alpha in range(len(self.lat.unit_cell)):
				self.add_onsite(U, alpha, 'NuNd')  # Hubbard n_up n_down term
					
			#Add V as next-nearest neighbor coupling (site 0<->2, site 1<->3)
			# if abs(V) > 0:
			# 	# Use dx=2 for next-nearest neighbor with periodic BC
			# 	for alpha in range(len(self.lat.unit_cell)):
			# 		#NOTE! I think it's V/4 here because I added it as a NNN hopping term
			# 		#with hermitian conjugates in each so I double count.
			# 		self.add_coupling(V, alpha, 'Cdd', alpha, 'Cd', 2, plus_hc=True)
			# 		self.add_coupling(V, alpha, 'Cdu', alpha, 'Cu', 2, plus_hc=True)

			if abs(V)>0:
				# NNN hopping V within a 1D Chain (1 site / unit cell)
				# u = 0
				# dx = 2

				# shape, shift = self.lat.coupling_shape((dx,))  # authoritative length & shift
				# mask = np.zeros(shape, dtype=float)

				# periodic = (self.lat.bc[0] == False)  # TeNPy: False == periodic, True == open
				# left_sites = [0, 1]  # bonds (0->2) and (1->3); replace with [L_start, L_start+1] per cluster

				# for i_left in left_sites:
				# 	idx = (i_left - shift[0]) % shape[0] if periodic else (i_left - shift[0])
				# 	if periodic or (0 <= idx < shape[0]):   # under OBC, skip bonds that would leave the chain
				# 		mask[idx] = V

				# # spin ↓ and ↑; JW strings are handled automatically
				# self.add_coupling(mask, u, 'Cdd', u, 'Cd', dx, plus_hc=True)
				# self.add_coupling(mask, u, 'Cdu', u, 'Cu', dx, plus_hc=True)


				# self.add_coupling_term(V, 0, 2, 'Cdd', 'Cd', plus_hc=True)
				# self.add_coupling_term(V, 0, 2, 'Cdu', 'Cu', plus_hc=True)
				# self.add_coupling_term(V, 1, 3, 'Cdu', 'Cu', plus_hc=True)
				# self.add_coupling_term(V, 1, 3, 'Cdd', 'Cd', plus_hc=True)

				self.add_coupling(V,0,'Cdd',0,'Cd',2,plus_hc=True)
				self.add_coupling(V,0,'Cdu',0,'Cu',2,plus_hc=True)
				
		else:
			raise ValueError("No basis class provided")
		
		# print("lat.bc (False=periodic, True=open):", self.lat.bc)
		# print("dx=1 shape/shift:", self.lat.coupling_shape((1,)))
		# print("dx=2 shape/shift:", self.lat.coupling_shape((2,)))

		
		# def bonds(self, dx):
		# 	# dx must be a tuple for TeNPy (1D chain -> (dx,))
		# 	dx_t = (dx,)
		# 	shape, shift = self.lat.coupling_shape(dx_t)   # authoritative length & shift
		# 	strength = np.ones(shape, dtype=float)         # match coupling_shape exactly
		# 	i, j, _ = self.lat.possible_couplings(0, 0, dx_t, strength)
		# 	return list(zip(i, j))

		# print("NN bonds (dx=1):", bonds(self, 1))
		# print("NNN bonds (dx=2):", bonds(self, 2))

		# def active_bonds(self, dx, mask):
		# 	dx_t = (dx,)
		# 	i, j, vals = self.lat.possible_couplings(0, 0, dx_t, mask)
		# 	return [(int(a), int(b)) for a, b, v in zip(i, j, vals) if abs(v) > 1e-15]

		# # example: mask with a single NN bond at i_left
		# dx = 1
		# shape, shift = self.lat.coupling_shape((dx,))
		# mask = np.zeros(shape); mask[i_left - shift[0]] = t_tilde  # OBC indexing
		# print("NN bonds actually added:", active_bonds(self, dx, mask))
		#raise ValueError('debug')

class SpectrumSolver():
	"""
	Class that will solve for the spectrum
	save: either false or folder path
	"""
	def __init__(self,hamiltonian,cluster_object:LocalClusterBasis,ham_lib:str='tenpy',solver_method='dense_ED',states_retained:Union[int,'all']=4,save:Union[bool,str]=False):
		self.hamiltonian=hamiltonian
		self.states_retained=states_retained
		self.cluster_object=cluster_object
		self.save=save
		self.ham_lib=ham_lib
		self.solver_method=solver_method
	def solve_spectrum(self):
		if self.ham_lib=='tenpy':
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
				eigvecs.append(psi_mps)
				n_ups.append(n_up)
				n_downs.append(n_down)
			eigvals=np.stack(eigvals,axis=0)
			eigvecs=np.stack(eigvecs,axis=0)
			n_ups=np.stack(n_ups,axis=0)
			n_downs=np.stack(n_downs,axis=0)
			n_tot=n_ups+n_downs
			
			return eigvals,eigvecs,n_ups,n_downs,n_tot
			#TODO:add functionality to efficiently get smaller number of total states.

		elif self.ham_lib=='quspin':
			#Note: I need the basis as well as the hamitlonian for quspin
			#if I want to get the more general operator expectation values
			#i.e the spin and number operators.
			from scipy import sparse
			
			ham,basis=self.hamiltonian
			ed_ham=ham.toarray()

			# Import logging only when needed
			try:
				from aah_code.cluster_model.logging_config import info
			except ImportError:
				info = lambda x: None  # No-op if logging not available
			
			if self.solver_method=='dense_ED':
				eigvals,eigvecs=np.linalg.eigh(ed_ham)
				info(f"Dense ED: computed {len(eigvals)} eigenvalues")
			elif self.solver_method=='sparse_ED':
				eigvals,eigvecs=sparse_diagonalize(ham,k=self.states_retained,return_eigenvectors=True)
				info(f"Sparse ED: computed {len(eigvals)} eigenvalues")
			else:
				raise ValueError(f'Solver method {self.solver_method} not implemented yet')
			  # After getting eigenvalues and eigenvectors

			# Simplified approach: construct total number operators directly
			from quspin.operators import hamiltonian
			
			# Build total number operators for each site explicitly
			n_up_site_ops = []
			n_down_site_ops = []
			
			for site in range(basis.L):
				# Construct n_up and n_down operators for this site
				n_up_list = [[1.0, site]]   # coefficient, site index
				n_down_list = [[1.0, site]]
				
				static_up = [["n|", n_up_list]]    # spin-up number operator  
				static_down = [["|n", n_down_list]]  # spin-down number operator
				
				# Suppress quspin's successful check messages but keep error checking
				with redirect_stdout(io.StringIO()):
					n_up_op = hamiltonian(static_up, [], basis=basis, dtype=np.complex128)
					n_down_op = hamiltonian(static_down, [], basis=basis, dtype=np.complex128)
				
				n_up_site_ops.append(n_up_op)
				n_down_site_ops.append(n_down_op)

			n_ups=[]
			n_downs=[]
			energy_eigvals=[]
			energy_eigvecs=[]

			# For each eigenvector, calculate site-resolved expectation values
			for i, (E_i, psi_vec) in enumerate(zip(eigvals, eigvecs.T)):
				# Calculate site-resolved number expectations using hamiltonian expectation values
				n_up_sites = np.array([np.real(op.expt_value(psi_vec)) for op in n_up_site_ops])
				n_down_sites = np.array([np.real(op.expt_value(psi_vec)) for op in n_down_site_ops])
				
				n_ups.append(n_up_sites)
				n_downs.append(n_down_sites)
				energy_eigvals.append(E_i)
				energy_eigvecs.append(psi_vec)
			
			n_ups=np.stack(n_ups,axis=0)  # shape: (n_eigenstates, n_sites)
			n_downs=np.stack(n_downs,axis=0)  # shape: (n_eigenstates, n_sites)
			n_tot=n_ups+n_downs
			energy_eigvals=np.array(energy_eigvals)

			return energy_eigvals,energy_eigvecs,n_ups,n_downs,n_tot

		else:

			raise ValueError(f'Solver {self.ham_lib} not implemented yet')
	
		
#Maybe I'll leave this for later
#class SpectrumContainer():

class FullSpectrum():
	"""
	Class to get the full spectrum of the Hamiltonian
	None temperature is zero temperature
	"""
	def __init__(self,clustered_k_points:np.ndarray,state_params:StatesParams,physical_params:HamiltonianParams,temperature:Union[None,float]=None,ham_lib:str='quspin'):
		self.clustered_k_points=clustered_k_points
		self.state_params=state_params
		self.physical_params=physical_params
		self.temperature=temperature
		self.ham_lib=ham_lib

		

	def get_full_spectrum(self,return_ham:bool=False):
		k_points=[]
		energy_spectrum=[]
		number_spectrum=[]
		spin_spectrum=[]
		ham_objects=[]
		for cluster_k in self.clustered_k_points:		
			cluster_object=LocalClusterBasis(cluster_k,self.state_params)
			ham_dict={'basis_class':cluster_object,
												'V':self.physical_params.V,
												't':self.physical_params.hopping,
												'mu':self.physical_params.mu_0,
												'U':self.physical_params.U,
												}
			if self.ham_lib=='tenpy':
				hamiltonian_object=Hubbard1D(ham_dict)
			elif self.ham_lib=='quspin':
				
				ham_object,basis_object=QuSpinHamiltonian(ham_dict).create_pi_V_pi_int_ham()
				hamiltonian_object=(ham_object,basis_object)
			else:
				raise ValueError(f'Hamiltonian library {self.ham_lib} not implemented yet')
			
			spectrum_solver=SpectrumSolver(hamiltonian_object,cluster_object,self.ham_lib)
			eigvals,eigvecs,n_ups,n_downs,n_tot=spectrum_solver.solve_spectrum()
			
			k_points.append(cluster_k)
			energy_spectrum.append(eigvals)
			number_spectrum.append(n_tot)
			spin_spectrum.append(np.array([n_ups,n_downs]))
			if return_ham:
				ham_objects.append(hamiltonian_object)

			
		
		k_points=np.stack(k_points,axis=0)
		energy_spectrum=np.stack(energy_spectrum,axis=0)
		number_spectrum=np.stack(number_spectrum,axis=0)
		spin_spectrum=np.stack(spin_spectrum,axis=0)

		logger.debug(f'spin spectrum shape: {spin_spectrum.shape}')

		if return_ham:
			return ham_objects
		else:
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
				#logger.info('Temperature is None - returning zero temperature expectations')
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
				#logger.info(f'Temperature is {temperature} - returning temperature dependent expectations')
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
			#shift to match boundary of original
			new_k_clusters=new_k_clusters-np.pi
			
			logger.info(f'shape new_k_clusters:{new_k_clusters.shape}')
			return new_k_clusters,new_k_clusters_idxs
		else:
			logger.info('No new k-clusters found')
			return np.array([])

def get_spectra(cluster_ks, state_params, physical_params,return_ham:bool=False,ham_lib:str='tenpy'):
	#Lets try to make the hamiltonian

	k_points=[]
	energy_spectrum=[]
	number_spectrum=[]
	spin_spectrum=[]
	ham_objects=[]

	for cluster_k in cluster_ks:
		total_cluster_size=cluster_k.shape[0]*cluster_k.shape[1]
		test_basis_1=LocalClusterBasis(cluster_k[0],state_params)
		test_basis_2=LocalClusterBasis(cluster_k[1],state_params)
		#logger.info(f'total_cluster size: {total_cluster_size}')
		ham_dict={'basis_classes':[test_basis_1,test_basis_2],
						'L':total_cluster_size,
						'L_cluster':cluster_k.shape[0],
						'V':physical_params.V,
						't':physical_params.hopping,
						'U':physical_params.U,
						'mu':physical_params.mu_0,
						}
		if ham_lib=='tenpy':
			test_ham=QuickHubbard1D(ham_dict)
		elif ham_lib=='quspin':
			quspin_object=QuSpinHamiltonian(ham_dict)
			
			#note that test_ham here is a tuple of (test_ham,test_basis), just formatting it this way to be consistent with what to feed in from tenpy.

			test_ham=quspin_object.create_pi_V_pi_int_half_pi_ham()
		else:
			raise ValueError(f'Hamiltonian library {ham_lib} not implemented yet')
			
		
		
		solver=SpectrumSolver(test_ham,None,ham_lib)#basis object never explicitly used anyway
		eigvals,eigvecs,n_ups,n_downs,n_tot=solver.solve_spectrum()
		


		k_points.append(cluster_k)
		energy_spectrum.append(eigvals)
		number_spectrum.append(n_tot)
		spin_spectrum.append(np.array([n_ups,n_downs]))
		if return_ham:
			ham_objects.append(test_ham)

			
		
	k_points=np.stack(k_points,axis=0)
	energy_spectrum=np.stack(energy_spectrum,axis=0)
	number_spectrum=np.stack(number_spectrum,axis=0)
	spin_spectrum=np.stack(spin_spectrum,axis=0)

	#logger.info(f'spin spectrum shape: {spin_spectrum.shape}')
		
	if return_ham:
		return ham_objects
	else:
		return k_points,energy_spectrum,number_spectrum,spin_spectrum



def test_quick_mismatched(lattice_points,cluster_size,physical_params,ham_lib:str='tenpy'):
	#lattice_points=16
	#cluster_size=2
	#physical_params=HamiltonianParams(U=10,V=5,hopping=1,mu_0=10/2)
	state_params=StatesParams(spin_states=2)
	int_lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//4)
	print(type(int_lattice_object))
	test=MismatchedQuick(int_lattice_object,physical_params,lattice_points//2)
	cluster_ks,cluster_idxs=test.recluster()
	#print(f'cluster ks shape: {cluster_idxs.shape}')
	#print(f'cluster idxs: {cluster_idxs}')

	#k_points,energies,number_spectrum,spin_spectrum=get_spectra(cluster_ks)
	spectra_4tuple=get_spectra(cluster_ks, state_params, physical_params,ham_lib=ham_lib)
	
	print(f'k points shape: {spectra_4tuple[0].shape},\n energies shape: {spectra_4tuple[1].shape},\n number_spectrum shape: {spectra_4tuple[2].shape}, spin spectrum shape: {spectra_4tuple[3].shape}')

	full_spectrum_obj=FullSpectrum(None,state_params,physical_params,None)
	system_expectations,cluster_expectations=full_spectrum_obj.get_cluster_thermodynamic_expectations(spectra_4tuple,None)
	
	return system_expectations,cluster_expectations
	# print(f"system energy density: {system_expectations[0]/lattice_points}",
	#    	f"system energy density mu_subtracted: {(system_expectations[0]+system_expectations[1]*physical_params.mu_0)/lattice_points} "
	#    	f"system filling density: {system_expectations[1]/lattice_points}",
	# 	f"system spin density:{system_expectations[2]/lattice_points}")


def sparse_diagonalize(H, k=4, which='SA', return_eigenvectors=True):
    """
    Diagonalize a sparse Hamiltonian using sparse eigensolvers.
    
    Parameters
    ----------
    H : quspin hamiltonian or scipy.sparse matrix
        The Hamiltonian to diagonalize
    k : int
        Number of eigenvalues/eigenvectors to compute (default: 16)
    which : str
        Which eigenvalues to find: 'SA' (smallest algebraic, default), 
        'LA' (largest algebraic), 'SM' (smallest magnitude), etc.
    return_eigenvectors : bool
        Whether to return eigenvectors (default: True)
    
    Returns
    -------
    eigenvalues : np.ndarray
        The k lowest eigenvalues
    eigenvectors : np.ndarray (if return_eigenvectors=True)
        The corresponding eigenvectors
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix
    import time
    
    # Convert to sparse matrix if it's a quspin hamiltonian
    if hasattr(H, 'tocsr'):
        H_sparse = H.tocsr()
    elif hasattr(H, 'toarray'):
        # If it's already sparse-like but not csr
        H_sparse = csr_matrix(H.toarray())
    else:
        # Assume it's already a sparse matrix
        H_sparse = H
    
    # Ensure k is not larger than matrix dimension - 1
    n_dim = H_sparse.shape[0]
    k_actual = min(k, n_dim - 1)
    
    if k_actual < k:
        print(f"Warning: Requested k={k} but matrix dimension is {n_dim}. Using k={k_actual}")
    
    # Create a non-zero initial vector to avoid ARPACK error -9
    # Use a random vector with small perturbation to avoid exact zeros
    v0 = np.random.RandomState(42).randn(n_dim) + 0.1
    v0 = v0 / np.linalg.norm(v0)  # Normalize
    
    # Use sparse eigenvalue solver with explicit initial vector
    try:
        if return_eigenvectors:
            eigenvalues, eigenvectors = eigsh(H_sparse, k=k_actual, which=which, 
                                             v0=v0, return_eigenvectors=True,
                                             tol=1e-10, maxiter=10000)
            # Sort by eigenvalue
            idx = eigenvalues.argsort()
            return eigenvalues[idx], eigenvectors[:, idx]
        else:
            eigenvalues = eigsh(H_sparse, k=k_actual, which=which,
                              v0=v0, return_eigenvectors=False,
                              tol=1e-10, maxiter=10000)
            return np.sort(eigenvalues)
    except Exception as e:
        # If sparse solver fails, try with different parameters
        print(f"Warning: Sparse solver failed with error: {e}")
        print("Attempting with relaxed tolerance and different initial vector...")
        
        # Try with a different random seed and relaxed tolerance
        v0_alt = np.ones(n_dim) + 0.01 * np.random.RandomState(123).randn(n_dim)
        v0_alt = v0_alt / np.linalg.norm(v0_alt)
        
        if return_eigenvectors:
            eigenvalues, eigenvectors = eigsh(H_sparse, k=k_actual, which=which,
                                             v0=v0_alt, return_eigenvectors=True,
                                             tol=1e-8, maxiter=5000)
            idx = eigenvalues.argsort()
            return eigenvalues[idx], eigenvectors[:, idx]
        else:
            eigenvalues = eigsh(H_sparse, k=k_actual, which=which,
                              v0=v0_alt, return_eigenvectors=False,
                              tol=1e-8, maxiter=5000)
            return np.sort(eigenvalues)


def benchmark_sparse_vs_dense(H, k=16):
    """
    Compare performance of sparse vs dense eigensolvers.
    
    Parameters
    ----------
    H : quspin hamiltonian
        The Hamiltonian to benchmark
    k : int
        Number of eigenvalues for sparse solver
    
    Returns
    -------
    dict
        Dictionary with timing and eigenvalue results
    """
    import time
    
    results = {}
    
    # Dense diagonalization
    print("Running dense diagonalization...")
    start_time = time.time()
    H_dense = H.toarray()
    eigvals_dense, _ = np.linalg.eigh(H_dense)
    dense_time = time.time() - start_time
    results['dense_time'] = dense_time
    results['dense_eigvals'] = eigvals_dense[:k]  # First k eigenvalues
    
    print(f"Dense diagonalization took {dense_time:.3f} seconds")
    
    # Sparse diagonalization
    print(f"Running sparse diagonalization (k={k})...")
    start_time = time.time()
    eigvals_sparse, _ = sparse_diagonalize(H, k=k)
    sparse_time = time.time() - start_time
    results['sparse_time'] = sparse_time
    results['sparse_eigvals'] = eigvals_sparse
    
    print(f"Sparse diagonalization took {sparse_time:.3f} seconds")
    
    # Compare results
    speedup = dense_time / sparse_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Check accuracy
    max_diff = np.max(np.abs(results['dense_eigvals'] - results['sparse_eigvals']))
    print(f"Maximum eigenvalue difference: {max_diff:.2e}")
    
    results['speedup'] = speedup
    results['max_diff'] = max_diff
    
    return results
		
	
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
		for term_name, onsite_term in hamiltonian.onsite_terms.items():
			# Access the actual terms list
			if hasattr(onsite_term, 'onsite_terms'):
				for site_idx, site_terms in enumerate(onsite_term.onsite_terms):
					for op_name, strength in site_terms.items():
						description = f"Site {site_idx}"
						print(f"| Onsite    | {site_idx:7} | {op_name:11} | {strength:8.3f} | N/A    | {description:11} |")
	
	# Coupling terms - need to extract from CouplingTerms objects
	if hasattr(hamiltonian, 'coupling_terms') and hamiltonian.coupling_terms:
		for term_name, coupling_term in hamiltonian.coupling_terms.items():
			if hasattr(coupling_term, 'coupling_terms'):
				# The coupling_terms attribute contains nested dictionaries
				for i, site_terms in coupling_term.coupling_terms.items():
					for op_pair, target_sites in site_terms.items():
						op_name = f"{op_pair[0]} {op_pair[1]}"
						for j, operators in target_sites.items():
							dx = j - i
							for target_op, strength in operators.items():
								sites = f"({i},{j})"
								description = f"dx={dx}"
								print(f"| Coupling  | {sites:7} | {op_name:11} | {strength:8.3f} | {dx:6} | {description:11} |")
	
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
	physical_params = HamiltonianParams(U=3, V=2, hopping=1, mu_0=0)
	
	# Create two test clusters with different k-points
	cluster_k_1 = np.array([0.0, np.pi/2])  # k-points for cluster 1
	cluster_k_2 = np.array([np.pi, 3*np.pi/2])  # k-points for cluster 2
	
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
		energy_dmrg, filling_dmrg, psi = run_dmrg_method(U, mu_0, V, t, system_size, chi)
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
		#height=500
	)
	
	# Update axes
	fig.update_xaxes(title_text="U", row=1, col=1)
	fig.update_yaxes(title_text="Energy Density", row=1, col=1)
	fig.update_xaxes(title_text="U", row=1, col=2)
	fig.update_yaxes(title_text="Filling Density", row=1, col=2)
	
	return fig

def compare_methods_line_plots(U_values, V_values, t=1, precomputed_results=None, ham_lib='tenpy'):
	"""
	Create line plots showing all four methods for varying U at fixed V values.
	Each figure shows 2x3 subplots (energy top row, filling bottom row).
	If more than 3 V values, creates multiple figures.
	
	Args:
		U_values: Array of U values
		V_values: Array of V values
		t: Hopping parameter
		precomputed_results: Optional dict with pre-computed results to avoid re-solving
		ham_lib: Hamiltonian library backend ('tenpy' or 'quspin')
	"""
	from aah_code.main import run_cluster_method, run_dmrg_method, run_twosite
	import math
	
	# Group V values into chunks of 3
	n_v_per_fig = 3
	n_figures = math.ceil(len(V_values) / n_v_per_fig)
	figures = []
	
	# Use pre-computed results if provided, otherwise compute
	if precomputed_results is not None:
		print("Using pre-computed results for line plots")
		all_results = precomputed_results
	else:
		# Pre-compute all results (same as in heatmap function)
		print(f"Computing line plots for U values: {U_values}")
		print(f"V values: {V_values}")
		print(f"t = {t}")
		
		# Storage for all results
		all_results = {}
		
		system_size = 100
		lattice_points = 100
		cluster_size = 2
		chi = 32
		
		for i, V in enumerate(V_values):
			print(f"\nComputing for V = {V}")
			
			# Initialize storage for this V
			all_results[V] = {
				'energies_dmrg': [],
				'energies_twosite': [],
				'energies_cluster_2site': [],
				'energies_cluster_4site': [],
				'fillings_dmrg': [],
				'fillings_twosite': [],
				'fillings_cluster_2site': [],
				'fillings_cluster_4site': []
			}
			
			for j, U in enumerate(U_values):
				mu_0 = U / 2
				print(f"  U = {U}, μ₀ = {mu_0}")
				
				# Run all methods
				energy_dmrg, filling_dmrg, psi_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
				energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)
				
				energy_twosite = run_twosite(U, mu_0, V, t, system_size)
				filling_twosite = 1.0
				
				energy_cluster_2, filling_cluster_2 = run_cluster_method(U, mu_0, V, t, system_size, ham_lib=ham_lib)
				energy_cluster_2_subtracted = (energy_cluster_2 + mu_0 * filling_cluster_2) / system_size
				filling_cluster_2_normalized = filling_cluster_2 / system_size
				
				physical_params = HamiltonianParams(U, V, t, mu_0)
				system_expectations, cluster_expectations = test_quick_mismatched(lattice_points, cluster_size, physical_params, ham_lib=ham_lib)
				total_energy, total_filling, total_spin = system_expectations
				energy_cluster_4_subtracted = (total_energy + physical_params.mu_0 * total_filling) / lattice_points
				filling_cluster_4_normalized = total_filling / lattice_points
				
				# Store results
				all_results[V]['energies_dmrg'].append(energy_dmrg_subtracted)
				all_results[V]['energies_twosite'].append(energy_twosite)
				all_results[V]['energies_cluster_2site'].append(energy_cluster_2_subtracted)
				all_results[V]['energies_cluster_4site'].append(energy_cluster_4_subtracted)
				all_results[V]['fillings_dmrg'].append(filling_dmrg)
				all_results[V]['fillings_twosite'].append(filling_twosite)
				all_results[V]['fillings_cluster_2site'].append(filling_cluster_2_normalized)
				all_results[V]['fillings_cluster_4site'].append(filling_cluster_4_normalized)
	
	# Create figures
	for fig_idx in range(n_figures):
		start_idx = fig_idx * n_v_per_fig
		end_idx = min(start_idx + n_v_per_fig, len(V_values))
		current_V_values = V_values[start_idx:end_idx]
		n_cols = len(current_V_values)
		
		fig = make_subplots(
			rows=2, cols=n_cols,
			subplot_titles=[f'Energy vs U (V={V:.2f})' for V in current_V_values] + 
						   [f'Filling vs U (V={V:.1f})' for V in current_V_values],
			vertical_spacing=0.15
		)
		
		colors = {'DMRG': 'red', 'Two-site': 'green', '2-site Cluster': 'blue', '4-site Cluster': 'purple'}
		
		for col_idx, V in enumerate(current_V_values):
			col = col_idx + 1
			
			# Energy plots (top row)
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['energies_dmrg'],
					mode='lines+markers', name='DMRG',
					line=dict(color=colors['DMRG'], width=2),
					marker=dict(size=6),
					showlegend=(col_idx == 0)
				),
				row=1, col=col
			)
			
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['energies_twosite'],
					mode='lines+markers', name='Two-site',
					line=dict(color=colors['Two-site'], width=2),
					marker=dict(size=6),
					showlegend=(col_idx == 0)
				),
				row=1, col=col
			)
			
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['energies_cluster_2site'],
					mode='lines+markers', name='2-site Cluster',
					line=dict(color=colors['2-site Cluster'], width=2),
					marker=dict(size=6),
					showlegend=(col_idx == 0)
				),
				row=1, col=col
			)
			
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['energies_cluster_4site'],
					mode='lines+markers', name='4-site Cluster',
					line=dict(color=colors['4-site Cluster'], width=2),
					marker=dict(size=6),
					showlegend=(col_idx == 0)
				),
				row=1, col=col
			)
			
			# Filling plots (bottom row)
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['fillings_dmrg'],
					mode='lines+markers', name='DMRG',
					line=dict(color=colors['DMRG'], width=2),
					marker=dict(size=6),
					showlegend=False
				),
				row=2, col=col
			)
			
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['fillings_twosite'],
					mode='lines+markers', name='Two-site',
					line=dict(color=colors['Two-site'], width=2),
					marker=dict(size=6),
					showlegend=False
				),
				row=2, col=col
			)
			
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['fillings_cluster_2site'],
					mode='lines+markers', name='2-site Cluster',
					line=dict(color=colors['2-site Cluster'], width=2),
					marker=dict(size=6),
					showlegend=False
				),
				row=2, col=col
			)
			
			fig.add_trace(
				go.Scatter(
					x=U_values, y=all_results[V]['fillings_cluster_4site'],
					mode='lines+markers', name='4-site Cluster',
					line=dict(color=colors['4-site Cluster'], width=2),
					marker=dict(size=6),
					showlegend=False
				),
				row=2, col=col
			)
		
		# Update layout
		fig.update_layout(
			title=f'Method Comparison Line Plots ({ham_lib.upper()} backend) (Figure {fig_idx + 1}/{n_figures})',
			showlegend=True
		)
		
		# Update axes
		for col in range(1, n_cols + 1):
			fig.update_xaxes(title_text="U", row=1, col=col)
			fig.update_yaxes(title_text="Energy Density", row=1, col=col)
			fig.update_xaxes(title_text="U", row=2, col=col)
			fig.update_yaxes(title_text="Filling Density", range=[0, 2], row=2, col=col)
		
		figures.append(fig)
	
	return figures

def compare_methods_heatmap_mu_fixed(U_values, V_values, mu_fixed=None, t=1, show_line_plots=False):
	"""
	Create a heatmap comparing methods with energy relative differences to DMRG (top row) 
	and filling (bottom row). Columns are reordered as: 2-site cluster (col 1), 4-site (col 2), 2-site analytical (col 3)
	"""
	from aah_code.main import run_cluster_method, run_dmrg_method, run_twosite
	
		
	fig = make_subplots(
		rows=2, cols=3,
		subplot_titles=[
			'2-site Cluster vs DMRG (Energy Diff %)', 
			'4-site Cluster vs DMRG (Energy Diff %)', 
			'2-site Analytical vs DMRG (Energy Diff %)',
			'2-site Cluster Filling', 
			'4-site Cluster Filling', 
			'2-site Analytical Filling'
		],
		vertical_spacing=0.15
	)
	
	# Initialize result arrays
	n_U, n_V = len(U_values), len(V_values)
	
	# Energy relative differences (percentage)
	energy_diff_2site_cluster = np.zeros((n_V, n_U))
	energy_diff_4site_cluster = np.zeros((n_V, n_U))
	energy_diff_2site_analytical = np.zeros((n_V, n_U))
	
	# Fillings
	filling_2site_cluster = np.zeros((n_V, n_U))
	filling_4site_cluster = np.zeros((n_V, n_U))
	filling_2site_analytical = np.zeros((n_V, n_U))
	
	# Storage for line plots (if needed)
	line_plot_results = {}
	
	system_size = 100  # For DMRG and 2-site cluster method
	lattice_points = 100  # For 4-site cluster method
	cluster_size = 2
	chi = 32
	
	tqdm.write(f"Computing heatmap for U values: {U_values}")
	tqdm.write(f"V values: {V_values}")
	tqdm.write(f"t = {t}")
	
	for i, V in tqdm(enumerate(V_values)):
		# Initialize storage for this V (for line plots)
		if show_line_plots:
			line_plot_results[V] = {
				'energies_dmrg': [],
				'energies_twosite': [],
				'energies_cluster_2site': [],
				'energies_cluster_4site': [],
				'fillings_dmrg': [],
				'fillings_twosite': [],
				'fillings_cluster_2site': [],
				'fillings_cluster_4site': []
			}
		
		for j, U in tqdm(enumerate(U_values)):
			if mu_fixed is None:
				mu_0 = U / 2  # Half-filling condition
				logger.info(f"Using half-filling condition: μ₀ = U/2, U: {U}, μ₀: {mu_0}")
			else:
				mu_0 = mu_fixed
				logger.info(f"Using fixed chemical potential: μ₀ = {mu_fixed}, U: {U}")
			tqdm.write(f"\nComputing U = {U}, V = {V}, μ₀ = {mu_0}")
			
			# 1. DMRG method (reference)
			tqdm.write("Running DMRG...")
			energy_dmrg, filling_dmrg, psi_dmrg = run_dmrg_method(U, mu_0, V, t, system_size, chi)
			energy_dmrg_subtracted = energy_dmrg + (mu_0 * filling_dmrg)
			
			# 2. Two-site analytical
			tqdm.write("Running two-site analytical...")
			energy_twosite = run_twosite(U, mu_0, V, t, system_size)
			filling_twosite = 1.0  # Half-filling by construction
			
			# 3. Cluster method (2-site clusters)
			tqdm.write("Running 2-site cluster method...")
			energy_cluster_2, filling_cluster_2 = run_cluster_method(U, mu_0, V, t, system_size)
			energy_cluster_2_subtracted = (energy_cluster_2 + mu_0 * filling_cluster_2) / system_size
			filling_cluster_2_normalized = filling_cluster_2 / system_size
			
			# 4. Four-site cluster method
			tqdm.write("Running 4-site cluster method...")
			physical_params = HamiltonianParams(U, V, t, mu_0)
			system_expectations, _ = test_quick_mismatched(lattice_points, cluster_size, physical_params)
			total_energy, total_filling, _ = system_expectations
			energy_cluster_4_subtracted = (total_energy + physical_params.mu_0 * total_filling) / lattice_points
			filling_cluster_4_normalized = total_filling / lattice_points
			
			# Calculate relative energy differences as percentages
			energy_diff_2site_cluster[i, j] = 100 * (energy_cluster_2_subtracted - energy_dmrg_subtracted) / abs(energy_dmrg_subtracted)
			energy_diff_4site_cluster[i, j] = 100 * (energy_cluster_4_subtracted - energy_dmrg_subtracted) / abs(energy_dmrg_subtracted)
			energy_diff_2site_analytical[i, j] = 100 * (energy_twosite - energy_dmrg_subtracted) / abs(energy_dmrg_subtracted)
			
			# Store fillings
			filling_2site_cluster[i, j] = filling_cluster_2_normalized
			filling_4site_cluster[i, j] = filling_cluster_4_normalized
			filling_2site_analytical[i, j] = filling_twosite
			
			# Store results for line plots if needed
			if show_line_plots:
				line_plot_results[V]['energies_dmrg'].append(energy_dmrg_subtracted)
				line_plot_results[V]['energies_twosite'].append(energy_twosite)
				line_plot_results[V]['energies_cluster_2site'].append(energy_cluster_2_subtracted)
				line_plot_results[V]['energies_cluster_4site'].append(energy_cluster_4_subtracted)
				line_plot_results[V]['fillings_dmrg'].append(filling_dmrg)
				line_plot_results[V]['fillings_twosite'].append(filling_twosite)
				line_plot_results[V]['fillings_cluster_2site'].append(filling_cluster_2_normalized)
				line_plot_results[V]['fillings_cluster_4site'].append(filling_cluster_4_normalized)
			
			tqdm.write(f"DMRG:              E={energy_dmrg_subtracted:.3f}, n={filling_dmrg:.3f}")
			tqdm.write(f"Two-site:          E={energy_twosite:.3f}, n={filling_twosite:.3f}, diff={energy_diff_2site_analytical[i,j]:.1f}%")
			tqdm.write(f"Cluster (2-site):  E={energy_cluster_2_subtracted:.3f}, n={filling_cluster_2_normalized:.3f}, diff={energy_diff_2site_cluster[i,j]:.1f}%")
			tqdm.write(f"Cluster (4-site):  E={energy_cluster_4_subtracted:.3f}, n={filling_cluster_4_normalized:.3f}, diff={energy_diff_4site_cluster[i,j]:.1f}%")
	
	# Calculate symmetric range for energy differences
	all_energy_diffs = np.concatenate([
		energy_diff_2site_cluster.flatten(),
		energy_diff_4site_cluster.flatten(),
		energy_diff_2site_analytical.flatten()
	])
	max_abs_energy_diff = np.max(np.abs(all_energy_diffs))
	
	# Create heatmaps
	
	# Top row: Energy relative differences (no individual colorbars)
	fig.add_trace(
		go.Heatmap(
			z=energy_diff_2site_cluster,
			x=U_values,
			y=V_values,
			colorscale='RdBu',
			zmid=0,
			zmin=-max_abs_energy_diff,
			zmax=max_abs_energy_diff,
			showscale=False,
			text=[[f"{energy_diff_2site_cluster[i,j]:.1f}%" for j in range(len(U_values))] for i in range(len(V_values))],
			texttemplate="%{text}",
			textfont={"size": 10},
			hovertemplate='U=%{x}<br>V=%{y}<br>Energy Diff: %{z:.1f}%<extra></extra>'
		),
		row=1, col=1
	)
	
	fig.add_trace(
		go.Heatmap(
			z=energy_diff_4site_cluster,
			x=U_values,
			y=V_values,
			colorscale='RdBu',
			zmid=0,
			zmin=-max_abs_energy_diff,
			zmax=max_abs_energy_diff,
			showscale=False,
			text=[[f"{energy_diff_4site_cluster[i,j]:.1f}%" for j in range(len(U_values))] for i in range(len(V_values))],
			texttemplate="%{text}",
			textfont={"size": 10},
			hovertemplate='U=%{x}<br>V=%{y}<br>Energy Diff: %{z:.1f}%<extra></extra>'
		),
		row=1, col=2
	)
	
	fig.add_trace(
		go.Heatmap(
			z=energy_diff_2site_analytical,
			x=U_values,
			y=V_values,
			colorscale='RdBu',
			zmid=0,
			zmin=-max_abs_energy_diff,
			zmax=max_abs_energy_diff,
			colorbar=dict(title="Energy Diff (%)", x=1.02, y=0.8, len=0.4),
			text=[[f"{energy_diff_2site_analytical[i,j]:.1f}%" for j in range(len(U_values))] for i in range(len(V_values))],
			texttemplate="%{text}",
			textfont={"size": 10},
			hovertemplate='U=%{x}<br>V=%{y}<br>Energy Diff: %{z:.1f}%<extra></extra>'
		),
		row=1, col=3
	)
	
	# Bottom row: Fillings (no individual colorbars for first two)
	fig.add_trace(
		go.Heatmap(
			z=filling_2site_cluster,
			x=U_values,
			y=V_values,
			colorscale='Viridis',
			zmin=0,
			zmax=2,
			showscale=False,
			text=[[f"{filling_2site_cluster[i,j]:.2f}" for j in range(len(U_values))] for i in range(len(V_values))],
			texttemplate="%{text}",
			textfont={"size": 10},
			hovertemplate='U=%{x}<br>V=%{y}<br>Filling: %{z:.3f}<extra></extra>'
		),
		row=2, col=1
	)
	
	fig.add_trace(
		go.Heatmap(
			z=filling_4site_cluster,
			x=U_values,
			y=V_values,
			colorscale='Viridis',
			zmin=0,
			zmax=2,
			showscale=False,
			text=[[f"{filling_4site_cluster[i,j]:.2f}" for j in range(len(U_values))] for i in range(len(V_values))],
			texttemplate="%{text}",
			textfont={"size": 10},
			hovertemplate='U=%{x}<br>V=%{y}<br>Filling: %{z:.3f}<extra></extra>'
		),
		row=2, col=2
	)
	
	fig.add_trace(
		go.Heatmap(
			z=filling_2site_analytical,
			x=U_values,
			y=V_values,
			colorscale='Viridis',
			zmin=0,
			zmax=2,
			colorbar=dict(title="Filling", x=1.02, y=0.25, len=0.4),
			text=[[f"{filling_2site_analytical[i,j]:.2f}" for j in range(len(U_values))] for i in range(len(V_values))],
			texttemplate="%{text}",
			textfont={"size": 10},
			hovertemplate='U=%{x}<br>V=%{y}<br>Filling: %{z:.3f}<extra></extra>'
		),
		row=2, col=3
	)
	
	# Update layout
	fig.update_layout(
		title='Method Comparison Heatmap: Energy Differences vs DMRG (top) and Filling (bottom)',
		showlegend=False
	)
	
	# Update axes
	for col in range(1, 4):
		fig.update_xaxes(title_text="U", row=1, col=col)
		fig.update_yaxes(title_text="V", row=1, col=col)
		fig.update_xaxes(title_text="U", row=2, col=col)
		fig.update_yaxes(title_text="V", row=2, col=col)
	
	# Optionally generate line plots
	if show_line_plots:
		line_plot_figures = compare_methods_line_plots(U_values, V_values, t, precomputed_results=line_plot_results)
		return fig, line_plot_figures
	else:
		return fig




def evaluate_arbitrary_hamiltonian_expectation(psi, new_model_params):
	"""
	Evaluate <psi|H_new|psi> where H_new is a different Hamiltonian
	and psi is an MPS from a previous DMRG calculation.
	
	Parameters:
	-----------
	psi : MPS
		The MPS state (e.g., ground state from previous DMRG)
	new_model_params : dict
		Parameters for the new Hamiltonian to evaluate
		
	Returns:
	--------
	float
		Expectation value <psi|H_new|psi>
	"""
	# Create the new Hamiltonian as an MPO
	new_model = Hubbard1D(new_model_params)
	H_new_mpo = new_model.H_MPO
	
	# Calculate expectation value
	energy_expectation = H_new_mpo.expectation_value(psi)
	
	return energy_expectation

def evaluate_hamiltonian_components(psi, model_params):
	"""
	Evaluate individual components of the Hamiltonian separately.
	
	Parameters:
	-----------
	psi : MPS
		The MPS state
	model_params : dict
		Parameters for the Hamiltonian
		
	Returns:
	--------
	dict
		Dictionary with expectation values of different terms
	"""
	basis_class = model_params['basis_class']
	t = model_params.get('t', 0.0)
	U = model_params.get('U', 0.0)
	mu_0 = model_params.get('mu', 0.0)
	V = model_params.get('V', 0.0)
	
	results = {}
	
	# 1. Kinetic energy (t_tilde terms)
	if abs(t) > 0:
		kinetic_params = model_params.copy()
		kinetic_params.update({'U': 0, 'mu': 0, 'V': 0})  # Only kinetic terms
		kinetic_model = Hubbard1D(kinetic_params)
		results['kinetic'] = kinetic_model.H_MPO.expectation_value(psi)
	
	# 2. Interaction energy (U terms)
	if abs(U) > 0:
		interaction_params = model_params.copy()
		interaction_params.update({'t': 0, 'mu': 0, 'V': 0})  # Only interaction
		interaction_model = Hubbard1D(interaction_params)
		results['interaction'] = interaction_model.H_MPO.expectation_value(psi)
	
	# 3. Chemical potential energy
	if abs(mu_0) > 0:
		mu_params = model_params.copy()
		mu_params.update({'t': 0, 'U': 0, 'V': 0})  # Only chemical potential
		mu_model = Hubbard1D(mu_params)
		results['chemical_potential'] = mu_model.H_MPO.expectation_value(psi)
	
	# 4. Staggered potential (V terms)
	if abs(V) > 0:
		v_params = model_params.copy()
		v_params.update({'t': 0, 'U': 0, 'mu': 0})  # Only staggered potential
		v_model = Hubbard1D(v_params)
		results['staggered_potential'] = v_model.H_MPO.expectation_value(psi)
	
	# 5. Total energy
	full_model = Hubbard1D(model_params)
	results['total'] = full_model.H_MPO.expectation_value(psi)
	
	return results

def compare_hamiltonians(basis_class, original_params, new_params_list):
	"""
	Compare expectation values of different Hamiltonians with the same ground state.
	
	Parameters:
	-----------
	basis_class : LocalClusterBasis
		The basis class defining the cluster
	original_params : dict
		Parameters for the original Hamiltonian (to get ground state)
	new_params_list : list of dict
		List of parameter dictionaries for new Hamiltonians to evaluate
		
	Returns:
	--------
	dict
		Results for each new Hamiltonian
	"""
	# Get ground state from original Hamiltonian
	original_params['basis_class'] = basis_class
	psi_original = get_gnd(original_params)
	
	# Get original energy for reference
	original_model = Hubbard1D(original_params)
	E_original = original_model.H_MPO.expectation_value(psi_original)
	
	results = {'original_energy': E_original}
	
	# Evaluate expectation of new Hamiltonians
	for i, new_params in enumerate(new_params_list):
		new_params['basis_class'] = basis_class
		E_new = evaluate_arbitrary_hamiltonian_expectation(psi_original, new_params)
		results[f'hamiltonian_{i}'] = E_new
		
		print(f"Original ground state energy: {E_original:.6f}")
		print(f"Expectation with Hamiltonian {i}: {E_new:.6f}")
		print(f"Energy difference: {E_new - E_original:.6f}")
		print("-" * 50)
	
	return results

def evaluate_custom_operator(psi, basis_class, operator_terms):
	"""
	Evaluate expectation value of a completely custom operator.
	
	Note: This is a simplified version that works with basic operators.
	For more complex operators, you may need to build MPOs manually.
	
	Parameters:
	-----------
	psi : MPS
		The MPS state
	basis_class : LocalClusterBasis
		The basis class defining the cluster
	operator_terms : list of tuples
		Each tuple is (coefficient, [(op_name, site), (op_name, site), ...])
		For simple operators like: [(1.0, [('Nu', 0)]), (2.0, [('Nd', 1)])]
		
	Returns:
	--------
	float
		Expectation value of the custom operator (for simple operators only)
	"""
	total_expectation = 0.0
	
	for coefficient, ops_and_sites in operator_terms:
		if len(ops_and_sites) == 1:
			# Single-site operator
			op_name, site = ops_and_sites[0]
			expectation = psi.expectation_value(op_name, sites=[site])[0]
			total_expectation += coefficient * expectation
		elif len(ops_and_sites) == 2:
			# Two-site operator - use correlation function
			op1_name, site1 = ops_and_sites[0]
			op2_name, site2 = ops_and_sites[1]
			if site1 == site2:
				# Same site - use expectation_value_term
				expectation = psi.expectation_value_term([(op1_name, site1), (op2_name, site2)])
			else:
				# Different sites - use correlation_function
				corr_matrix = psi.correlation_function(op1_name, op2_name, sites1=[site1], sites2=[site2])
				expectation = corr_matrix[0, 0]
			total_expectation += coefficient * expectation
		else:
			# Multi-site operator - use expectation_value_term
			term = [(op_name, site) for op_name, site in ops_and_sites]
			expectation = psi.expectation_value_term(term)
			total_expectation += coefficient * expectation
	
	return total_expectation

# Example usage functions
def example_usage():
	"""
	Example of how to use the arbitrary operator evaluation functions.
	"""
	# Assume you have a basis_class and original parameters
	# basis_class = LocalClusterBasis(...)  # your basis
	# original_params = {'t': 1.0, 'U': 2.0, 'mu': 0.5, 'V': 0.0}
	
	# Example 1: Compare different U values
	# new_u_values = [{'t': 1.0, 'U': 0.0, 'mu': 0.5, 'V': 0.0},
	#                 {'t': 1.0, 'U': 4.0, 'mu': 0.5, 'V': 0.0}]
	# results = compare_hamiltonians(basis_class, original_params, new_u_values)
	
	# Example 2: Evaluate components separately
	# original_params['basis_class'] = basis_class
	# psi = get_gnd(original_params)
	# components = evaluate_hamiltonian_components(psi, original_params)
	# print("Energy components:", components)
	
	# Example 3: Custom operator (e.g., spin-spin correlation at distance 2)
	# custom_terms = [(1.0, [('Sz', 0), ('Sz', 2)]),  # S_z(0) * S_z(2)
	#                 (0.5, [('Nu', 1)])]              # 0.5 * n_up(1)
	# custom_expectation = evaluate_custom_operator(psi, basis_class, custom_terms)
	# print(f"Custom operator expectation: {custom_expectation}")
	
	pass

def get_dmrg_expectations(physical_params: HamiltonianParams):
	"""
	Calculate expectation values for various operators using DMRG.
	
	Parameters:
	-----------
	physical_params : HamiltonianParams
		Parameters for the Hamiltonian (U, V, hopping, mu_0)
		
	Returns:
	--------
	dict
		Dictionary with expectation values for particle numbers, spin, hopping energy,
		interaction energy, and V energy.
	"""
	
	# Run DMRG method to get ground state and expectations


	energy_dmrg,filling_dmrg, psi_dmrg= run_dmrg_method(
		physical_params.U, physical_params.mu_0, physical_params.V, physical_params.hopping,
		system_size=100, chi=32
	)
	#NOTE: i needs to match the period of V in the iDMRG calculation because operator are periodic in V period	
	spin_up_dmrg=np.array([psi_dmrg.expectation_value('Nu', i) for i in range(2)])
	spin_down_dmrg=np.array([psi_dmrg.expectation_value('Nd', i) for i in range(2)])
	particle_numbers_dmrg=spin_up_dmrg+spin_down_dmrg
	spin_dmrg=spin_up_dmrg-spin_down_dmrg
	interaction_energy_dmrg=physical_params.U*np.array([psi_dmrg.expectation_value('NuNd', i) for i in range(2)])
	v_energy_dmrg=physical_params.V*particle_numbers_dmrg*np.array([[1],[-1]])
	#Note that a correlation function is <O_1O_2>, which is the same form as the expectation of the hopping
	#so you can use that as a hack here.
	# Spin up: <Cd_up(i) * C_up(i+1)>
	hop_up_forward=psi_dmrg.correlation_function('Cdu','Cu',sites1=[0],sites2=[1])[0,0]
	hop_up_back=psi_dmrg.correlation_function('Cdu','Cu',sites1=[1],sites2=[0])[0,0]
	#Spin down: <Cd_down(i) * C_down(i+1)>
	hop_down_forward=psi_dmrg.correlation_function('Cdd','Cd',sites1=[0],sites2=[1])[0,0]
	hop_down_back=psi_dmrg.correlation_function('Cdd','Cd',sites1=[1],sites2=[0])[0,0]
	#Total hopping energy
	#Just to match the shape of the other objects...
	hopping_energy_dmrg=2*physical_params.hopping*np.array([[hop_up_forward+hop_up_back],[hop_down_forward+hop_down_back]])



	

	dmrg_expectations = {
		'particle_numbers':particle_numbers_dmrg,
		'spin':spin_dmrg,
		'hopping_energy': hopping_energy_dmrg, 
		'interaction_energy': interaction_energy_dmrg,
		'v_energy': v_energy_dmrg
	}

	return dmrg_expectations

#TODO: move into the mismatched class
def get_thermodynamics_expectations_mismatched(energy_eigvals,energy_eigvecs,sites,operator_str:str,temperature:Union[float,None]=None):
	if temperature is None:
		#You already converted to mps you magnificent beast
		psi_mps=energy_eigvecs[0]
		#n_up=psi_mps.expectation_value('Nu')
		#n_down=psi_mps.expectation_value('Nd')
		operator_expectation=psi_mps.expectation_value(operator_str)
		
		return operator_expectation		
		
	else:
		#return thermodynamic averages
		raise ValueError("Thermodynamic averages not implemented for mismatched clusters yet.")

		pass
	
	return None


def get_V_exp(psi_mps, test_basis_1, test_basis_2, physical_params, total_cluster_size, cluster_k):
	"""
	Get site-resolved V expectation values from a V-only Hamiltonian.
	
	Parameters:
	-----------
	psi_mps : MPS
		The MPS state (ground state from spectrum solver)
	test_basis_1 : LocalClusterBasis
		First cluster basis
	test_basis_2 : LocalClusterBasis  
		Second cluster basis
	physical_params : HamiltonianParams
		Physical parameters containing V value
	total_cluster_size : int
		Total size of the cluster system
	cluster_k : np.ndarray
		The k-point cluster array
		
	Returns:
	--------
	dict
		Dictionary containing:
		- 'site_resolved': np.ndarray of V energy per site/bond
		- 'total': float, total V energy (for verification)
		- 'method': str, describing the calculation method
	"""
	
	# Create V-only Hamiltonian
	v_dict = {
		'basis_classes': [test_basis_1, test_basis_2],
		'L': total_cluster_size,
		'L_cluster': cluster_k.shape[0],
		'V': physical_params.V,
		't': 0,
		'U': 0,
		'mu': 0,
	}
	v_model = QuickHubbard1D(v_dict)
	
	# Method 1: Get site-resolved V energy from bond contributions
	# Since V is implemented as NNN coupling (dx=2) with coefficient V/4
	L = len(psi_mps.sites)
	v_bond_resolved = []
	
	for i in range(L):
		j = (i + 2) % L  # Next-nearest neighbor with periodic BC
		
		# V coupling terms for this bond
		hop_up = psi_mps.correlation_function('Cdu', 'Cu', sites1=[i], sites2=[j])[0,0]
		hop_down = psi_mps.correlation_function('Cdd', 'Cd', sites1=[i], sites2=[j])[0,0]
		hop_up_hc = psi_mps.correlation_function('Cu', 'Cdu', sites1=[i], sites2=[j])[0,0]
		hop_down_hc = psi_mps.correlation_function('Cd', 'Cdd', sites1=[i], sites2=[j])[0,0]
		
		# V energy contribution from this bond (using your V/4 coefficient)
		bond_v_energy = (physical_params.V/4) * (hop_up + hop_down + hop_up_hc + hop_down_hc)
		v_bond_resolved.append(bond_v_energy)
	
	v_bond_resolved = np.array(v_bond_resolved)
	
	# Method 2: Alternative - conceptual staggered potential form
	# Get particle numbers for comparison
	n_up_sites = psi_mps.expectation_value('Nu')
	n_down_sites = psi_mps.expectation_value('Nd')
	n_total_sites = n_up_sites + n_down_sites
	
	# V as staggered potential: +V/2 on even sites, -V/2 on odd sites
	v_staggered_pattern = np.array([+physical_params.V/2 if (i % 2 == 0) else -physical_params.V/2 
								   for i in range(L)])
	v_staggered_resolved = v_staggered_pattern * n_total_sites
	
	# Get total V energy for verification
	v_total_mpo = v_model.H_MPO.expectation_value(psi_mps)
	
	# Return results
	results = {
		'bond_resolved': v_bond_resolved,
		'staggered_resolved': v_staggered_resolved,  
		'total_mpo': v_total_mpo,
		'total_bond_sum': np.sum(v_bond_resolved),
		'total_staggered_sum': np.sum(v_staggered_resolved),
		'particle_numbers': n_total_sites,
		'method': 'NNN coupling (bond) + staggered potential (conceptual)'
	}
	
	return results

def get_mismatched_cluster_expectations(physical_params: HamiltonianParams, lattice_points: int):
	"""
	The equivalent of get_dmrg_expectations but for the mismatched cluster method.
	In this case, we can probably try to return over all lattice points so output is
	[L//4,2,2,1] (L//4 lattice points, 2 sub-cluster indices, 2 within cluster indices, 1 dimensional k point).
	and then you can return the averaged values too. 
	"""
	number_expectations=[]
	spin_expectations=[]
	hopping_expectations=[]
	interaction_expectations=[]
	v_expectations=[]

	
	state_params=StatesParams(spin_states=2)
	cluster_size=2
	int_lattice_object=ClusterExperiment(cluster_size,lattice_points,lattice_points//4)
	mismatched_object=MismatchedQuick(int_lattice_object,physical_params,lattice_points//2)

	cluster_ks,cluster_idxs=mismatched_object.recluster()
	
	
	
	
	n_tots=[]
	spins=[]
	U_terms=[]
	v_terms=[]
	t_terms=[]
	energies=[]
	
	for cluster_k in cluster_ks:
		total_cluster_size=cluster_k.shape[0]*cluster_k.shape[1]
		test_basis_1=LocalClusterBasis(cluster_k[0],state_params)
		test_basis_2=LocalClusterBasis(cluster_k[1],state_params)
		basic_dict={'basis_classes':[test_basis_1,test_basis_2],
					'L':total_cluster_size,
					'L_cluster':cluster_k.shape[0],
					'V':physical_params.V,
					't':physical_params.hopping,
					'U':physical_params.U,
					'mu':physical_params.mu_0,
					}
		test_ham=QuickHubbard1D(basic_dict)
		
		
		solver=SpectrumSolver(test_ham,None)#basis object never explicitly used anyway
		eigvals,eigvecs,_,_,_=solver.solve_spectrum()
		sites=test_ham.lat.mps_sites()
		#get_expectation_values:
		n_up = get_thermodynamics_expectations_mismatched(eigvals,eigvecs,sites,'Nu',temperature=None)
		n_down = get_thermodynamics_expectations_mismatched(eigvals,eigvecs,sites,'Nd',temperature=None)
		Uterm= physical_params.U*get_thermodynamics_expectations_mismatched(eigvals,eigvecs,sites,'NuNd',temperature=None)

		v_dict={'basis_classes':[test_basis_1,test_basis_2],
					'L':total_cluster_size,
					'L_cluster':cluster_k.shape[0],
					'V':physical_params.V,
					't':0,
					'U':0,
					'mu':0,
					}
		v_model=QuickHubbard1D(v_dict)
		v_exp=v_model.H_MPO.expectation_value(eigvecs[0])
		v_exp=np.ones((total_cluster_size))*(v_exp/total_cluster_size)
		#logger.warning("V was implemented without site resolution, need to do this properly later, no T>0 implementation")
		#hopping
		t_dict={'basis_classes':[test_basis_1,test_basis_2],
			'L':total_cluster_size,
			'L_cluster':cluster_k.shape[0],
			'V':0,
			't':physical_params.hopping,
			'U':0,
			'mu':0,
			}
		t_model=QuickHubbard1D(t_dict)
		t_exp=t_model.H_MPO.expectation_value(eigvecs[0])
		t_exp=np.ones((total_cluster_size))*(t_exp/total_cluster_size)
		


		n_tots.append(n_up+n_down)
		spins.append(n_up-n_down)
		v_terms.append(v_exp)
		t_terms.append(t_exp)
		U_terms.append(Uterm)
		energies.append(np.ones((total_cluster_size))*(eigvals[0]/total_cluster_size))

	n_tots=np.stack(n_tots,axis=0)
	spins=np.stack(spins,axis=0)
	U_terms=np.stack(U_terms,axis=0)
	v_terms=np.stack(v_terms,axis=0)
	t_terms=np.stack(t_terms,axis=0)
	energies=np.stack(energies,axis=0)

	mismatched_expectations = {
		'particle_numbers':n_tots,
		'spin':spins,
		'hopping_energy': t_terms,#energies-U_terms-v_terms+n_tots*physical_params.mu_0, 
		'interaction_energy': U_terms,
		'v_energy': v_terms,
		'total_energy':energies
	}

	return mismatched_expectations

		



def get_expectations(physical_params:HamiltonianParams):
	"""
	Want to return the (optionally site-resolved) expectation values
	for some operators. 
	Do this for all three methods: iDMRG, two-site cluster, and four-site cluster.
	The things I want to compare are:

	1. Site number bias
	2. Site spin expectation
	3. Site coupling expectation
	4. Site interaction energy expectation
	5. Site V expectation. 

	"""
	
	mismatched_cluster_dict=get_mismatched_cluster_expectations(physical_params, lattice_points=100)
	dmrg_dict=get_dmrg_expectations(physical_params)

	return dmrg_dict, mismatched_cluster_dict


def expectations_plot(physical_params: HamiltonianParams):
	
	dmrg_dict,mismatched_cluster_dict=get_expectations(physical_params)


	fig=make_subplots(rows=2,cols=5)

	observables=['particle_numbers','spin','interaction_energy','v_energy','hopping_energy']
	
	print(f'dmrg shape: {dmrg_dict["particle_numbers"].shape}')
	print(f'mismatched shape: {mismatched_cluster_dict["particle_numbers"].shape}')
	

	for i,observable in enumerate(observables):
		fig.add_trace(go.Scatter(
			x=np.arange(dmrg_dict[observable].size),
			y=dmrg_dict[observable].flatten(),
			mode='lines+markers',
			name=f'DMRG {observable}',
			line=dict(color='blue', width=2),
			marker=dict(size=8)
		), row=1, col=i+1)

		fig.add_trace(go.Scatter(
			x=np.arange(mismatched_cluster_dict[observable].size),
			y=mismatched_cluster_dict[observable].flatten(),
			mode='lines+markers',
			name=f'Mismatched {observable}',
			line=dict(color='red', width=2),
			marker=dict(size=8)
		), row=2, col=i+1)
	fig.update_layout(
		title=f'Expectation Values Comparison (U={physical_params.U}, V={physical_params.V}, t={physical_params.hopping}, mu_0={physical_params.mu_0})',
		xaxis_title='Site Index',
		yaxis_title='Expectation Value',
	)


	fig2=make_subplots(rows=1,cols=1)



	return fig,fig2

def expectations_plot_combined(physical_params: HamiltonianParams):
	
	dmrg_dict, mismatched_cluster_dict = get_expectations(physical_params)

	# Create subplots: original plots + stacked bar chart
	fig = make_subplots(
		rows=3, cols=5,
		subplot_titles=['DMRG Results', '', '', '', '',
					   'Mismatched Cluster Results', '', '', '', '',
					   'Energy Components Comparison', '', '', '', ''],
		specs=[[{}, {}, {}, {}, {}],
			   [{}, {}, {}, {}, {}],
			   [{"colspan": 5}, None, None, None, None]]
	)

	observables = ['particle_numbers', 'spin', 'interaction_energy', 'v_energy', 'hopping_energy']

	
	
	print(f'dmrg shape: {dmrg_dict["particle_numbers"].shape}')
	print(f'mismatched shape: {mismatched_cluster_dict["particle_numbers"].shape}')
	
	# Original scatter plots
	for i, observable in enumerate(observables):
		fig.add_trace(go.Scatter(
			x=np.arange(dmrg_dict[observable].size),
			y=dmrg_dict[observable].flatten(),
			mode='lines+markers',
			name=f'DMRG {observable}',
			line=dict(color='blue', width=2),
			marker=dict(size=8),
			showlegend=False
		), row=1, col=i+1)

		fig.add_trace(go.Scatter(
			x=np.arange(mismatched_cluster_dict[observable].size),
			y=mismatched_cluster_dict[observable].flatten(),
			mode='lines+markers',
			name=f'Mismatched {observable}',
			line=dict(color='red', width=2),
			marker=dict(size=8),
			showlegend=False
		), row=2, col=i+1)

	# Stacked bar chart for energy components
	energy_components = ['hopping_energy', 'v_energy', 'interaction_energy']
	
	# Calculate total energies for each method
	dmrg_totals = {}
	mismatched_totals = {}
	
	for component in energy_components:
		dmrg_totals[component] = np.sum(dmrg_dict[component])
		mismatched_totals[component] = np.sum(mismatched_cluster_dict[component])
	
	print(f"mismatched components: {np.array(list(mismatched_totals.values())).sum()},total energy: {mismatched_totals['total_energy'].sum()}")
	exit()

	methods = ['DMRG', 'Mismatched Cluster']
	colors = {'hopping_energy': 'lightblue', 'v_energy': 'lightgreen', 'interaction_energy': 'lightcoral'}
	
	# Add stacked bars
	for component in energy_components:
		values = [dmrg_totals[component]/2, mismatched_totals[component]/100]
		
		fig.add_trace(go.Bar(
			x=methods,
			y=values,
			name=component.replace('_', ' ').title(),
			marker_color=colors[component],
			showlegend=True
		), row=3, col=1)
	
	# Update layout
	fig.update_layout(
		title='Expectation Values Comparison: DMRG vs Mismatched Cluster',
		barmode='stack',
		#height=900  # Increase height for 3 rows
	)
	
	# Update axes for energy comparison
	fig.update_xaxes(title_text="Method", row=3, col=1)
	fig.update_yaxes(title_text="Total Energy", row=3, col=1)

	return fig
		



if __name__ == "__main__":
	print('main')


	# Test Hamiltonian inspection
	# test_hamiltonian_inspection()
	# exit()
	
	
	
	#exit('Inspected Hamiltonian - check that coupling is correct')
	
	# lattice_points=16
	# cluster_size=2
	# test_quick_mismatched(lattice_points,cluster_size,physical_params=HamiltonianParams(U=1.0,V=1.0,hopping=1.0,mu_0=0.5))
	# exit('Tested quick mismatched')

	#compare the different methods
	U_values=np.linspace(1e-8,1,1)
	V_values=np.linspace(0,1,4)

	fig,line_figs=compare_methods_heatmap_mu_fixed(U_values,V_values,mu_fixed=None,t=1,show_line_plots=True)
	
	
	for line_fig in line_figs:
		line_fig.show()
	fig.show()
	# exit()
	#fig=quick_spectrum_test_vary_U(U_values,V=0)
	#fig.show()

	#Test operator expectations
	

	
	
	
	#fig1,fig2=expectations_plot(HamiltonianParams(U=1.0,V=1.0,hopping=1.0,mu_0=0.5))
	
	#fig1.show()

	#fig=expectations_plot_combined(HamiltonianParams(U=1.0,V=1.0,hopping=1.0,mu_0=0.5))
	#fig.show()
	

	exit('testing expectations')
	fig=make_subplots(rows=1,cols=1)
	fig.add_trace(
		go.Scatter(
			x=np.arange(len(system_expectations)),
			y=system_expectations.squeeze(-1),
			mode='lines+markers',
			name='Expectation Values',
			line=dict(color='blue', width=2),
			marker=dict(size=8)
		)
	)
	fig.update_layout(
		title='Expectation Values of number operators',
		xaxis_title='Site',
		yaxis_title='Expectation Value',
	)
	fig.show()

	exit("Testing operator expectations")
	compare_all_methods_vs_U(U_values,V=5,t=1).show()
	#fig.write_html("quick_spectrum_test.html")
	#print("Quick spectrum test saved as 'quick_spectrum_test.html'")

	# Test the new heatmap function
	print("\nTesting new heatmap function...")
	U_test = np.linspace(2, 6, 3)  # Small test range
	V_test = np.linspace(0, 2, 3)  # Small test range
	
	try:
		fig_heatmap = compare_methods_heatmap_mu_fixed(U_test, V_test, t=1)
		fig_heatmap.write_html("method_comparison_heatmap.html")
		print("Heatmap function test successful! Saved as 'method_comparison_heatmap.html'")
	except Exception as e:
		print(f"Heatmap function test failed: {e}")
	
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



















