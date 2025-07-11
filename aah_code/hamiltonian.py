"""
This module is meant to generate the Hamiltonian on a given cluster.
For performance, you should generate a template and fill in the Hamiltonian
from a template of values.
There's a slightly tricky issue of how the Hamiltonian interacts with the clusters
since you're really defining an interaction cluster rather than a full Hamiltonian cluster.
"""

from aah_code.basis import LocalClusterBasis
from aah_code.global_params import StatesParams,HamiltonianParams
from aah_code.utils import mu_tilde_coefficient,cosine_dispersion
from tenpy.models import CouplingMPOModel,NearestNeighborModel,lattice
import tenpy as tp
import numpy as np
import matplotlib.pyplot as plt



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

	print(vars(test_ham.all_coupling_terms()))
	


	
	

	

	

	
		  



