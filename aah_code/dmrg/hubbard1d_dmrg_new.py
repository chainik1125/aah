import numpy as np
import tenpy as tp
from matplotlib import pyplot as plt
from tenpy.models import CouplingMPOModel, NearestNeighborModel
from tenpy.models import lattice
from collections import defaultdict
np.set_printoptions(precision=5, suppress=True, linewidth=100)
# SET TO TRUE IF YOU WANT LOGS OF THE DMRG PROCESS TO PRINT
logging = False
if logging:
	tp.tools.misc.setup_logging(to_stdout="INFO") # logging during DMRG calculation
	

class Hubbard1D(CouplingMPOModel, NearestNeighborModel):
	"""
	Input is a dictionary called model_params that includes:
		'L' (length), 'bc' ('open' or 'periodic'), 'bc_MPS' ('finite' or 'infinite'),
		't' (hopping strength), 'U' (Hubbard interaction strength), 'filling' (1 is half-filling)
	"""

	# Initialize spin-1/2 fermion d.o.f. on each site
	def init_sites(self, model_params):
		# Remove both particle number and spin conservation to allow DMRG to explore all sectors
		site = tp.networks.site.SpinHalfFermionSite(cons_N=None, cons_Sz=None)
		return site

	# Set 1D lattice
	def init_lattice(self, model_params):
		L = model_params['L'] # size
		bc = model_params.get('bc', 'open')  # always use 'open'
		bc_MPS = model_params.get('bc_MPS', 'finite')  # 'infinite' does iDMRG (still use open in 'bc')
		lat = lattice.Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=self.init_sites(model_params))
		return lat

	def init_terms(self, model_params):
		# default is U=1, t=1
		U = model_params.get('U', 1.0)
		t = model_params.get('t', 1.0)
		mu = model_params.get('mu', 0.0)
		V=model_params.get('V', 5.0)
		# nearest neighbor hopping -t
		for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
			self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)  # Cdagger_down C_down + h.c.
			self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)  # Cdagger_up C_up + h.c.
		
		# Onsite terms
		for v in range(len(self.lat.unit_cell)):
			self.add_onsite(U, v, 'NuNd')  # Hubbard n_up n_down term
			self.add_onsite(-mu, v, 'Nu')  # chemical potential n_up
			self.add_onsite(-mu, v, 'Nd')  # chemical potential n_down

			if abs(V) > 0:        # i = 0 … L-1
				sign =  +V if (v % 2 == 0) else -V   # even sites +V, odd sites –V
				self.add_onsite(sign, v, 'Nu')       # n↑ part
				self.add_onsite(sign, v, 'Nd')  


# chi is bond dimension of MPS (chi ~ log (S_ent))
# Should check increasing chi to see convergence of E_gnd
def get_gnd(L, chi, U=1, t=1, mu=0):
	# initialize Hamiltonian
	model = Hubbard1D({'L': L, 'U':U, 't':t, 'bc':'open', 'bc_MPS':'finite', 'mu':mu})
	
	# Choose initial state based on chemical potential
	if mu > U:
		product_state = L * ['full']
	elif mu < 0:
		product_state = L * ['empty']
	else:
		product_state = (L//2) * ['up', 'down']
	
	psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state)

	dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-8},
		'max_E_err': 1e-8, 'max_S_err': 1e-6, 'min_sweeps': 5, 'max_sweeps': 50, 'max_trunc_err': None}

	engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
	E, psi = engine.run()
	
	# Calculate average filling (n_up + n_down)
	N_up = np.mean([psi.expectation_value('Nu', i) for i in range(L)])
	N_down = np.mean([psi.expectation_value('Nd', i) for i in range(L)])
	filling = N_up + N_down
	
	return E, psi, filling

# uses iDMRG to get gnd state energy density in thermodynamic limit (L -> \infty)
# may not be best choice if system is gapless (test it a bit to check)
def get_gnd_infinite(chi, U=1, t=1, mu=0):
	# initialize Hamiltonian
	model = Hubbard1D({'L': 2, 'U':U, 't':t, 'bc':'periodic', 'bc_MPS':'infinite', 'mu':mu})
	
	# Choose initial state based on chemical potential
	if mu > U:
		# Start from fully filled state for positive mu
		product_state = ['full', 'full']
	elif mu < 0:
		# Start from empty state for negative mu
		product_state = ['empty', 'empty']
	else:
		# Start from Néel state at mu=0
		product_state = ['up', 'down']
	
	psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state, 'infinite')

	# parameters you can ignore for now (affect precision of DMRG)
	dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-8},
		'max_E_err': 1e-8, 'max_S_err': 1e-6, 'min_sweeps': 5, 'max_sweeps': 50, 'max_trunc_err': None}

	# initialize two site DMRG engine
	engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
	E, psi = engine.run()
	
	# For infinite system, measure on the unit cell (2 sites)
	N_up = np.mean([psi.expectation_value('Nu', i) for i in range(2)])
	N_down = np.mean([psi.expectation_value('Nd', i) for i in range(2)])
	filling = N_up + N_down
	
	return E, psi, filling

def main():
	# when U is large, system is more localized so easier for DMRG to be accurate
	#U = 5
	bond_dimensions = [8, 16, 32]
	system_sizes = [8, 12, 16, 100]

	# generate finite size data for ground state energy density 
	# data_finite = np.zeros( (len(system_sizes), len(bond_dimensions)) )
	# for i, L in enumerate(system_sizes):
	# 	for j, chi in enumerate(bond_dimensions):
	# 		print(f'Size: {L}, Bond Dimension: {chi}')
	# 		E, psi = get_gnd(L, chi, U)
	# 		print("Ground state energy density: ", E/L, "\n")
	# 		data_finite[i, j] = E/L
	
	# generate infinite size data using iDMRG 
	data_infinite = defaultdict(lambda: defaultdict(list))
	filling_infinite = defaultdict(lambda: defaultdict(list))
	U_values=np.linspace(10,20,5)
	for U in U_values:
		mu_values=np.linspace(-U,2*U,5)#range(-U,2*U,2)
		for mu in mu_values:
			for chi in bond_dimensions:
				E_density, psi, filling = get_gnd_infinite(chi, U, mu=mu)
				print(f'Size: Infinite, Bond Dimension: {chi}, mu: {mu}')
				print("Ground state energy density: ", E_density)
				print("Average filling: ", filling, "\n")
				data_infinite[U][mu].append(E_density)
				filling_infinite[U][mu].append(filling)
	# fig, ax = plt.subplots()
	# for j, row in enumerate(data_finite):
	# 	ax.plot(bond_dimensions, row, 'x-', label=f'L = {system_sizes[j]}')
	# ax.plot(bond_dimensions, data_infinite, 'o-', label=fr'L = $\infty$')
	# ax.set_xlabel('Bond Dimension', fontsize=16)
	# ax.set_ylabel('Ground State Energy Density', fontsize=16)
	# ax.legend(fontsize=14)
	# ax.set_title(f'DMRG comparison for 1D Hubbard Model with U = {U}', fontsize=16)
	# plt.show()
	return data_infinite, filling_infinite


if __name__ == '__main__':
	data_infinite, filling_infinite = main()
	print(data_infinite)
	print(filling_infinite)



