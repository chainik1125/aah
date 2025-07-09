"""
This file will calculate the real space DMRG to compare with the other values
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
from tqdm import tqdm

class RealSpaceHubbard1D(CouplingMPOModel, NearestNeighborModel):
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
		V=model_params.get('V', 0.0)
		# nearest neighbor hopping -t
		for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
			self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)  # Cdagger_down C_down + h.c.
			self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)  # Cdagger_up C_up + h.c.
		
		# Onsite terms
		for v in range(len(self.lat.unit_cell)):
			self.add_onsite(U, v, 'NuNd')  # Hubbard n_up n_down term
			self.add_onsite(-mu, v, 'Nu')  # chemical potential n_up
			self.add_onsite(-mu, v, 'Nd')  # chemical potential n_down

		L_cells=self.lat.Ls[0]
		if abs(V) > 0:
			# shape (L_cells,)  →  [+V/2, -V/2, +V/2, …]
			stagger = np.asarray([ +V/2 if (x % 2 == 0) else -V/2
								for x in range(L_cells) ])
			for alpha in range(len(self.lat.unit_cell)):      # usually alpha == 0
				self.add_onsite(stagger, alpha, 'Nu')         # n↑   term
				self.add_onsite(stagger, alpha, 'Nd')         # n↓   term


# chi is bond dimension of MPS (chi ~ log (S_ent))
# Should check increasing chi to see convergence of E_gnd
def get_gnd(L, chi, U=1, t=1, mu=0,V=0):
	# initialize Hamiltonian
	model = RealSpaceHubbard1D({'L': L, 'U':U, 't':t, 'bc':'open', 'bc_MPS':'finite', 'mu':mu,'V':V})
	
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
def get_gnd_infinite(chi, U=1, t=1, mu=0,V=0):
	# initialize Hamiltonian
	model = RealSpaceHubbard1D({'L': 2, 'U':U, 't':t, 'bc':'periodic', 'bc_MPS':'infinite', 'mu':mu,'V':V})
	
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
		

if __name__ == "__main__":
	print('real space dmrg main character')
	lattice_points=10
	cluster_size=2

	bond_dimensions = [8, 16, 32]
	system_sizes = [8, 12, 16, 100]

	chi=bond_dimensions[0]
	U=10
	mu_0=U/2
	E_density, psi, filling = get_gnd_infinite(chi, U, mu=mu_0)

	energies=[]
	fillings=[]

	for chi in tqdm(bond_dimensions):
		E_density,psi,filling= get_gnd_infinite(chi, U, mu=mu_0)

		energies.append(E_density)
		fillings.append(filling)

	# Create subplots for energy and filling vs bond dimension
	fig = make_subplots(
		rows=1, cols=2,
		subplot_titles=('Energy Density vs Bond Dimension', 'Filling vs Bond Dimension'),
		x_title='Bond Dimension (χ)'
	)

	# Add energy trace
	fig.add_trace(
		go.Scatter(
			x=bond_dimensions, 
			y=energies,
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
			x=bond_dimensions, 
			y=fillings,
			mode='lines+markers',
			name='Filling',
			line=dict(color='red', width=2),
			marker=dict(size=8)
		),
		row=1, col=2
	)

	# Update layout
	fig.update_layout(
		title=f'DMRG Convergence Study (U={U}, μ={mu_0})',
		showlegend=True,
		height=500,
		width=900
	)

	# Update y-axis labels
	fig.update_yaxes(title_text="Energy Density", row=1, col=1)
	fig.update_yaxes(title_text="Filling", row=1, col=2)

	# Show the plot
	fig.show()

	print(f'Bond dimensions: {bond_dimensions}')
	print(f'Energies: {energies}')
	print(f'Fillings: {fillings}')
	
