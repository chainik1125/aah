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
from typing import Optional, Sequence, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def aah_potential_integer_angle(V, V_sep, L_cells, phi=0.0):
    p, q = map(int, V_sep)
    i = np.arange(L_cells, dtype=int)
    residue = (p * (i % q)) % q                         # pure integer arithmetic
    angles = (2*np.pi / q) * residue + phi              # single float conversion
    pot = V * np.cos(angles)
    return pot

class RealSpaceHubbard1D(CouplingMPOModel, NearestNeighborModel):
	"""
	Input is a dictionary called model_params that includes:
		'L' (length), 'bc' ('open' or 'periodic'), 'bc_MPS' ('finite' or 'infinite'),
		't' (hopping strength), 'U' (Hubbard interaction strength), 'filling' (1 is half-filling)
	"""

	# Initialize spin-1/2 fermion d.o.f. on each site
	def init_sites(self, model_params):
		# Default: remove both particle number and spin conservation to allow DMRG to explore all sectors.
		cons_N = model_params.get('cons_N', None)
		cons_Sz = model_params.get('cons_Sz', None)
		site = tp.networks.site.SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
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
			self.add_coupling(t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)  # Cdagger_down C_down + h.c.
			self.add_coupling(t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)  # Cdagger_up C_up + h.c.
		
		# Onsite terms
		for v in range(len(self.lat.unit_cell)):
			self.add_onsite(U, v, 'NuNd')  # Hubbard n_up n_down term
			self.add_onsite(-mu, v, 'Nu')  # chemical potential n_up
			self.add_onsite(-mu, v, 'Nd')  # chemical potential n_down

		L_cells=self.lat.Ls[0]
		V_sep=model_params.get('V_sep', None)
		if V_sep is not None:
			# if isinstance(V_sep, tuple) and len(V_sep) == 2:
			# 	p,q=V_sep
			# 	alpha=float(p)/float(q)
			# else:
			# 	raise ValueError("V_sep must be a tuple of two integers")
			
			# V_array=V*np.cos(2*np.pi*alpha*np.arange(L_cells))
			# # shape (L_cells,)  →  [+V/2, -V/2, +V/2, …]
			# stagger = np.asarray([V_array[x] for x in range(L_cells) ])

			V_array=aah_potential_integer_angle(V, V_sep, L_cells)
			for alpha in range(len(self.lat.unit_cell)):      # usually alpha == 0
				self.add_onsite(V_array, alpha, 'Nu')         # n↑   term
				self.add_onsite(V_array, alpha, 'Nd')         # n↓   term
		#Old way of doing it which assumes pi modulation
		elif abs(V) > 0:
			stagger = np.asarray([ +V if (x % 2 == 0) else -V
								for x in range(L_cells) ])
			for alpha in range(len(self.lat.unit_cell)):      # usually alpha == 0
				self.add_onsite(stagger, alpha, 'Nu')         # n↑   term
				self.add_onsite(stagger, alpha, 'Nd')         # n↓   term


# chi is bond dimension of MPS (chi ~ log (S_ent))
# Should check increasing chi to see convergence of E_gnd
def get_gnd(L, chi, U=1, t=1, mu=0, V=0, V_sep=None):
	# initialize Hamiltonian
	model = RealSpaceHubbard1D({'L': L, 'U':U, 't':t, 'bc':'open', 'bc_MPS':'finite', 'mu':mu, 'V':V, 'V_sep':V_sep})
	
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


def get_gnd_fixed_filling(L, chi, filling_target, U=1, t=1, V=0, V_sep=None):
	"""
	Finite DMRG in the canonical ensemble (fixed total particle number).

	Args:
		L: Number of sites.
		chi: Max bond dimension.
		filling_target: Target filling per site (0 <= n <= 2).
		U, t, V, V_sep: Model parameters (same conventions as get_gnd).

	Returns:
		(E, psi, filling): Total ground state energy, MPS, and measured filling per site.
	"""
	if filling_target is None:
		raise ValueError("filling_target must be provided for fixed-filling DMRG.")

	filling_target = float(filling_target)
	if not np.isfinite(filling_target):
		raise ValueError(f"filling_target must be finite, got {filling_target!r}.")
	if filling_target < -1e-12 or filling_target > 2.0 + 1e-12:
		raise ValueError(f"filling_target must be between 0 and 2 (per site), got {filling_target}.")

	N_target_float = filling_target * L
	N_target = int(round(N_target_float))
	N_target = max(0, min(2 * L, N_target))

	# Use an (almost) unpolarized initial state: N_up ~= N_down.
	N_up_target = N_target // 2
	N_down_target = N_target - N_up_target

	min_double_occupancies = max(0, N_target - L)
	if min_double_occupancies > min(N_up_target, N_down_target):
		raise ValueError(
			f"Cannot realize N_target={N_target} on L={L} with N_up={N_up_target}, N_down={N_down_target}."
		)

	N_full = min_double_occupancies
	N_up_singles = N_up_target - N_full
	N_down_singles = N_down_target - N_full

	product_state = L * ['empty']
	idx = 0
	for _ in range(N_full):
		product_state[idx] = 'full'
		idx += 1
	for _ in range(N_up_singles):
		product_state[idx] = 'up'
		idx += 1
	for _ in range(N_down_singles):
		product_state[idx] = 'down'
		idx += 1

	# Canonical ensemble: enforce U(1) symmetry for total N and set mu=0.
	model = RealSpaceHubbard1D({
		'L': L,
		'U': U,
		't': t,
		'bc': 'open',
		'bc_MPS': 'finite',
		'mu': 0.0,
		'V': V,
		'V_sep': V_sep,
		'cons_N': 'N',
		'cons_Sz': None,
	})

	psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state)

	dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-8},
		'max_E_err': 1e-8, 'max_S_err': 1e-6, 'min_sweeps': 5, 'max_sweeps': 50, 'max_trunc_err': None}

	engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
	E, psi = engine.run()

	N_up = np.mean([psi.expectation_value('Nu', i) for i in range(L)])
	N_down = np.mean([psi.expectation_value('Nd', i) for i in range(L)])
	filling = N_up + N_down

	return E, psi, filling


def get_finite_dmrg_density_wave_observable(
	L,
	chi,
	U=1,
	t=1,
	mu=0,
	V=0,
	V_sep=None,
	*,
	filling_target=None,
	dmrg_fixed_filling: bool = False,
	return_profile: bool = False,
):
	if V_sep is None:
		raise ValueError("V_sep must be provided to define rho_Q.")

	if dmrg_fixed_filling:
		if filling_target is None:
			raise ValueError("filling_target must be provided when dmrg_fixed_filling=True.")
		_, psi, _ = get_gnd_fixed_filling(
			L,
			chi,
			filling_target,
			U,
			t,
			V,
			V_sep,
		)
	else:
		_, psi, _ = get_gnd(
			L,
			chi,
			U,
			t,
			mu,
			V,
			V_sep,
		)

	density_profile = np.array([
		float(np.real(psi.expectation_value('Nu', site) + psi.expectation_value('Nd', site)))
		for site in range(L)
	], dtype=float)

	p, q = map(int, V_sep)
	Q = 2 * np.pi * p / q
	site_indices = np.arange(density_profile.size)
	rho_q = float(np.abs(np.mean(np.exp(1j * Q * site_indices) * density_profile)))

	if return_profile:
		return rho_q, density_profile
	return rho_q


def _resolve_bulk_sites(L: int, bulk_slice=None) -> np.ndarray:
	if bulk_slice is None:
		if L < 8:
			return np.arange(L, dtype=int)
		start = L // 4
		stop = L - start
		return np.arange(start, stop, dtype=int)
	if isinstance(bulk_slice, slice):
		return np.arange(L, dtype=int)[bulk_slice]
	if isinstance(bulk_slice, tuple) and len(bulk_slice) == 2:
		start, stop = bulk_slice
		return np.arange(int(start), int(stop), dtype=int)
	sites = np.asarray(bulk_slice, dtype=int)
	if sites.ndim != 1:
		raise ValueError(f"bulk_slice must resolve to a 1-D list of sites, got shape {sites.shape}")
	return sites


def _get_finite_dmrg_ground_state(
	L,
	chi,
	*,
	U=1,
	t=1,
	mu=0,
	V=0,
	V_sep=None,
	filling_target=None,
	dmrg_fixed_filling: bool = False,
):
	if dmrg_fixed_filling:
		if filling_target is None:
			raise ValueError("filling_target must be provided when dmrg_fixed_filling=True.")
		return get_gnd_fixed_filling(
			L,
			chi,
			filling_target,
			U,
			t,
			V,
			V_sep,
		)
	return get_gnd(
		L,
		chi,
		U,
		t,
		mu,
		V,
		V_sep,
	)


def get_dmrg_static_structure_factors(
	L,
	chi,
	*,
	U=1,
	t=1,
	mu=0,
	V=0,
	V_sep=None,
	filling_target=None,
	dmrg_fixed_filling: bool = False,
	q_values: Optional[Sequence[float]] = None,
	bulk_slice=None,
	return_correlators: bool = False,
):
	_, psi, filling = _get_finite_dmrg_ground_state(
		L,
		chi,
		U=U,
		t=t,
		mu=mu,
		V=V,
		V_sep=V_sep,
		filling_target=filling_target,
		dmrg_fixed_filling=dmrg_fixed_filling,
	)

	all_sites = np.arange(L, dtype=int)
	bulk_sites = _resolve_bulk_sites(L, bulk_slice)
	if bulk_sites.size == 0:
		raise ValueError("bulk_slice resolved to zero sites.")

	n_up = np.array([float(np.real(psi.expectation_value('Nu', i))) for i in all_sites], dtype=float)
	n_down = np.array([float(np.real(psi.expectation_value('Nd', i))) for i in all_sites], dtype=float)
	n_total = n_up + n_down
	sz_profile = 0.5 * (n_up - n_down)

	sites_list = bulk_sites.tolist()
	nu_nu = np.asarray(psi.correlation_function('Nu', 'Nu', sites1=sites_list, sites2=sites_list), dtype=complex)
	nu_nd = np.asarray(psi.correlation_function('Nu', 'Nd', sites1=sites_list, sites2=sites_list), dtype=complex)
	nd_nu = np.asarray(psi.correlation_function('Nd', 'Nu', sites1=sites_list, sites2=sites_list), dtype=complex)
	nd_nd = np.asarray(psi.correlation_function('Nd', 'Nd', sites1=sites_list, sites2=sites_list), dtype=complex)

	n_bulk = n_total[bulk_sites]
	sz_bulk = sz_profile[bulk_sites]
	charge_corr_raw = np.real(nu_nu + nu_nd + nd_nu + nd_nd)
	spin_corr_raw = 0.25 * np.real(nu_nu - nu_nd - nd_nu + nd_nd)
	charge_corr = charge_corr_raw - np.outer(n_bulk, n_bulk)
	spin_corr = spin_corr_raw - np.outer(sz_bulk, sz_bulk)

	if q_values is None:
		q_values = 2.0 * np.pi * np.arange(L // 2 + 1, dtype=float) / float(L)
	else:
		q_values = np.asarray(q_values, dtype=float)

	positions = bulk_sites.astype(float)
	norm = float(len(bulk_sites))
	Nq = np.empty(len(q_values), dtype=float)
	Sq = np.empty(len(q_values), dtype=float)
	for idx, q in enumerate(q_values):
		phase = np.exp(1j * q * (positions[:, np.newaxis] - positions[np.newaxis, :]))
		Nq[idx] = float(np.real(np.sum(phase * charge_corr)) / norm)
		Sq[idx] = float(np.real(np.sum(phase * spin_corr)) / norm)

	result = {
		"q_values": np.asarray(q_values, dtype=float),
		"Nq": Nq,
		"Sq": Sq,
		"density_profile": n_total,
		"spin_profile": sz_profile,
		"bulk_sites": bulk_sites,
		"filling": float(filling),
	}
	if return_correlators:
		result["charge_corr"] = charge_corr
		result["spin_corr"] = spin_corr
	return result

# uses iDMRG to get gnd state energy density in thermodynamic limit (L -> \infty)
# may not be best choice if system is gapless (test it a bit to check)
def get_gnd_infinite(chi, U=1, t=1, mu=0, V=0, V_sep=None):
	# initialize Hamiltonian
	
	# Determine unit cell size based on V_sep
	if V_sep is not None:
		# V_sep is a tuple (p, q) - need q sites for the unit cell
		# to capture the periodicity of the potential
		L_V_sep = V_sep[1]
	else:
		# Default unit cell size for standard iDMRG
		L_V_sep = 2
	
	model = RealSpaceHubbard1D({'L': L_V_sep, 'U':U, 't':t, 'bc':'periodic', 'bc_MPS':'infinite', 'mu':mu, 'V':V, 'V_sep':V_sep})
	
	# Choose initial state based on chemical potential
	if mu > U:
		# Start from fully filled state for positive mu
		product_state = L_V_sep * ['full']
	elif mu < 0:
		# Start from empty state for negative mu
		product_state = L_V_sep * ['empty']
	else:
		# Start from Néel state at mu=0
		product_state = (L_V_sep // 2) * ['up', 'down']
		# Handle odd L_V_sep
		if L_V_sep % 2 == 1:
			product_state.append('up')
	
	psi = tp.MPS.from_product_state(model.lat.mps_sites(), product_state, 'infinite')

	# parameters you can ignore for now (affect precision of DMRG)
	dmrg_params = {'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-8},
		'max_E_err': 1e-8, 'max_S_err': 1e-6, 'min_sweeps': 5, 'max_sweeps': 50, 'max_trunc_err': None}

	# initialize two site DMRG engine
	engine = tp.TwoSiteDMRGEngine(psi, model, dmrg_params)
	E, psi = engine.run()
	
	# For infinite system, measure on the unit cell
	N_up = np.mean([psi.expectation_value('Nu', i) for i in range(L_V_sep)])
	N_down = np.mean([psi.expectation_value('Nd', i) for i in range(L_V_sep)])
	filling = N_up + N_down
	
	return E, psi, filling

def run_dmrg_method(U, mu_0, V=0, V_sep=None, t=1, system_size=10, chi=32):
    """
    Run real-space DMRG calculation
    
    Args:
        U: Hubbard interaction strength
        mu_0: Chemical potential
        V: Staggered potential (default 0)
        t: Hopping parameter (default 1)
        system_size: Number of lattice sites
        chi: Bond dimension for DMRG
    
    Returns:
        (energy, filling): Total energy and filling
    """
    energy, psi, filling = get_gnd_infinite(chi=chi, U=U, t=t, mu=mu_0, V=V, V_sep=V_sep)
    
    return energy, filling, psi
		

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
	
