import logging

from aah_code.clusters import ClusterExperiment
from aah_code.basis import LocalClusterBasis
from aah_code.global_params import StatesParams,HamiltonianParams
from aah_code.utils import mu_tilde_coefficient,cosine_dispersion
from tenpy.models import CouplingMPOModel,lattice
import tenpy as tp
from tenpy.algorithms import exact_diag
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tenpy.networks import mps
from tqdm import tqdm
import itertools

from tenpy.networks.mps import MPS
from tenpy.algorithms.exact_diag import get_numpy_Hamiltonian, get_full_wavefunction

from aah_code.real_space_dmrg import run_dmrg_method, get_gnd_infinite,get_gnd
# Set Plotly to use browser renderer to avoid nbformat issues
#pio.renderers.default = "browser"


logger=logging.getLogger(__name__)





from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfFermionSite


#Lets try to do it from Hubbard1D




class ExtendedFermiHubbard1D(CouplingMPOModel):
	"""1D spinful fermions with U, mu, NN/NNN hopping and NN/NNN density-density V.

	H = -t1 * sum_<i,i+1,s> (c^†_{i,s} c_{i+1,s} + h.c.)
		-t2 * sum_<<i,i+2>,s> (c^†_{i,s} c_{i+2,s} + h.c.)
		+U * sum_i n_{i,up} n_{i,down}
		-mu * sum_i (n_{i,up} + n_{i,down})
		+V1 * sum_<i,i+1> n_i n_{i+1}
		+V2 * sum_<<i,i+2>> n_i n_{i+2}
	"""
	# Make this a 1D chain by default (length L, boundary conditions via model_params)
	default_lattice = Chain
	force_default_lattice = True

	def init_sites(self, model_params):
		# Conserved charges; use 'best' in your params if you want auto-choice
		
		# Spin-1/2 fermionic site provides Cu/Cdu, Cd/Cdd, Nu, Nd, NuNd, Ntot, etc.
		return SpinHalfFermionSite(cons_N=None, cons_Sz=None)

	def init_terms(self, model_params):
		# Read couplings (all optional, default 0 except t1=1)
		t = model_params.get('t', 1.0)   # NN hopping
		V = model_params.get('V', 0.0)   # NNN hopping
		U  = model_params.get('U', 0.0)    # onsite interaction
		mu = model_params.get('mu', 0.0)   # chemical potential (H -= mu * n)
		

		# Onsite: chemical potential and Hubbard-U
		for u in range(len(self.lat.unit_cell)):
			if mu != 0.0:
				self.add_onsite(-mu, u, 'Ntot', category='mu')     # -mu * n_i
			if U  != 0.0:
				self.add_onsite(U,  u, 'NuNd', category='U')       # U  * n_up n_down

		# Hopping (spin up & down). JW strings are inserted automatically; use plus_hc. 
		if t != 0.0:
			for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
				self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx,  plus_hc=True, category='hop_NN_up')
				self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx,  plus_hc=True, category='hop_NN_dn')
		if V != 0.0:
			for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
				self.add_coupling(V, u1, 'Cdu', u2, 'Cu', dx,  plus_hc=True, category='hop_NNN_up')
				self.add_coupling(V, u1, 'Cdd', u2, 'Cd', dx,  plus_hc=True, category='hop_NNN_dn')




class test_hub(CouplingMPOModel):
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
		
		for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
			self.add_coupling(V, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
			self.add_coupling(V, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)

		
		
		# Onsite terms
		# for v in range(len(self.lat.unit_cell)):
		# 	self.add_onsite(U, v, 'NuNd')  # Hubbard n_up n_down term
		# 	self.add_onsite(-mu, v, 'Nu')  # chemical potential n_up
		# 	self.add_onsite(-mu, v, 'Nd')  # chemical potential n_down

		
			# if abs(V) > 0:        # i = 0 … L-1
			# 	sign =  +V if (v % 2 == 0) else -V   # even sites +V, odd sites –V
			# 	self.add_onsite(sign, v, 'Nu')       # n↑ part
			# 	self.add_onsite(sign, v, 'Nd')
		# self.add_coupling(V, 0, 'Cdd', 0, 'Cd', 2, plus_hc=True)
		# self.add_coupling(V, 0, 'Cdu', 0, 'Cu', 2, plus_hc=True)

# minimal_hubbard_nnn.py
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite

class Hubbard1D_NN_NNN(CouplingMPOModel):
	"""1D Fermi-Hubbard with NN (t1) and NNN (t2) hopping.
	   No charge or spin quantum numbers are conserved."""
	def init_sites(self, model_params):
		# turn OFF number and Sz conservation in the site
		return SpinHalfFermionSite(cons_N=None, cons_Sz=None)
	
	def init_lattice(self,model_params):
		L = model_params['L']
		bc = model_params.get('bc', 'open')          # <- consumes 'bc'
		bc_MPS = model_params.get('bc_MPS', 'finite')# <- consumes 'bc_MPS'
		site = self.init_sites(model_params)
		return Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=site)

	def init_terms(self, model_params):
		t = model_params.get('t', 1.0)      # NN hopping
		V = model_params.get('V', 0.0)      # NNN hopping
		U  = model_params.get('U', 0.0)       # on-site
		mu = model_params.get('mu', 0.0)      # chemical potential

		# on-site terms
		for u in range(len(self.lat.unit_cell)):
			if U != 0.0:
				self.add_onsite(U, u, 'NuNd', category='U')        # U n_up n_dn
			if mu != 0.0:
				self.add_onsite(-mu, u, 'Ntot', category='mu')     # -mu (n_up + n_dn)

		# inside init_terms(...)
		# NN hops (two-site): fine as-is
		for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
			if t != 0.0:
				self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True, category='hop_NN_up')
				self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True, category='hop_NN_dn')

		# NNN hops (three-site path i -> i+2): use add_multi_coupling with op_string='JW'
		# ops is a list of (op_name, relative_dx, unit_cell_index) tuples.
		# For Chain (one site per unit cell), u = 0.
		if V != 0.0:
			u = 0
			# ↑: c†_{i+2,↑} [JW over sites between] c_{i,↑} + h.c.
			self.add_multi_coupling(-V, [('Cdu', 0, u), ('Cu', 2, u)],
									op_string='JW', plus_hc=True, category='hop_NNN_up')
			# ↓: c†_{i+2,↓} [JW] c_{i,↓} + h.c.
			self.add_multi_coupling(-V, [('Cdd', 0, u), ('Cd', 2, u)],
									op_string='JW', plus_hc=True, category='hop_NNN_dn')
						

def single_particle_projected(model, return_eig=False, spin_block='both'):
	"""
	Project the full Hamiltonian onto the N=1 subspace spanned by
	{|i,↑>, |i,↓>} and return either the matrix or its eigensystem.

	Args:
		model: any finite TeNPy model with an MPO Hamiltonian.
		return_eig (bool): if True, returns (eigvals, eigvecs) of the projected H.
		spin_block: 'both' -> 2L basis (|i,↑>,|i,↓>), 'up' or 'down' -> L basis.

	Returns:
		H1 (2Lx2L or LxL) or (w, V) if return_eig=True.
	"""
	lat = model.lat
	sites = lat.mps_sites()                      # sites for building product states
	L = len(sites)
	bc_mps = getattr(lat, 'bc_MPS', 'finite')

	# 1) Full Hamiltonian in the same basis ordering as get_full_wavefunction
	H_full = get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)  # dense (d^L x d^L)
	# (Works directly from the MPO representation of your CouplingMPOModel.) :contentReference[oaicite:1]{index=1}

	# 2–3) Build N=1 basis columns (as full vectors) and stack them into P
	spins = {'both': ['up', 'down'], 'up': ['up'], 'down': ['down']}[spin_block]
	cols = []
	for s in spins:
		for i in range(L):
			p_state = ['empty'] * L
			p_state[i] = s
			psi = MPS.from_product_state(sites, p_state, bc=bc_mps)              # product state |i,s>
			vec = get_full_wavefunction(psi, undo_sort_charge=True)              # 1D numpy array
			cols.append(vec)
	P = np.column_stack(cols)                                                    # shape: (d^L, dim_sub)
	# MPS.from_product_state + state labels are canonical; get_full_wavefunction pairs with
	# get_numpy_Hamiltonian’s ordering. :contentReference[oaicite:2]{index=2}

	# 4) Project and (optionally) diagonalize
	H1 = P.conj().T @ H_full @ P                                                # (2L x 2L) or (L x L)
	if return_eig:
		w, V = np.linalg.eigh(H1)
		return w, V
	return H1

def eigvals_in_sector(model, N):
    H = get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)
    sites = model.lat.mps_sites(); L = len(sites); bc = getattr(model.lat, 'bc_MPS', 'finite')
    cols = []
    # orbitals 0..L-1 are ↑, L..2L-1 are ↓ (ordering doesn’t matter for energies)
    for occ in itertools.combinations(range(2*L), N):
        up = [0]*L; dn = [0]*L
        for o in occ: (up if o < L else dn)[o % L] = 1
        p_state = [('updown' if up[i] and dn[i] else
                    'up'     if up[i] else
                    'down'   if dn[i] else 'empty') for i in range(L)]
        psi = MPS.from_product_state(sites, p_state, bc=bc)
        cols.append(get_full_wavefunction(psi, undo_sort_charge=True))
    P = np.column_stack(cols)
    return np.sort(np.linalg.eigvalsh(P.conj().T @ H @ P))




from tenpy.models.fermions_spinless import FermionChain

class TTprimeSpinfulChain(CouplingMPOModel):
    """Spin-1/2 fermions on a 1D chain with NN hopping t and NNN hopping t'."""
    def init_sites(self, p):
        # conserve total N and Sz; degenerate spin DOF
        return SpinHalfFermionSite(cons_N=None, cons_Sz=None)

    # CouplingMPOModel already defaults to a Chain lattice with length p['L'].

    def init_terms(self, p):
        t  = float(p.get("t", 1.0))       # NN hopping
        tp = float(p.get("tp", 0.0))      # NNN hopping
        mu = float(p.get("mu", 0.0))      # chemical potential
        U  = float(p.get("U", 0.0))       # onsite Hubbard U (optional)

        # onsite: -mu * (n_up + n_down) + U * n_up n_down
        self.add_onsite(-mu, 0, "Ntot")
        if abs(U) > 0:
            self.add_onsite(U, 0, "NuNd")

        # NN hopping: -t * (c†_{iσ} c_{i+1,σ} + h.c.)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t,  u1, "Cdu", u2, "Cu", dx, plus_hc=True)  # spin ↑
            self.add_coupling(-t,  u1, "Cdd", u2, "Cd", dx, plus_hc=True)  # spin ↓

        # NNN hopping: -t' * (c†_{iσ} c_{i+2,σ} + h.c.)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(-tp, u1, "Cdu", u2, "Cu", dx, plus_hc=True)
            self.add_coupling(-tp, u1, "Cdd", u2, "Cd", dx, plus_hc=True)







if __name__ == "__main__":
	
	
	ham_dict = {
    "L":  4,
    "t":  1.0,
    "tp": 0.25,
    "mu": 0.0,
    "bc": "open",      # or "periodic"
    "bc_MPS": "finite",
		}
	
	
	ham=TTprimeSpinfulChain(ham_dict)
	
	sp_ham=single_particle_projected(ham,spin_block='up')

	# eigenvalues (spin-degenerate)
	sp_evals, _ = np.linalg.eigh(sp_ham)
	sp_evals = np.repeat(sp_evals, 2)  # ↑/↓ degeneracy
	
	# sp_ham=single_particle_projected(ham,spin_block='up')
	
	mb_ham=get_numpy_Hamiltonian(ham,from_mpo=True,undo_sort_charge=True)

	print(f'num sites: {ham.lat.N_sites}')

	total_particles=ham.lat.N_sites*2

	sp_evals,sp_evecs=np.linalg.eigh(sp_ham)
	sp_evals=np.repeat(sp_evals,2)

	reconstructed_eigvals=[]
	for particle_num in range(total_particles+1):
		test=np.array(list(itertools.combinations(sp_evals, particle_num)))
		summed_test=test.sum(axis=1)
		print(f'test shape: {test.shape}')
		reconstructed_eigvals.extend(summed_test)
	
	reconstructed_eigvals.sort()
	
	mb_evals,mb_evecs=np.linalg.eigh(mb_ham)

	print(f'mb evals first 4: {mb_evals[:4]}')
	print(f'reconstructed evals first 4: {reconstructed_eigvals[:4]}')

	np.testing.assert_array_almost_equal(mb_evals,reconstructed_eigvals)




	#ChatGPT
	# ham_dict = {'L': 4, 't': 1, 'V': 1, 'mu': 0, 'U': 0, 'bc': 'open', 'bc_MPS': 'finite'}
	# ham = Hubbard1D_NN_NNN(ham_dict)

	# # single-particle (both spins -> size 2L)
	# sp_evals = np.linalg.eigvalsh(single_particle_projected(ham, spin_block='both'))

	# # all many-body eigenvalues
	# mb_ham = get_numpy_Hamiltonian(ham, from_mpo=True, undo_sort_charge=True)
	# mb_evals = np.linalg.eigvalsh(mb_ham)

	# # reconstruct by summing over occupations of the 2L spin-orbitals
	# recon = []
	# for k in range(0, 2*ham.lat.N_sites + 1):
	# 	for combo in itertools.combinations(sp_evals, k):
	# 		recon.append(sum(combo))
	# recon = np.sort(np.array(recon))

	# print(f'recon first 4: {recon[:4]}')
	# print(f'mb first four: {mb_evals[:4]}')

	# # These two should now agree up to ~1e-10 for small L
	# print(np.max(np.abs(np.sort(mb_evals) - recon)))






		



	
	
