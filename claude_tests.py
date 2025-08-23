import numpy as np
import itertools
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.mps import MPS
from tenpy.algorithms.exact_diag import get_numpy_Hamiltonian, get_full_wavefunction


class StandardFermiHubbard1D(CouplingMPOModel):
    """Standard 1D Fermi-Hubbard with NN and NNN hopping.
    
    H = -t * sum_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.)
        -t2 * sum_{<<i,j>>,σ} (c†_{i,σ} c_{j,σ} + h.c.)
        + U * sum_i n_{i,↑} n_{i,↓}
        - μ * sum_i (n_{i,↑} + n_{i,↓})
    """
    
    def init_sites(self, model_params):
        # No charge conservation for full spectrum access
        return SpinHalfFermionSite(cons_N=None, cons_Sz=None)
    
    def init_lattice(self, model_params):
        L = model_params['L']
        bc = model_params.get('bc', 'open')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        site = self.init_sites(model_params)
        return Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=site)

    def init_terms(self, model_params):
        t = model_params.get('t', 1.0)      # NN hopping
        t2 = model_params.get('t2', 0.0)    # NNN hopping  
        U = model_params.get('U', 0.0)      # Hubbard U
        mu = model_params.get('mu', 0.0)    # chemical potential

        # On-site terms
        for u in range(len(self.lat.unit_cell)):
            if U != 0.0:
                self.add_onsite(U, u, 'NuNd', category='U')
            if mu != 0.0:
                self.add_onsite(-mu, u, 'Ntot', category='mu')

        # NN hopping: standard tight-binding has NEGATIVE sign
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if t != 0.0:
                self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

        # NNN hopping: use explicit add_coupling with correct dx
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            if t2 != 0.0:
                self.add_coupling(-t2, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
                self.add_coupling(-t2, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)


def single_particle_projected(model, return_eig=False, spin_block='both'):
    """Project Hamiltonian onto N=1 subspace."""
    lat = model.lat
    sites = lat.mps_sites()
    L = len(sites)
    bc_mps = getattr(lat, 'bc_MPS', 'finite')

    # Full Hamiltonian
    H_full = get_numpy_Hamiltonian(model, from_mpo=True, undo_sort_charge=True)

    # Build N=1 basis
    spins = {'both': ['up', 'down'], 'up': ['up'], 'down': ['down']}[spin_block]
    cols = []
    for s in spins:
        for i in range(L):
            p_state = ['empty'] * L
            p_state[i] = s
            psi = MPS.from_product_state(sites, p_state, bc=bc_mps)
            vec = get_full_wavefunction(psi, undo_sort_charge=True)
            cols.append(vec)
    P = np.column_stack(cols)

    # Project
    H1 = P.conj().T @ H_full @ P
    if return_eig:
        w, V = np.linalg.eigh(H1)
        return w, V
    return H1


def manual_tight_binding_matrix(L, t, t2):
    """Manually construct the single-particle tight-binding Hamiltonian."""
    # Create 2L x 2L matrix (L sites x 2 spins)
    H = np.zeros((2*L, 2*L))
    
    # NN hopping for spin up (indices 0 to L-1)
    for i in range(L-1):
        H[i, i+1] = -t
        H[i+1, i] = -t
    
    # NN hopping for spin down (indices L to 2L-1) 
    for i in range(L, 2*L-1):
        H[i, i+1] = -t
        H[i+1, i] = -t
        
    # NNN hopping for spin up
    for i in range(L-2):
        H[i, i+2] = -t2
        H[i+2, i] = -t2
        
    # NNN hopping for spin down
    for i in range(L, 2*L-2):
        H[i, i+2] = -t2
        H[i+2, i] = -t2
    
    return H


def test_individual_terms():
    """Test NN-only and NNN-only separately to confirm they work."""
    L = 4
    
    def test_case(t, t2, label):
        ham_dict = {'L': L, 't': t, 't2': t2, 'mu': 0, 'U': 0, 'bc': 'open', 'bc_MPS': 'finite'}
        ham = StandardFermiHubbard1D(ham_dict)
        
        sp_evals = np.linalg.eigvalsh(single_particle_projected(ham, spin_block='both'))
        mb_ham = get_numpy_Hamiltonian(ham, from_mpo=True, undo_sort_charge=True)
        mb_evals = np.linalg.eigvalsh(mb_ham)
        
        recon = []
        for k in range(0, 2*L + 1):
            for combo in itertools.combinations(sp_evals, k):
                recon.append(sum(combo))
        recon = np.sort(np.array(recon))
        
        error = np.max(np.abs(np.sort(mb_evals) - recon))
        print(f"{label}: Error = {error:.2e}")
        return error < 1e-10
    
    print("=== Testing individual terms ===")
    nn_works = test_case(1.0, 0.0, "NN only (t=1, t2=0)")
    nnn_works = test_case(0.0, 2.0, "NNN only (t=0, t2=2)")  
    both_works = test_case(1.0, 2.0, "NN+NNN (t=1, t2=2)")
    
    print(f"NN works: {nn_works}")
    print(f"NNN works: {nnn_works}")
    print(f"NN+NNN works: {both_works}")
    print()
    
    return nn_works, nnn_works, both_works


class CorrectFermiHubbard1D(CouplingMPOModel):
    """Copy of your working trial_ham.py implementation but with V parameter."""
    
    def init_sites(self, model_params):
        return SpinHalfFermionSite(cons_N=None, cons_Sz=None)
    
    def init_lattice(self, model_params):
        L = model_params['L']
        bc = model_params.get('bc', 'open')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        site = self.init_sites(model_params)
        return Chain(L=L, bc=bc, bc_MPS=bc_MPS, site=site)

    def init_terms(self, model_params):
        t = model_params.get('t', 1.0)      
        V = model_params.get('V', 0.0)  # Using V instead of t2 to match your code    
        U = model_params.get('U', 0.0)      
        mu = model_params.get('mu', 0.0)    

        # On-site terms (exactly matching your trial_ham.py)
        for u in range(len(self.lat.unit_cell)):
            if U != 0.0:
                self.add_onsite(U, u, 'NuNd', category='U')        
            if mu != 0.0:
                self.add_onsite(-mu, u, 'Ntot', category='mu')     

        # NN hops (exactly matching your trial_ham.py)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if t != 0.0:
                self.add_coupling(-t, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True, category='hop_NN_up')
                self.add_coupling(-t, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True, category='hop_NN_dn')

        # NNN hops (exactly matching your trial_ham.py)
        if V != 0.0:
            u = 0
            self.add_multi_coupling(-V, [('Cdu', 0, u), ('Cu', 2, u)], 
                                  plus_hc=True, category='hop_NNN_up')
            self.add_multi_coupling(-V, [('Cdd', 0, u), ('Cd', 2, u)], 
                                  plus_hc=True, category='hop_NNN_dn')


def test_spectrum_reconstruction():
    """Test the corrected implementation."""
    nn_works, nnn_works, both_works = test_individual_terms()
    
    if nn_works and nnn_works and not both_works:
        print("=== Confirmed: NN and NNN work individually, but not together ===")
        print("Testing corrected implementation...")
        
        L = 4
        ham_dict = {'L': L, 't': 1.0, 't2': 2.0, 'mu': 0, 'U': 0, 'bc': 'open', 'bc_MPS': 'finite'}
        
        try:
            # Test with V parameter (matching your code style)
            ham_dict_corrected = {'L': L, 't': 1.0, 'V': 2.0, 'mu': 0, 'U': 0, 'bc': 'open', 'bc_MPS': 'finite'}
            ham = CorrectFermiHubbard1D(ham_dict_corrected)
            
            sp_evals = np.linalg.eigvalsh(single_particle_projected(ham, spin_block='both'))
            mb_ham = get_numpy_Hamiltonian(ham, from_mpo=True, undo_sort_charge=True)
            mb_evals = np.linalg.eigvalsh(mb_ham)
            
            recon = []
            for k in range(0, 2*L + 1):
                for combo in itertools.combinations(sp_evals, k):
                    recon.append(sum(combo))
            recon = np.sort(np.array(recon))
            
            error = np.max(np.abs(np.sort(mb_evals) - recon))
            print(f"Final corrected implementation error: {error:.2e}")
            
            if error < 1e-10:
                print("SUCCESS: Spectrum reconstruction now works!")
                print("The fix is to use add_multi_coupling for NNN without op_string='JW'")
            else:
                print(f"Still not working - error: {error}")
                print(f"Vacuum energy: {np.sort(mb_evals)[0]}")
                print(f"Expected: 0")
                
        except Exception as e:
            print(f"Corrected implementation failed: {e}")
            print("Need to fix the multi-coupling syntax")


if __name__ == "__main__":
    test_spectrum_reconstruction()