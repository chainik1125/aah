import numpy as np
from aah_code.cluster_model.logging_config import debug, info

from aah_code.cluster_model.clustering import generate_clusters,convert_site_clusters_to_k
from aah_code.cluster_model.model_ham import make_cluster_ham
from aah_code.hamiltonian import FullSpectrum,SpectrumSolver,HamiltonianParams,StatesParams,ClusterExperiment,MismatchedQuick,get_spectra
from aah_code.cluster_model.model import ClusterModelConfig,PhysicalParams




def get_general_spectra(run_config:ClusterModelConfig,return_ham:bool=False):
	#Lets try to make the hamiltonian

	k_points=[]
	energy_spectrum=[]
	number_spectrum=[]
	spin_spectrum=[]
	ham_objects=[]
	U=run_config.physical_params.U
	V=run_config.physical_params.V
	mu_0=run_config.physical_params.mu_0
	t=run_config.physical_params.t
	L=run_config.L
	Nc=run_config.int_cluster_size
	solver_method=run_config.solver_method
	int_sep_ratio=run_config.cluster_separation_ratio
	v_sep_ratio=run_config.V_separation_ratio
	states_retained=run_config.states_retained


	all_superclusters=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)
	all_superclusters_k=convert_site_clusters_to_k(all_superclusters,L)
	
	all_superclusters_k_unsqueezed=np.expand_dims(all_superclusters_k,axis=-1)

	for sc_idx,cluster_k in enumerate(all_superclusters_k_unsqueezed):
		cluster_k_no_last_dim=cluster_k[...,0]
		cluster_indices=all_superclusters[sc_idx]
		# Log verbose information (only shown when CLUSTER_VERBOSE=true)
		debug(f'cluster_k_no_last_dim shape: {cluster_k_no_last_dim.shape}')
		debug(f'cluster_k: {cluster_k/np.pi}')
		debug(f'cluster_indices: {cluster_indices}')
		sc_cluster_ham,sc_cluster_basis=make_cluster_ham(cluster_k_no_last_dim,cluster_indices,t,V,U,mu_0,L,Nc,int_sep_ratio,v_sep_ratio,ham_lib='quspin')

		input=(sc_cluster_ham,sc_cluster_basis)	
	
		solver=SpectrumSolver(input,sc_cluster_basis,ham_lib='quspin',solver_method=solver_method,states_retained=states_retained)
		eigvals,eigvecs,n_ups,n_downs,n_tot=solver.solve_spectrum()
		
		k_points.append(cluster_k)
		energy_spectrum.append(eigvals)
		number_spectrum.append(n_tot)
		spin_spectrum.append(np.array([n_ups,n_downs]))
		if return_ham:
			ham_objects.append(sc_cluster_ham)

	k_points=np.stack(k_points,axis=0)
	energy_spectrum=np.stack(energy_spectrum,axis=0)
	number_spectrum=np.stack(number_spectrum,axis=0)
	spin_spectrum=np.stack(spin_spectrum,axis=0)

	if return_ham:
		return ham_objects
	else:
		return k_points,energy_spectrum,number_spectrum,spin_spectrum


def get_general_expectations(run_config:ClusterModelConfig):

	spectra_4tuple=get_general_spectra(run_config)
	
	#state_params=StatesParams(spin_states=2)
	physical_params=run_config.physical_params
	info(f'k points shape: {spectra_4tuple[0].shape}, energies shape: {spectra_4tuple[1].shape}, number_spectrum shape: {spectra_4tuple[2].shape}, spin spectrum shape: {spectra_4tuple[3].shape}')

	full_spectrum_obj=FullSpectrum(None,None,physical_params,None)
	system_expectations,cluster_expectations=full_spectrum_obj.get_cluster_thermodynamic_expectations(spectra_4tuple,None)
	
	return system_expectations,cluster_expectations




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
	
	info(f'k points shape: {spectra_4tuple[0].shape}, energies shape: {spectra_4tuple[1].shape}, number_spectrum shape: {spectra_4tuple[2].shape}, spin spectrum shape: {spectra_4tuple[3].shape}')

	full_spectrum_obj=FullSpectrum(None,state_params,physical_params,None)
	system_expectations,cluster_expectations=full_spectrum_obj.get_cluster_thermodynamic_expectations(spectra_4tuple,None)
	
	return system_expectations,cluster_expectations

def run_general_cluster_method(U,mu_0,V,t,L,Nc,v_sep_ratio,int_sep_ratio,ham_lib:str='quspin'):
	"""
	Run general cluster method
	"""
	
	all_superclusters_idx=generate_clusters(L,Nc,int_sep_ratio,v_sep_ratio)
	all_superclusters_k=convert_site_clusters_to_k(all_superclusters_idx,L)

	all_superclusters_k = np.expand_dims(all_superclusters_k, axis=-1)


	print(f"all_superclusters_k shape: {all_superclusters_k.shape}")
	print(f"all_superclusters_idx shape: {all_superclusters_idx.shape}")
	print(f'first supercluster idx: {all_superclusters_idx[0]}')
	print(f'first supercluster k: {all_superclusters_k[0]/np.pi}')

	

	
	

def run_cluster_old(U, mu_0, V=0, t=1, system_size=10,ham_lib:str='tenpy'):
	"""
	Run cluster method calculation using corrected FullSpectrum class
	
	Args:
		U: Hubbard interaction strength
		mu_0: Chemical potential
		V: Staggered potential (default 0)
		t: Hopping parameter (default 1)
		system_size: Number of lattice sites
	
	Returns:
		(energy, filling): Total energy and filling
	"""
	# Create grid and partition into clusters (proper cluster method)
	cluster_size = 2  # 2-site clusters
	cluster_k_generator = system_size // 2  # π separation case
	
	cluster_experiment = ClusterExperiment(
		cluster_size=cluster_size,
		lattice_points=system_size,
		cluster_k_generator=cluster_k_generator,
	)
	
	# Generate k-points grid and clusters
	k_points = cluster_experiment.generate_clusters()
	
	# Set up parameters
	state_params = StatesParams(spin_states=2)
	physical_params = HamiltonianParams(U=U, V=V, hopping=t, mu_0=mu_0)
	
	# Now use the corrected FullSpectrum class
	full_spectrum_object = FullSpectrum(k_points, state_params, physical_params,ham_lib=ham_lib)
	cluster_spectra = full_spectrum_object.get_full_spectrum()
	
	# Get ground state expectations (zero temperature)
	system_expectations, cluster_expectations = full_spectrum_object.get_cluster_thermodynamic_expectations(
		cluster_spectra, temperature=None
	)
	
	energy, filling, spin = system_expectations
	
	return energy, filling

if __name__ == "__main__":
	U=0
	mu_0=U/2
	V=2.0
	t=1.0
	L=12
	Nc=2
	v_sep_ratio=(1,6)
	int_sep_ratio=(1,6)

	physical_params=PhysicalParams(U=U,mu_0=mu_0,V=V,t=t)

	run_config=ClusterModelConfig(
		L=L,
		int_cluster_size=Nc,
		cluster_separation_ratio=int_sep_ratio,
		V_separation_ratio=v_sep_ratio,
		ham_lib='quspin',
		physical_params=physical_params,
		model_bc='periodic',
		int_cluster_bc='periodic',
		super_cluster_bc='periodic',
		solver_method='sparse_ED'
	)


	system_expectations,cluster_expectations=get_general_expectations(run_config)

	